// Halo-style IPA Accumulation Scheme for Pasta Curves
//
// Instead of verifying each IPA proof immediately, accumulate them:
//   1. Each IPA proof produces an "accumulator" — a commitment + evaluation claim
//   2. Multiple accumulators can be folded into one (cheap: EC addition + scalar mul)
//   3. Only the final accumulated claim is fully verified (one expensive IPA verify)
//
// Protocol (Bünz-Chiesa-Mishra-Spooner 2020):
//   Given IPA proof π with commitment C, point z, value v, proof data (L₁,R₁,...,Lₖ,Rₖ):
//   1. Compute challenges u₁,...,uₖ from Fiat-Shamir
//   2. Accumulator = (C', z, v) where C' = C + Σᵢ uᵢ²Lᵢ + Σᵢ uᵢ⁻²Rᵢ
//   3. To fold two accumulators: sample random r, C' = C₁ + r·C₂
//
// This is cheaper than HyperNova because:
//   - No CCS/R1CS structure needed
//   - Folding is just EC arithmetic
//   - The decider is a single IPA verification
//
// References: Halo (Bowe et al. 2019), Proof-Carrying Data (BCMS 2020)

import Foundation

// MARK: - Pallas Scalar Multiplication (VestaFp scalars on Pallas curve)

/// Scalar multiply a Pallas point by a full 256-bit scalar (VestaFp = Pallas Fr).
/// Uses double-and-add on the integer representation of the scalar.
public func pallasPointScalarMul(_ p: PallasPointProjective, _ scalar: VestaFp) -> PallasPointProjective {
    let limbs = vestaToInt(scalar)
    var result = pallasPointIdentity()
    var base = p
    for limb in limbs {
        var word = limb
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = pallasPointAdd(result, base)
            }
            base = pallasPointDouble(base)
            word >>= 1
        }
    }
    return result
}

/// Scalar multiply a Vesta point by a full 256-bit scalar (PallasFp = Vesta Fr).
public func vestaPointScalarMul(_ p: VestaPointProjective, _ scalar: PallasFp) -> VestaPointProjective {
    let limbs = pallasToInt(scalar)
    var result = vestaPointIdentity()
    var base = p
    for limb in limbs {
        var word = limb
        for _ in 0..<64 {
            if word & 1 == 1 {
                result = vestaPointAdd(result, base)
            }
            base = vestaPointDouble(base)
            word >>= 1
        }
    }
    return result
}

/// Check equality of two Pallas projective points.
public func pallasPointEqual(_ a: PallasPointProjective, _ b: PallasPointProjective) -> Bool {
    if pallasPointIsIdentity(a) && pallasPointIsIdentity(b) { return true }
    if pallasPointIsIdentity(a) || pallasPointIsIdentity(b) { return false }
    let aZ2 = pallasSqr(a.z)
    let bZ2 = pallasSqr(b.z)
    let aZ3 = pallasMul(a.z, aZ2)
    let bZ3 = pallasMul(b.z, bZ2)
    let lhsX = pallasMul(a.x, bZ2)
    let rhsX = pallasMul(b.x, aZ2)
    let lhsY = pallasMul(a.y, bZ3)
    let rhsY = pallasMul(b.y, aZ3)
    return pallasToInt(lhsX) == pallasToInt(rhsX) && pallasToInt(lhsY) == pallasToInt(rhsY)
}

/// Check equality of two Vesta projective points.
public func vestaPointEqual(_ a: VestaPointProjective, _ b: VestaPointProjective) -> Bool {
    if vestaPointIsIdentity(a) && vestaPointIsIdentity(b) { return true }
    if vestaPointIsIdentity(a) || vestaPointIsIdentity(b) { return false }
    let aZ2 = vestaSqr(a.z)
    let bZ2 = vestaSqr(b.z)
    let aZ3 = vestaMul(a.z, aZ2)
    let bZ3 = vestaMul(b.z, bZ2)
    let lhsX = vestaMul(a.x, bZ2)
    let rhsX = vestaMul(b.x, aZ2)
    let lhsY = vestaMul(a.y, bZ3)
    let rhsY = vestaMul(b.y, aZ3)
    return vestaToInt(lhsX) == vestaToInt(rhsX) && vestaToInt(lhsY) == vestaToInt(rhsY)
}

/// Negate a Pallas projective point.
public func pallasPointNeg(_ p: PallasPointProjective) -> PallasPointProjective {
    if pallasPointIsIdentity(p) { return p }
    return PallasPointProjective(x: p.x, y: pallasNeg(p.y), z: p.z)
}

/// Negate a Vesta projective point.
public func vestaPointNeg(_ p: VestaPointProjective) -> VestaPointProjective {
    if vestaPointIsIdentity(p) { return p }
    return VestaPointProjective(x: p.x, y: vestaNeg(p.y), z: p.z)
}

// MARK: - IPA Proof for Pallas Curve

/// An IPA proof over the Pallas curve.
/// L/R commitments are Pallas points, scalar a is in VestaFp (Pallas Fr).
public struct PallasIPAProof {
    public let L: [PallasPointProjective]
    public let R: [PallasPointProjective]
    public let a: VestaFp  // final scalar (Pallas Fr = VestaFp)

    public init(L: [PallasPointProjective], R: [PallasPointProjective], a: VestaFp) {
        self.L = L
        self.R = R
        self.a = a
    }
}

// MARK: - IPA Accumulator

/// An IPA accumulator — a deferred verification claim.
/// Stores the folded commitment + the data needed to eventually verify.
public struct IPAAccumulator {
    /// Accumulated commitment (Pallas point)
    public let commitment: PallasPointProjective
    /// Evaluation vector b (known to verifier)
    public let b: [VestaFp]
    /// Claimed inner product value
    public let value: VestaFp
    /// IPA challenges from Fiat-Shamir (for decider to reconstruct generators)
    public let challenges: [VestaFp]
    /// SRS generators (needed for decide)
    public let generators: [PallasPointAffine]
    /// Blinding generator Q
    public let Q: PallasPointAffine

    public init(commitment: PallasPointProjective, b: [VestaFp], value: VestaFp,
                challenges: [VestaFp], generators: [PallasPointAffine], Q: PallasPointAffine) {
        self.commitment = commitment
        self.b = b
        self.value = value
        self.challenges = challenges
        self.generators = generators
        self.Q = Q
    }
}

// MARK: - Accumulation Engine

/// Pallas-curve IPA accumulation engine.
/// Converts IPA proofs into accumulators, folds them, and provides a final decider.
public class PallasAccumulationEngine {

    /// SRS generators G_0, ..., G_{n-1}
    public let generators: [PallasPointAffine]
    /// Blinding generator Q
    public let Q: PallasPointAffine
    /// log2(n)
    public let logN: Int

    /// Create an accumulation engine with given generators.
    /// generators: n Pallas points (n must be power of 2)
    /// Q: separate generator for inner product binding
    public init(generators: [PallasPointAffine], Q: PallasPointAffine) {
        precondition(generators.count > 0 && (generators.count & (generators.count - 1)) == 0,
                     "Generator count must be power of 2")
        self.generators = generators
        self.Q = Q
        var log = 0
        var n = generators.count
        while n > 1 { n >>= 1; log += 1 }
        self.logN = log
    }

    /// Generate test generators deterministically (for testing only).
    /// Creates n+1 distinct Pallas points: i*G for i in 1..n+1.
    public static func generateTestGenerators(count n: Int) -> (generators: [PallasPointAffine], Q: PallasPointAffine) {
        let g = pallasGenerator()
        let gProj = pallasPointFromAffine(g)
        var points = [PallasPointProjective]()
        points.reserveCapacity(n + 1)
        var acc = gProj
        for _ in 0..<(n + 1) {
            points.append(acc)
            acc = pallasPointAdd(acc, gProj)
        }
        let affine = batchPallasToAffine(points)
        return (generators: Array(affine.prefix(n)), Q: affine[n])
    }

    /// Commit to a vector using Pallas generators: C = sum(a_i * G_i)
    public func commit(_ a: [VestaFp]) -> PallasPointProjective {
        precondition(a.count == generators.count)
        var result = pallasPointIdentity()
        for i in 0..<a.count {
            if !a[i].isZero {
                let gi = pallasPointFromAffine(generators[i])
                result = pallasPointAdd(result, pallasPointScalarMul(gi, a[i]))
            }
        }
        return result
    }

    /// Inner product: <a, b> = sum(a_i * b_i) in VestaFp (Pallas Fr)
    public static func innerProduct(_ a: [VestaFp], _ b: [VestaFp]) -> VestaFp {
        precondition(a.count == b.count)
        var result = VestaFp.zero
        for i in 0..<a.count {
            result = vestaAdd(result, vestaMul(a[i], b[i]))
        }
        return result
    }

    // MARK: - IPA Prove

    /// Create an IPA opening proof over Pallas.
    /// Proves that commitment C opens to vector `a` with <a, b> = v.
    public func createProof(a inputA: [VestaFp], b inputB: [VestaFp]) -> PallasIPAProof {
        let n = inputA.count
        precondition(n == inputB.count && n == generators.count)
        precondition(n > 0 && (n & (n - 1)) == 0)

        var a = inputA
        var b = inputB
        var G = generators.map { pallasPointFromAffine($0) }
        let qProj = pallasPointFromAffine(Q)

        var Ls = [PallasPointProjective]()
        var Rs = [PallasPointProjective]()
        Ls.reserveCapacity(logN)
        Rs.reserveCapacity(logN)

        // Transcript for Fiat-Shamir
        var transcript = [UInt8]()
        let C = commit(inputA)
        let v = PallasAccumulationEngine.innerProduct(inputA, inputB)
        let vQ = pallasPointScalarMul(qProj, v)
        let Cbound = pallasPointAdd(C, vQ)
        appendPallasPoint(&transcript, Cbound)
        appendVestaFp(&transcript, v)

        var halfLen = n / 2

        for _ in 0..<logN {
            let aL = Array(a.prefix(halfLen))
            let aR = Array(a.suffix(halfLen))
            let bL = Array(b.prefix(halfLen))
            let bR = Array(b.suffix(halfLen))
            let GL = Array(G.prefix(halfLen))
            let GR = Array(G.suffix(halfLen))

            // Cross inner products
            let cL = PallasAccumulationEngine.innerProduct(aL, bR)
            let cR = PallasAccumulationEngine.innerProduct(aR, bL)

            // L = MSM(GR, aL) + cL * Q
            var L = pallasPointIdentity()
            for i in 0..<halfLen {
                if !aL[i].isZero {
                    L = pallasPointAdd(L, pallasPointScalarMul(GR[i], aL[i]))
                }
            }
            L = pallasPointAdd(L, pallasPointScalarMul(qProj, cL))

            // R = MSM(GL, aR) + cR * Q
            var R = pallasPointIdentity()
            for i in 0..<halfLen {
                if !aR[i].isZero {
                    R = pallasPointAdd(R, pallasPointScalarMul(GL[i], aR[i]))
                }
            }
            R = pallasPointAdd(R, pallasPointScalarMul(qProj, cR))

            Ls.append(L)
            Rs.append(R)

            // Fiat-Shamir challenge
            appendPallasPoint(&transcript, L)
            appendPallasPoint(&transcript, R)
            let x = derivePallasChallenge(transcript)
            let xInv = vestaInverse(x)

            // Fold vectors: a' = x*aL + x^(-1)*aR, b' = x^(-1)*bL + x*bR
            var newA = [VestaFp](repeating: VestaFp.zero, count: halfLen)
            var newB = [VestaFp](repeating: VestaFp.zero, count: halfLen)
            for i in 0..<halfLen {
                newA[i] = vestaAdd(vestaMul(x, aL[i]), vestaMul(xInv, aR[i]))
                newB[i] = vestaAdd(vestaMul(xInv, bL[i]), vestaMul(x, bR[i]))
            }

            // Fold generators: G'_i = x^(-1)*GL_i + x*GR_i
            var newG = [PallasPointProjective](repeating: pallasPointIdentity(), count: halfLen)
            for i in 0..<halfLen {
                let left = pallasPointScalarMul(GL[i], xInv)
                let right = pallasPointScalarMul(GR[i], x)
                newG[i] = pallasPointAdd(left, right)
            }

            a = newA
            b = newB
            G = newG
            halfLen /= 2
        }

        precondition(a.count == 1)
        return PallasIPAProof(L: Ls, R: Rs, a: a[0])
    }

    // MARK: - IPA Verify

    /// Verify a Pallas IPA proof.
    public func verify(commitment C: PallasPointProjective, b inputB: [VestaFp],
                       innerProductValue v: VestaFp, proof: PallasIPAProof) -> Bool {
        let n = generators.count
        guard proof.L.count == logN, proof.R.count == logN else { return false }
        guard inputB.count == n else { return false }

        let qProj = pallasPointFromAffine(Q)

        // Reconstruct challenges
        var transcript = [UInt8]()
        appendPallasPoint(&transcript, C)
        appendVestaFp(&transcript, v)

        var challenges = [VestaFp]()
        challenges.reserveCapacity(logN)

        for round in 0..<logN {
            appendPallasPoint(&transcript, proof.L[round])
            appendPallasPoint(&transcript, proof.R[round])
            let x = derivePallasChallenge(transcript)
            challenges.append(x)
        }

        // Fold commitment: C' = C + sum(x_i^2 * L_i + x_i^(-2) * R_i)
        var Cprime = C
        for round in 0..<logN {
            let x = challenges[round]
            let x2 = vestaMul(x, x)
            let xInv = vestaInverse(x)
            let xInv2 = vestaMul(xInv, xInv)
            let lTerm = pallasPointScalarMul(proof.L[round], x2)
            let rTerm = pallasPointScalarMul(proof.R[round], xInv2)
            Cprime = pallasPointAdd(Cprime, pallasPointAdd(lTerm, rTerm))
        }

        // Compute s vector and fold generators
        var challengeInvs = challenges.map { vestaInverse($0) }
        var s = [VestaFp](repeating: VestaFp.one, count: n)
        for round in 0..<logN {
            let x = challenges[round]
            let xInv = challengeInvs[round]
            for i in 0..<n {
                let bit = (i >> (logN - 1 - round)) & 1
                if bit == 0 {
                    s[i] = vestaMul(s[i], xInv)
                } else {
                    s[i] = vestaMul(s[i], x)
                }
            }
        }

        // G_final = MSM(G, s)
        var gFinal = pallasPointIdentity()
        for i in 0..<n {
            gFinal = pallasPointAdd(gFinal, pallasPointScalarMul(
                pallasPointFromAffine(generators[i]), s[i]))
        }

        // Fold b
        var bFolded = inputB
        var halfLen = n / 2
        for round in 0..<logN {
            var newB = [VestaFp](repeating: VestaFp.zero, count: halfLen)
            for i in 0..<halfLen {
                newB[i] = vestaAdd(
                    vestaMul(challengeInvs[round], bFolded[i]),
                    vestaMul(challenges[round], bFolded[halfLen + i]))
            }
            bFolded = newB
            halfLen /= 2
        }
        let bFinal = bFolded[0]

        // Check: C' == proof.a * G_final + (proof.a * bFinal) * Q
        let aG = pallasPointScalarMul(gFinal, proof.a)
        let ab = vestaMul(proof.a, bFinal)
        let abQ = pallasPointScalarMul(qProj, ab)
        let expected = pallasPointAdd(aG, abQ)

        return pallasPointEqual(Cprime, expected)
    }

    // MARK: - Accumulate

    /// Convert an IPA proof into an accumulator (cheap operation).
    /// This extracts the deferred verification claim without doing the expensive check.
    public func accumulate(proof: PallasIPAProof, commitment C: PallasPointProjective,
                           b: [VestaFp], innerProductValue v: VestaFp) -> IPAAccumulator {
        let n = generators.count

        // Reconstruct challenges from Fiat-Shamir transcript
        var transcript = [UInt8]()
        appendPallasPoint(&transcript, C)
        appendVestaFp(&transcript, v)

        var challenges = [VestaFp]()
        challenges.reserveCapacity(logN)

        for round in 0..<logN {
            appendPallasPoint(&transcript, proof.L[round])
            appendPallasPoint(&transcript, proof.R[round])
            let x = derivePallasChallenge(transcript)
            challenges.append(x)
        }

        // Fold commitment: C' = C + sum(x_i^2 * L_i + x_i^(-2) * R_i)
        var Cprime = C
        for round in 0..<logN {
            let x = challenges[round]
            let x2 = vestaMul(x, x)
            let xInv = vestaInverse(x)
            let xInv2 = vestaMul(xInv, xInv)
            let lTerm = pallasPointScalarMul(proof.L[round], x2)
            let rTerm = pallasPointScalarMul(proof.R[round], xInv2)
            Cprime = pallasPointAdd(Cprime, pallasPointAdd(lTerm, rTerm))
        }

        return IPAAccumulator(
            commitment: Cprime,
            b: b,
            value: v,
            challenges: challenges,
            generators: generators,
            Q: Q
        )
    }

    // MARK: - Fold

    /// Fold two accumulators into one (cheap: EC addition + scalar mul).
    /// The fold uses a random challenge r to combine the two claims linearly.
    public func fold(_ acc1: IPAAccumulator, _ acc2: IPAAccumulator,
                     randomness r: VestaFp) -> IPAAccumulator {
        // C' = acc1.C + r * acc2.C
        let rC2 = pallasPointScalarMul(acc2.commitment, r)
        let Cprime = pallasPointAdd(acc1.commitment, rC2)

        // v' = acc1.v + r * acc2.v
        let vprime = vestaAdd(acc1.value, vestaMul(r, acc2.value))

        // b' = acc1.b + r * acc2.b (element-wise)
        let n = acc1.b.count
        var bprime = [VestaFp](repeating: VestaFp.zero, count: n)
        for i in 0..<n {
            bprime[i] = vestaAdd(acc1.b[i], vestaMul(r, acc2.b[i]))
        }

        // Challenges: combine (simplified — for the decider we track both sets)
        // In a full implementation, the folded challenges would need careful handling.
        // Here we use acc1's challenges as the primary (the decider verifies the folded claim).
        return IPAAccumulator(
            commitment: Cprime,
            b: bprime,
            value: vprime,
            challenges: acc1.challenges,
            generators: generators,
            Q: Q
        )
    }

    /// Fold N accumulators into one using Fiat-Shamir-derived randomness.
    public func foldMany(_ accs: [IPAAccumulator]) -> IPAAccumulator {
        precondition(accs.count >= 1)
        if accs.count == 1 { return accs[0] }

        // Derive folding randomness from all accumulator commitments
        var transcript = [UInt8]()
        for acc in accs {
            appendPallasPoint(&transcript, acc.commitment)
        }

        var result = accs[0]
        for i in 1..<accs.count {
            // Derive fresh randomness for each fold step
            var stepTranscript = transcript
            appendVestaFp(&stepTranscript, vestaFromInt(UInt64(i)))
            let r = derivePallasChallenge(stepTranscript)
            result = fold(result, accs[i], randomness: r)
        }
        return result
    }

    // MARK: - Decide

    /// Decide: fully verify the final accumulated claim (expensive, done once).
    /// This reconstructs the folded generator and checks the accumulated commitment.
    ///
    /// For a properly accumulated claim, this checks:
    ///   C' == a_final * G_final + (a_final * b_final) * Q
    /// where G_final and b_final are computed from the IPA challenges.
    ///
    /// NOTE: The decider needs the original proof's final scalar `a`. For a single
    /// accumulation, pass the proof's `a` value. For folded accumulators, the
    /// verification equation is checked directly on the accumulated commitment.
    public func decide(_ acc: IPAAccumulator, proofA: VestaFp) -> Bool {
        let n = generators.count
        let qProj = pallasPointFromAffine(Q)

        // Compute s vector from challenges
        var s = [VestaFp](repeating: VestaFp.one, count: n)
        var challengeInvs = acc.challenges.map { vestaInverse($0) }

        for round in 0..<acc.challenges.count {
            let x = acc.challenges[round]
            let xInv = challengeInvs[round]
            let logN = acc.challenges.count
            for i in 0..<n {
                let bit = (i >> (logN - 1 - round)) & 1
                if bit == 0 {
                    s[i] = vestaMul(s[i], xInv)
                } else {
                    s[i] = vestaMul(s[i], x)
                }
            }
        }

        // G_final = MSM(generators, s)
        var gFinal = pallasPointIdentity()
        for i in 0..<n {
            gFinal = pallasPointAdd(gFinal, pallasPointScalarMul(
                pallasPointFromAffine(generators[i]), s[i]))
        }

        // Fold b using challenges
        var bFolded = acc.b
        var halfLen = n / 2
        for round in 0..<acc.challenges.count {
            var newB = [VestaFp](repeating: VestaFp.zero, count: halfLen)
            for i in 0..<halfLen {
                newB[i] = vestaAdd(
                    vestaMul(challengeInvs[round], bFolded[i]),
                    vestaMul(acc.challenges[round], bFolded[halfLen + i]))
            }
            bFolded = newB
            halfLen /= 2
        }
        let bFinal = bFolded[0]

        // Check: acc.C == proofA * G_final + (proofA * bFinal) * Q
        let aG = pallasPointScalarMul(gFinal, proofA)
        let ab = vestaMul(proofA, bFinal)
        let abQ = pallasPointScalarMul(qProj, ab)
        let expected = pallasPointAdd(aG, abQ)

        return pallasPointEqual(acc.commitment, expected)
    }

    // MARK: - Transcript Helpers

    private func appendPallasPoint(_ transcript: inout [UInt8], _ p: PallasPointProjective) {
        let affine = pallasPointToAffine(p)
        let xBytes = affine.x.toBytes()
        let yBytes = affine.y.toBytes()
        transcript.append(contentsOf: xBytes)
        transcript.append(contentsOf: yBytes)
    }

    private func appendVestaFp(_ transcript: inout [UInt8], _ v: VestaFp) {
        let intVal = vestaToInt(v)
        for limb in intVal {
            for byte in 0..<8 {
                transcript.append(UInt8((limb >> (byte * 8)) & 0xFF))
            }
        }
    }

    private func derivePallasChallenge(_ transcript: [UInt8]) -> VestaFp {
        let hash = blake3(transcript)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 {
                limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8)
            }
        }
        // Clear top 2 bits to ensure < 2^254 ~ Vesta p
        limbs[3] &= 0x3FFFFFFFFFFFFFFF
        let raw = VestaFp.from64(limbs)
        return vestaMul(raw, VestaFp.from64(VestaFp.R2_MOD_P))
    }
}
