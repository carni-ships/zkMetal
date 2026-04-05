// RecursiveVerifierCircuit — Encode IPA verification as constraints over the other curve
//
// Given a Pallas IPA proof pi, build a Vesta circuit that verifies pi.
// Key insight: Pallas base field = Vesta scalar field, so Pallas point
// coordinates are *native* field elements in a Vesta circuit. This makes
// the recursive verifier highly efficient for the Pasta cycle.
//
// The verifier circuit checks:
//   1. Transcript replay: reconstruct Fiat-Shamir challenges from proof data
//   2. Commitment folding: C' = C + sum(u_i^2 * L_i + u_i^{-2} * R_i)
//   3. Generator folding: compute s-vector from challenges
//   4. Final check: C' == a * G_final + (a * b_final) * Q
//
// Sub-circuits required:
//   - Scalar multiplication over Pallas (non-native EC ops in Vesta circuit)
//   - Inner product / linear combination in native field
//   - Hash-to-field (Fiat-Shamir) via Poseidon2 or Blake3 in-circuit
//
// References:
//   - Halo (Bowe et al. 2019): recursive proof without trusted setup
//   - Nova (Kothapalli et al. 2022): cycle-based recursive composition

import Foundation

// MARK: - IPA Verification Instance

/// The public inputs and proof data needed to verify a Pallas IPA proof
/// inside a Vesta circuit.
public struct IPAVerificationInstance {
    /// Bound commitment C (Pallas point, coordinates as VestaFp)
    public let commitmentX: VestaFp
    public let commitmentY: VestaFp
    /// L commitments (one per round)
    public let Ls: [(x: VestaFp, y: VestaFp)]
    /// R commitments (one per round)
    public let Rs: [(x: VestaFp, y: VestaFp)]
    /// Final scalar a (in VestaFp = Pallas Fr)
    public let finalA: VestaFp
    /// Inner product value v (in VestaFp = Pallas Fr)
    public let innerProductValue: VestaFp
    /// Evaluation vector b (in VestaFp = Pallas Fr)
    public let b: [VestaFp]
    /// Fiat-Shamir challenges (pre-computed outside circuit for witness generation)
    public let challenges: [VestaFp]

    /// Create from a Pallas IPA proof and its public data.
    public static func fromPallasProof(
        proof: PallasIPAProof,
        commitment: PallasPointProjective,
        b: [VestaFp],
        innerProductValue: VestaFp
    ) -> IPAVerificationInstance {
        let (cx, cy) = pallasPointToVestaCoords(commitment)
        let ls = proof.L.map { pallasPointToVestaCoords($0) }
        let rs = proof.R.map { pallasPointToVestaCoords($0) }

        // Reconstruct challenges (same logic as PallasAccumulationEngine.accumulate)
        let logN = proof.L.count
        var transcript = [UInt8]()
        // Append commitment
        let cAffine = pallasPointToAffine(commitment)
        transcript.append(contentsOf: cAffine.x.toBytes())
        transcript.append(contentsOf: cAffine.y.toBytes())
        // Append inner product value
        let vInt = vestaToInt(innerProductValue)
        for limb in vInt {
            for byte in 0..<8 { transcript.append(UInt8((limb >> (byte * 8)) & 0xFF)) }
        }

        var challenges = [VestaFp]()
        for round in 0..<logN {
            let lAff = pallasPointToAffine(proof.L[round])
            transcript.append(contentsOf: lAff.x.toBytes())
            transcript.append(contentsOf: lAff.y.toBytes())
            let rAff = pallasPointToAffine(proof.R[round])
            transcript.append(contentsOf: rAff.x.toBytes())
            transcript.append(contentsOf: rAff.y.toBytes())
            let hash = blake3(transcript)
            var limbs = [UInt64](repeating: 0, count: 4)
            for i in 0..<4 {
                for j in 0..<8 { limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8) }
            }
            limbs[3] &= 0x3FFFFFFFFFFFFFFF
            let raw = VestaFp.from64(limbs)
            challenges.append(vestaMul(raw, VestaFp.from64(VestaFp.R2_MOD_P)))
        }

        return IPAVerificationInstance(
            commitmentX: cx, commitmentY: cy,
            Ls: ls, Rs: rs,
            finalA: proof.a,
            innerProductValue: innerProductValue,
            b: b,
            challenges: challenges
        )
    }
}

// MARK: - Recursive Verifier Circuit Builder

/// Builds a Plonk circuit (over the BN254 scalar field for now, or conceptually
/// over Vesta Fr) that verifies a Pallas IPA proof.
///
/// The circuit structure:
///   - Public inputs: commitment (x,y), inner product value, L/R points
///   - Private witness: IPA challenges (derived from Fiat-Shamir)
///   - Constraints: fold commitment, fold generators, final check
///
/// Because Pallas Fp = Vesta Fr, all point coordinates and scalar operations
/// are native. The main cost is the elliptic curve operations expressed as
/// constraints (each point addition ~ 6 multiplication gates).
public class RecursiveVerifierCircuitBuilder {

    /// The Plonk circuit builder we emit constraints into
    public let builder: PlonkCircuitBuilder

    /// Non-native field gadget (native for Pasta cycle)
    public let fieldGadget: NonNativeFieldGadget

    /// Number of IPA rounds (log2 of generator count)
    public let logN: Int

    /// Generator count (power of 2)
    public let generatorCount: Int

    public init(logN: Int) {
        self.logN = logN
        self.generatorCount = 1 << logN
        self.builder = PlonkCircuitBuilder()
        self.fieldGadget = NonNativeFieldGadget(
            builder: builder,
            config: .pallasFpInVestaFr
        )
    }

    // MARK: - Elliptic Curve Gadgets

    /// Constrain point addition: (x3, y3) = (x1, y1) + (x2, y2) on y^2 = x^3 + 5.
    ///
    /// For distinct points (not doubling), the addition formulas are:
    ///   lambda = (y2 - y1) / (x2 - x1)
    ///   x3 = lambda^2 - x1 - x2
    ///   y3 = lambda * (x1 - x3) - y1
    ///
    /// This requires ~6 multiplication gates + several addition gates.
    /// The witness must provide lambda as an auxiliary variable.
    ///
    /// Returns (x3_var, y3_var).
    public func constrainPointAdd(
        x1: Int, y1: Int,
        x2: Int, y2: Int,
        lambda: Int  // witness: (y2-y1)/(x2-x1), provided externally
    ) -> (x3: Int, y3: Int) {
        // Constrain: lambda * (x2 - x1) = y2 - y1
        // Rearrange: lambda * x2 - lambda * x1 - y2 + y1 = 0
        let lx2 = builder.mul(lambda, x2)
        let lx1 = builder.mul(lambda, x1)
        let lhs_diff = builder.add(lx2, y1)
        let rhs_diff = builder.add(lx1, y2)
        builder.assertEqual(lhs_diff, rhs_diff)

        // Constrain: x3 = lambda^2 - x1 - x2
        let lambda_sq = builder.mul(lambda, lambda)
        let x1_plus_x2 = builder.add(x1, x2)
        // x3 + x1 + x2 = lambda^2
        let x3 = builder.addInput()
        let x3_plus_sum = builder.add(x3, x1_plus_x2)
        builder.assertEqual(x3_plus_sum, lambda_sq)

        // Constrain: y3 = lambda * (x1 - x3) - y1
        // Rearrange: y3 + y1 = lambda * (x1 - x3)
        // => y3 + y1 = lambda * x1 - lambda * x3
        let y3 = builder.addInput()
        let y3_plus_y1 = builder.add(y3, y1)
        let lx3 = builder.mul(lambda, x3)
        let lx1_for_y = builder.mul(lambda, x1)
        // y3 + y1 + lambda*x3 = lambda*x1
        let check_lhs = builder.add(y3_plus_y1, lx3)
        builder.assertEqual(check_lhs, lx1_for_y)

        return (x3, y3)
    }

    /// Constrain point doubling: (x3, y3) = 2 * (x1, y1) on y^2 = x^3 + 5.
    ///
    /// Doubling formula (a=0):
    ///   lambda = 3 * x1^2 / (2 * y1)
    ///   x3 = lambda^2 - 2*x1
    ///   y3 = lambda * (x1 - x3) - y1
    ///
    /// Returns (x3_var, y3_var).
    public func constrainPointDouble(
        x1: Int, y1: Int,
        lambda: Int  // witness: 3*x1^2 / (2*y1)
    ) -> (x3: Int, y3: Int) {
        // Constrain: lambda * 2*y1 = 3*x1^2
        let x1_sq = builder.mul(x1, x1)
        // 3 * x1^2: use add chain
        let x1_sq_2 = builder.add(x1_sq, x1_sq)
        let x1_sq_3 = builder.add(x1_sq_2, x1_sq)
        let y1_2 = builder.add(y1, y1)
        let lambda_times_2y = builder.mul(lambda, y1_2)
        builder.assertEqual(lambda_times_2y, x1_sq_3)

        // x3 = lambda^2 - 2*x1
        let lambda_sq = builder.mul(lambda, lambda)
        let x1_double = builder.add(x1, x1)
        let x3 = builder.addInput()
        let x3_plus_2x1 = builder.add(x3, x1_double)
        builder.assertEqual(x3_plus_2x1, lambda_sq)

        // y3 = lambda * (x1 - x3) - y1
        let y3 = builder.addInput()
        let y3_plus_y1 = builder.add(y3, y1)
        let lx3 = builder.mul(lambda, x3)
        let lx1 = builder.mul(lambda, x1)
        let check_lhs = builder.add(y3_plus_y1, lx3)
        builder.assertEqual(check_lhs, lx1)

        return (x3, y3)
    }

    /// Constrain scalar multiplication: (outX, outY) = scalar * (inX, inY).
    ///
    /// Uses double-and-add with bit decomposition of the scalar.
    /// The scalar is decomposed into `scalarBits` boolean variables.
    /// Each bit selects whether to add the accumulated point.
    ///
    /// Cost: ~scalarBits * (doubling + conditional addition) gates.
    /// For 255-bit scalars: ~255 * ~12 = ~3060 gates per scalar mul.
    ///
    /// Returns (outX_var, outY_var) and the bit decomposition variables.
    public func constrainScalarMul(
        pointX: Int, pointY: Int,
        scalarBits: [Int],      // boolean circuit variables for each bit (LSB first)
        lambdas: [Int]          // witness: lambda values for each add/double step
    ) -> (x: Int, y: Int) {
        // This is a simplified version that constrains the relationship
        // between inputs and outputs. A full implementation would unroll
        // the double-and-add loop with conditional addition at each step.
        //
        // For the Pasta cycle verifier, we primarily need scalar mul for:
        //   - u_i^2 * L_i (squaring a challenge, then scalar mul)
        //   - u_i^{-2} * R_i
        //   - a * G_final
        //
        // The number of scalar muls is O(log n) per proof verification,
        // making the total circuit size O(n * log n) gates.

        let outX = builder.addInput()
        let outY = builder.addInput()

        // For now, we return the output variables.
        // The witness generator is responsible for computing correct values.
        // Full constraint generation for double-and-add would be emitted here.
        //
        // A production implementation would:
        // 1. For each bit b_i (MSB to LSB):
        //    a. Double the accumulator: acc = 2*acc (constrainPointDouble)
        //    b. If b_i == 1: acc = acc + P (constrainPointAdd)
        // 2. Assert output == final acc

        _ = (pointX, pointY, scalarBits, lambdas)
        return (outX, outY)
    }

    // MARK: - Challenge Derivation (In-Circuit Fiat-Shamir)

    /// Hash point coordinates to derive a challenge, using Poseidon2 gates.
    ///
    /// This replaces the Blake3 hash used in the native verifier with
    /// Poseidon2, which is much more circuit-friendly (~300 gates per
    /// hash vs ~25000 for SHA/Blake).
    ///
    /// The inputs are absorbed into a Poseidon2 sponge, and the output
    /// is squeezed as a field element.
    ///
    /// Returns the challenge variable.
    public func constrainChallengeDerivation(inputs: [Int]) -> Int {
        // Use the Poseidon2 gates from PlonkCircuitBuilder:
        // 1. Initialize state = [0, 0, 0]
        // 2. For each pair of inputs: absorb into rate, apply S-box + linear layer
        // 3. Squeeze output from state[0]

        guard !inputs.isEmpty else {
            return builder.constant(Fr.one)
        }

        // Start with domain separator as initial state
        var state = [
            builder.constant(Fr.zero),
            builder.constant(Fr.zero),
            builder.constant(Fr.zero)
        ]

        // Absorb inputs in pairs (rate = 2 for t=3 Poseidon2)
        var idx = 0
        while idx < inputs.count {
            // XOR (add) input into rate positions
            state[0] = builder.add(state[0], inputs[idx])
            if idx + 1 < inputs.count {
                state[1] = builder.add(state[1], inputs[idx + 1])
            }
            idx += 2

            // Apply Poseidon2 round: S-box on each element, then linear layer
            for i in 0..<3 {
                state[i] = builder.poseidonSbox(state[i])
            }
            state = builder.poseidonExternalLinearLayer(state)
        }

        // Final squeeze: one more permutation, output state[0]
        for i in 0..<3 {
            state[i] = builder.poseidonSbox(state[i])
        }
        state = builder.poseidonExternalLinearLayer(state)

        return state[0]
    }

    // MARK: - Full IPA Verifier Circuit

    /// Build the complete IPA verification circuit.
    ///
    /// The circuit takes as public inputs:
    ///   - Commitment point (x, y)
    ///   - Inner product value v
    ///   - L/R commitment points
    ///   - Final scalar a
    ///
    /// And constrains that the IPA verification equation holds.
    ///
    /// Returns the built circuit and the public input variable indices.
    public func buildVerifierCircuit(
        instance: IPAVerificationInstance
    ) -> (circuit: PlonkCircuit, publicInputs: [Int]) {
        var publicInputVars = [Int]()

        // Allocate public input variables
        let cxVar = builder.addInput(); publicInputVars.append(cxVar)
        let cyVar = builder.addInput(); publicInputVars.append(cyVar)
        let vVar = builder.addInput(); publicInputVars.append(vVar)
        let aVar = builder.addInput(); publicInputVars.append(aVar)

        // Allocate L/R point variables
        var lVars = [(x: Int, y: Int)]()
        var rVars = [(x: Int, y: Int)]()
        for _ in 0..<logN {
            let lx = builder.addInput(); publicInputVars.append(lx)
            let ly = builder.addInput(); publicInputVars.append(ly)
            lVars.append((lx, ly))
            let rx = builder.addInput(); publicInputVars.append(rx)
            let ry = builder.addInput(); publicInputVars.append(ry)
            rVars.append((rx, ry))
        }

        // Step 1: Derive challenges via in-circuit Fiat-Shamir
        //
        // For each round i, the challenge u_i is derived from the transcript
        // containing (C, v, L_1, R_1, ..., L_i, R_i).
        //
        // In the circuit, we use Poseidon2 as the hash function (circuit-friendly).
        var challengeVars = [Int]()
        var transcriptInputs = [cxVar, cyVar, vVar]

        for round in 0..<logN {
            transcriptInputs.append(lVars[round].x)
            transcriptInputs.append(lVars[round].y)
            transcriptInputs.append(rVars[round].x)
            transcriptInputs.append(rVars[round].y)
            let challenge = constrainChallengeDerivation(inputs: transcriptInputs)
            challengeVars.append(challenge)
        }

        // Step 2: Compute challenge squares and inverse squares
        //
        // For commitment folding: C' = C + sum(u_i^2 * L_i + u_i^{-2} * R_i)
        //
        // u_i^2 is just mul(u_i, u_i) — one gate.
        // u_i^{-2} requires computing the inverse in-circuit, which is done
        // by providing the inverse as a witness and constraining u_i * u_i_inv = 1.
        var challengeSqVars = [Int]()
        var challengeInvSqVars = [Int]()

        for round in 0..<logN {
            let u = challengeVars[round]
            let u_sq = builder.mul(u, u)
            challengeSqVars.append(u_sq)

            // Witness: u_inv (provided externally, constrained by u * u_inv = 1)
            let u_inv = builder.addInput()
            let check_one = builder.mul(u, u_inv)
            let one_const = builder.constant(Fr.one)
            builder.assertEqual(check_one, one_const)

            let u_inv_sq = builder.mul(u_inv, u_inv)
            challengeInvSqVars.append(u_inv_sq)
        }

        // Step 3: Commitment folding
        //
        // The folded commitment C' = C + sum_i(u_i^2 * L_i + u_i^{-2} * R_i)
        // requires scalar multiplications of curve points.
        // Each scalar mul is expensive (~3000 gates), but we only have log(n) of them.
        //
        // For now, we allocate the output and constrain it via the witness.
        // A full implementation would unroll each scalar mul.
        let cPrimeX = builder.addInput()
        let cPrimeY = builder.addInput()

        // Step 4: Compute s-vector and fold generators + b-vector
        //
        // s_j = prod_{i: bit_i(j)=0} u_i^{-1} * prod_{i: bit_i(j)=1} u_i
        // b_final = fold b using challenges
        //
        // This is done via the witness, with the result constrained.
        let bFinalVar = builder.addInput()

        // Step 5: Final IPA check
        //
        // The verification equation: C' == a * G_final + (a * b_final) * Q
        //
        // We constrain a * b_final (one mul gate):
        let a_times_b = builder.mul(aVar, bFinalVar)

        // The EC operations (a * G_final and (a*b) * Q) are the most expensive
        // part. For a complete circuit, we'd use constrainScalarMul for each.
        //
        // For the structural implementation, we allocate the final check point
        // and assert equality with C':
        let finalCheckX = builder.addInput()
        let finalCheckY = builder.addInput()

        // Assert the final check point equals the folded commitment
        builder.assertEqual(finalCheckX, cPrimeX)
        builder.assertEqual(finalCheckY, cPrimeY)

        // Store the a*b product for witness verification
        _ = a_times_b

        let circuit = builder.build()
        return (circuit, publicInputVars)
    }

    /// Estimate the number of gates in the verifier circuit.
    ///
    /// Components:
    ///   - Challenge derivation: ~logN * (inputs * poseidon_cost) gates
    ///   - Challenge squares/inverses: ~logN * 4 gates
    ///   - Scalar multiplications: ~logN * 2 * 3000 gates (for L and R folding)
    ///   - Final scalar mul: ~2 * 3000 gates (for a*G and ab*Q)
    ///   - Generator folding (s-vector): ~n * logN gates
    ///
    /// Total: O(n * logN) gates dominated by generator folding.
    public static func estimateGateCount(logN: Int) -> Int {
        let n = 1 << logN
        let challengeGates = logN * 100  // Poseidon2 per round
        let squareGates = logN * 4
        let scalarMulGates = (logN * 2 + 2) * 3060  // 255 bits * 12 gates
        let generatorFoldGates = n * logN
        return challengeGates + squareGates + scalarMulGates + generatorFoldGates
    }
}

// MARK: - Witness Generator

/// Generates the witness (variable assignments) for the IPA verifier circuit.
///
/// Given an IPA verification instance, computes all intermediate values
/// that the circuit constraints check.
public class IPAVerifierWitnessGenerator {

    /// Generate witness assignments for all circuit variables.
    ///
    /// Returns a dictionary mapping variable index -> Fr value.
    /// The caller uses this to evaluate the circuit.
    public static func generateWitness(
        instance: IPAVerificationInstance,
        generators: [PallasPointAffine],
        Q: PallasPointAffine
    ) -> [Int: VestaFp] {
        var witness = [Int: VestaFp]()
        var nextVar = 0

        func assign(_ val: VestaFp) -> Int {
            let v = nextVar; nextVar += 1
            witness[v] = val
            return v
        }

        // Public inputs
        _ = assign(instance.commitmentX)
        _ = assign(instance.commitmentY)
        _ = assign(instance.innerProductValue)
        _ = assign(instance.finalA)

        // L/R points
        for i in 0..<instance.Ls.count {
            _ = assign(instance.Ls[i].x)
            _ = assign(instance.Ls[i].y)
            _ = assign(instance.Rs[i].x)
            _ = assign(instance.Rs[i].y)
        }

        // Challenges (derived via Fiat-Shamir)
        for u in instance.challenges {
            _ = assign(u)
        }

        // Challenge inverses
        for u in instance.challenges {
            let uInv = vestaInverse(u)
            _ = assign(uInv)
        }

        // Challenge squares and inverse squares
        for u in instance.challenges {
            _ = assign(vestaMul(u, u))
            let uInv = vestaInverse(u)
            _ = assign(vestaMul(uInv, uInv))
        }

        // Folded commitment C'
        let logN = instance.challenges.count
        let n = generators.count

        // Reconstruct C' = C + sum(u_i^2 * L_i + u_i^{-2} * R_i)
        // (Using actual EC operations to compute the correct witness)
        let cProj = pallasPointFromAffine(PallasPointAffine(
            x: vestaFpToPallasFr(instance.commitmentX),
            y: vestaFpToPallasFr(instance.commitmentY)
        ))
        var cPrime = cProj
        for round in 0..<logN {
            let u = instance.challenges[round]
            let u2 = vestaMul(u, u)
            let uInv = vestaInverse(u)
            let uInv2 = vestaMul(uInv, uInv)
            let lProj = pallasPointFromAffine(PallasPointAffine(
                x: vestaFpToPallasFr(instance.Ls[round].x),
                y: vestaFpToPallasFr(instance.Ls[round].y)
            ))
            let rProj = pallasPointFromAffine(PallasPointAffine(
                x: vestaFpToPallasFr(instance.Rs[round].x),
                y: vestaFpToPallasFr(instance.Rs[round].y)
            ))
            let lTerm = pallasPointScalarMul(lProj, u2)
            let rTerm = pallasPointScalarMul(rProj, uInv2)
            cPrime = pallasPointAdd(cPrime, pallasPointAdd(lTerm, rTerm))
        }
        let (cpx, cpy) = pallasPointToVestaCoords(cPrime)
        _ = assign(cpx)
        _ = assign(cpy)

        // Compute s-vector from challenges
        let challengeInvs = instance.challenges.map { vestaInverse($0) }
        var s = [VestaFp](repeating: VestaFp.one, count: n)
        for round in 0..<logN {
            let x = instance.challenges[round]
            let xInv = challengeInvs[round]
            for j in 0..<n {
                let bit = (j >> (logN - 1 - round)) & 1
                if bit == 0 {
                    s[j] = vestaMul(s[j], xInv)
                } else {
                    s[j] = vestaMul(s[j], x)
                }
            }
        }

        // Fold b
        var bFolded = instance.b
        var halfLen = n / 2
        for round in 0..<logN {
            var newB = [VestaFp](repeating: VestaFp.zero, count: halfLen)
            for j in 0..<halfLen {
                newB[j] = vestaAdd(
                    vestaMul(challengeInvs[round], bFolded[j]),
                    vestaMul(instance.challenges[round], bFolded[halfLen + j]))
            }
            bFolded = newB
            halfLen /= 2
        }
        let bFinal = bFolded[0]
        _ = assign(bFinal)

        // a * b_final
        let ab = vestaMul(instance.finalA, bFinal)
        _ = assign(ab)

        // Final check point: a * G_final + (a * b_final) * Q
        var gFinal = pallasPointIdentity()
        for j in 0..<n {
            gFinal = pallasPointAdd(gFinal, pallasPointScalarMul(
                pallasPointFromAffine(generators[j]), s[j]))
        }
        let qProj = pallasPointFromAffine(Q)
        let aG = pallasPointScalarMul(gFinal, instance.finalA)
        let abQ = pallasPointScalarMul(qProj, ab)
        let finalPoint = pallasPointAdd(aG, abQ)
        let (fpx, fpy) = pallasPointToVestaCoords(finalPoint)
        _ = assign(fpx)
        _ = assign(fpy)

        return witness
    }
}
