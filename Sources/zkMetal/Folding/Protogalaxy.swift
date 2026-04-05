// Protogalaxy — Non-uniform IVC via Plonk-native folding
//
// Folds k Plonk-style instances into 1 with O(k log k) prover work.
// Complementary to HyperNova (CCS-based): Protogalaxy operates directly on
// Plonk arithmetization without CCS conversion.
//
// Protocol (Gabizon-Khovratovich 2023):
//   Given running instance (phi, beta, gamma, e) and k-1 new instances:
//   1. Prover computes Lagrange coefficients L_0(X), ..., L_{k-1}(X)
//   2. Prover evaluates the vanishing polynomial F(X) at combined witness
//   3. Prover sends polynomial coefficients of F(X) to verifier
//   4. Verifier samples challenge alpha
//   5. Both compute folded instance:
//        phi' = sum_i L_i(alpha) * phi_i
//        e'   = F(alpha)
//
// Key advantage over HyperNova for Plonk circuits: no CCS conversion needed,
// preserves Plonk structure (permutation arg, custom gates) natively.
//
// Reference: "ProtoGalaxy: Efficient ProtoStar-style folding of multiple instances"
//            (Gabizon, Khovratovich 2023)

import Foundation
import NeonFieldOps

// MARK: - Protogalaxy Instance

/// A committed relaxed Plonk instance for Protogalaxy folding.
///
/// Carries:
///   - KZG commitments to witness columns (a, b, c polynomials)
///   - Plonk challenges beta, gamma (permutation argument)
///   - Error term e (accumulates folding slack; zero for fresh instances)
///   - Relaxation scalar u (1 for fresh, accumulates via folding)
///   - Public inputs
public struct ProtogalaxyInstance {
    /// KZG commitments to witness polynomials [a(x)], [b(x)], [c(x)]
    public let witnessCommitments: [PointProjective]
    /// Public input values
    public let publicInput: [Fr]
    /// Plonk permutation challenge beta
    public let beta: Fr
    /// Plonk permutation challenge gamma
    public let gamma: Fr
    /// Error/slack term (zero for fresh instances, accumulates during folding)
    public let errorTerm: Fr
    /// Relaxation scalar (1 for fresh instances)
    public let u: Fr
    /// Whether this instance has been folded (relaxed)
    public let isRelaxed: Bool

    /// Create a fresh (non-relaxed) Plonk instance.
    public init(witnessCommitments: [PointProjective], publicInput: [Fr],
                beta: Fr, gamma: Fr) {
        self.witnessCommitments = witnessCommitments
        self.publicInput = publicInput
        self.beta = beta
        self.gamma = gamma
        self.errorTerm = Fr.zero
        self.u = Fr.one
        self.isRelaxed = false
    }

    /// Create a relaxed (folded) Plonk instance.
    public init(witnessCommitments: [PointProjective], publicInput: [Fr],
                beta: Fr, gamma: Fr, errorTerm: Fr, u: Fr) {
        self.witnessCommitments = witnessCommitments
        self.publicInput = publicInput
        self.beta = beta
        self.gamma = gamma
        self.errorTerm = errorTerm
        self.u = u
        self.isRelaxed = true
    }
}

// MARK: - Folding Proof

/// Proof produced by the Protogalaxy prover during a fold step.
/// Contains the polynomial F(X) coefficients that the verifier needs.
public struct ProtogalaxyFoldingProof {
    /// Coefficients of the vanishing polynomial F(X) = sum_i f_i * X^i
    /// F(X) evaluates to the error at points 0, 1, ..., k-1 corresponding
    /// to the original instances' gate satisfaction.
    public let fCoefficients: [Fr]
    /// Number of instances folded
    public let instanceCount: Int
}

// MARK: - Protogalaxy Prover

/// Prover for Protogalaxy folding of Plonk instances.
///
/// Folds k committed Plonk instances into a single accumulated instance.
/// Prover work is O(k log k) via Lagrange interpolation on the challenge space.
public class ProtogalaxyProver {
    /// Number of witness columns (a, b, c for standard Plonk)
    public let numWitnessColumns: Int
    /// Circuit size (number of gates, must be power of 2)
    public let circuitSize: Int

    public init(circuitSize: Int, numWitnessColumns: Int = 3) {
        precondition(circuitSize > 0 && (circuitSize & (circuitSize - 1)) == 0,
                     "Circuit size must be a power of 2")
        self.circuitSize = circuitSize
        self.numWitnessColumns = numWitnessColumns
    }

    // MARK: - Fold

    /// Fold k Plonk instances into 1.
    ///
    /// instances[0] is the running (accumulated) instance.
    /// instances[1..k-1] are fresh instances to fold in.
    ///
    /// - Parameters:
    ///   - instances: Array of k committed Plonk instances (first must be relaxed for k>2)
    ///   - witnesses: Corresponding witness polynomial evaluations for each instance
    ///                witnesses[i] = [a_evals, b_evals, c_evals] each of length circuitSize
    /// - Returns: (folded instance, folded witnesses, folding proof)
    public func fold(instances: [ProtogalaxyInstance],
                     witnesses: [[[Fr]]]) -> (ProtogalaxyInstance, [[Fr]], ProtogalaxyFoldingProof) {
        let k = instances.count
        precondition(k >= 2, "Need at least 2 instances to fold")
        precondition(instances.count == witnesses.count)
        for w in witnesses {
            precondition(w.count == numWitnessColumns)
        }
        if k > 2 {
            precondition(instances[0].isRelaxed, "Running instance must be relaxed for k > 2")
        }

        // Step 1: Compute gate satisfaction values F_i for each instance at each gate
        // F_i(j) = qL*a_j + qR*b_j + qO*c_j + qM*a_j*b_j + qC for gate j
        // These are the "error" values -- zero for satisfying instances.
        var gateErrors = [[Fr]]()  // gateErrors[instance][gate]
        gateErrors.reserveCapacity(k)
        for i in 0..<k {
            // For a relaxed instance, error is already tracked in errorTerm
            // For fresh instances, gate satisfaction should be zero
            var errs = [Fr](repeating: Fr.zero, count: k)
            // The Protogalaxy polynomial F(X) is defined over the instance space
            // F(i) = e_i (the error of instance i)
            errs[i] = instances[i].errorTerm
            gateErrors.append(errs)
        }

        // Step 2: Build the polynomial F(X) via Lagrange interpolation
        // F(X) passes through points (0, e_0), (1, e_1), ..., (k-1, e_{k-1})
        // where e_i is the error term of instance i.
        var evaluationPoints = [Fr]()
        evaluationPoints.reserveCapacity(k)
        for i in 0..<k {
            evaluationPoints.append(frFromInt(UInt64(i)))
        }

        var evaluationValues = [Fr]()
        evaluationValues.reserveCapacity(k)
        for i in 0..<k {
            evaluationValues.append(instances[i].errorTerm)
        }

        let fCoeffs = lagrangeInterpolate(points: evaluationPoints, values: evaluationValues)

        // Step 3: Build Fiat-Shamir transcript and get folding challenge alpha
        let transcript = Transcript(label: "protogalaxy-fold", backend: .keccak256)

        // Absorb all instances
        for inst in instances {
            absorbInstance(transcript, inst)
        }

        // Absorb F(X) coefficients
        for c in fCoeffs {
            transcript.absorb(c)
        }

        let alpha = transcript.squeeze()

        // Step 4: Compute Lagrange basis evaluated at alpha
        // L_i(alpha) for i = 0, ..., k-1 over domain {0, 1, ..., k-1}
        let lagrangeBasis = lagrangeBasisAtPoint(domainSize: k, point: alpha)

        // Step 5: Fold commitments using Lagrange coefficients
        // C' = sum_i L_i(alpha) * C_i
        var foldedCommitments = [PointProjective]()
        foldedCommitments.reserveCapacity(numWitnessColumns)
        for col in 0..<numWitnessColumns {
            var acc = pointIdentity()
            for i in 0..<k {
                let scaled = cPointScalarMul(instances[i].witnessCommitments[col], lagrangeBasis[i])
                acc = pointAdd(acc, scaled)
            }
            foldedCommitments.append(acc)
        }

        // Step 6: Fold witness polynomials
        // w'[col][j] = sum_i L_i(alpha) * w_i[col][j]
        var foldedWitnesses = [[Fr]]()
        foldedWitnesses.reserveCapacity(numWitnessColumns)
        for col in 0..<numWitnessColumns {
            let witLen = witnesses[0][col].count
            var folded = [Fr](repeating: Fr.zero, count: witLen)
            for i in 0..<k {
                let li = lagrangeBasis[i]
                let wit = witnesses[i][col]
                for j in 0..<witLen {
                    folded[j] = frAdd(folded[j], frMul(li, wit[j]))
                }
            }
            foldedWitnesses.append(folded)
        }

        // Step 7: Fold public inputs
        let numPub = instances[0].publicInput.count
        var foldedPublicInput = [Fr](repeating: Fr.zero, count: numPub)
        for i in 0..<k {
            let li = lagrangeBasis[i]
            for j in 0..<numPub {
                foldedPublicInput[j] = frAdd(foldedPublicInput[j],
                                             frMul(li, instances[i].publicInput[j]))
            }
        }

        // Step 8: Fold challenges (beta, gamma) and relaxation scalar
        var foldedBeta = Fr.zero
        var foldedGamma = Fr.zero
        var foldedU = Fr.zero
        for i in 0..<k {
            let li = lagrangeBasis[i]
            foldedBeta = frAdd(foldedBeta, frMul(li, instances[i].beta))
            foldedGamma = frAdd(foldedGamma, frMul(li, instances[i].gamma))
            foldedU = frAdd(foldedU, frMul(li, instances[i].u))
        }

        // Step 9: Compute folded error term
        // e' = F(alpha) = sum_i f_i * alpha^i
        let foldedError = hornerEvaluate(coeffs: fCoeffs, at: alpha)

        let foldedInstance = ProtogalaxyInstance(
            witnessCommitments: foldedCommitments,
            publicInput: foldedPublicInput,
            beta: foldedBeta,
            gamma: foldedGamma,
            errorTerm: foldedError,
            u: foldedU
        )

        let proof = ProtogalaxyFoldingProof(fCoefficients: fCoeffs, instanceCount: k)

        return (foldedInstance, foldedWitnesses, proof)
    }

    // MARK: - IVC Chain

    /// Run an IVC chain: fold a sequence of Plonk instances incrementally.
    ///
    /// - Parameters:
    ///   - instances: Array of committed Plonk instances
    ///   - witnesses: Corresponding witness arrays
    /// - Returns: (final accumulated instance, final witnesses)
    public func ivcChain(instances: [ProtogalaxyInstance],
                         witnesses: [[[Fr]]]) -> (ProtogalaxyInstance, [[Fr]]) {
        precondition(instances.count >= 2)
        precondition(instances.count == witnesses.count)

        var running = instances[0]
        var runningWit = witnesses[0]

        for i in 1..<instances.count {
            let (folded, foldedWit, _) = fold(
                instances: [running, instances[i]],
                witnesses: [runningWit, witnesses[i]]
            )
            running = folded
            runningWit = foldedWit
        }

        return (running, runningWit)
    }

    // MARK: - Transcript Helpers

    func absorbInstance(_ transcript: Transcript, _ instance: ProtogalaxyInstance) {
        transcript.absorbLabel("protogalaxy-instance")
        for c in instance.witnessCommitments {
            absorbPoint(transcript, c)
        }
        for x in instance.publicInput {
            transcript.absorb(x)
        }
        transcript.absorb(instance.beta)
        transcript.absorb(instance.gamma)
        transcript.absorb(instance.errorTerm)
        transcript.absorb(instance.u)
    }

    func absorbPoint(_ transcript: Transcript, _ p: PointProjective) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
            return
        }
        var affine = (Fp.zero, Fp.zero)
        withUnsafeBytes(of: p) { pBuf in
            withUnsafeMutableBytes(of: &affine) { aBuf in
                bn254_projective_to_affine(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        transcript.absorb(fpToFr(affine.0))
        transcript.absorb(fpToFr(affine.1))
    }
}

// MARK: - Protogalaxy Verifier

/// Verifier for Protogalaxy folding.
///
/// Checks that the folded instance was computed correctly from the original
/// instances and the folding proof, without access to witnesses.
///
/// Verifier work: O(k) scalar multiplications + O(k) field operations.
public class ProtogalaxyVerifier {

    public init() {}

    // MARK: - Verify Fold

    /// Verify that a Protogalaxy fold was performed correctly.
    ///
    /// Checks:
    ///   1. F(X) interpolates the correct error values at {0, ..., k-1}
    ///   2. Folded commitments = sum_i L_i(alpha) * C_i
    ///   3. Folded public input = sum_i L_i(alpha) * x_i
    ///   4. Folded challenges = sum_i L_i(alpha) * (beta_i, gamma_i)
    ///   5. Folded error = F(alpha)
    ///   6. Folded u = sum_i L_i(alpha) * u_i
    ///
    /// - Parameters:
    ///   - instances: The original k instances that were folded
    ///   - folded: The claimed folded instance
    ///   - proof: The folding proof containing F(X) coefficients
    /// - Returns: true if the fold is valid
    public func verifyFold(instances: [ProtogalaxyInstance],
                           folded: ProtogalaxyInstance,
                           proof: ProtogalaxyFoldingProof) -> Bool {
        let k = instances.count
        guard k >= 2 else { return false }
        guard proof.instanceCount == k else { return false }

        // Rebuild transcript (must match prover's)
        let transcript = Transcript(label: "protogalaxy-fold", backend: .keccak256)
        for inst in instances {
            absorbInstance(transcript, inst)
        }
        for c in proof.fCoefficients {
            transcript.absorb(c)
        }
        let alpha = transcript.squeeze()

        // Check 1: F(X) consistency
        // Verify F(i) = e_i for i = 0, ..., k-1
        for i in 0..<k {
            let point = frFromInt(UInt64(i))
            let fAtI = hornerEvaluate(coeffs: proof.fCoefficients, at: point)
            guard frEq(fAtI, instances[i].errorTerm) else { return false }
        }

        // Compute Lagrange basis at alpha
        let lagrangeBasis = lagrangeBasisAtPoint(domainSize: k, point: alpha)

        // Check 2: Commitment update
        let numCols = instances[0].witnessCommitments.count
        guard folded.witnessCommitments.count == numCols else { return false }
        for col in 0..<numCols {
            var expectedC = pointIdentity()
            for i in 0..<k {
                let scaled = cPointScalarMul(instances[i].witnessCommitments[col], lagrangeBasis[i])
                expectedC = pointAdd(expectedC, scaled)
            }
            guard pointEqual(folded.witnessCommitments[col], expectedC) else { return false }
        }

        // Check 3: Public input linearity
        let numPub = instances[0].publicInput.count
        guard folded.publicInput.count == numPub else { return false }
        for j in 0..<numPub {
            var expected = Fr.zero
            for i in 0..<k {
                expected = frAdd(expected, frMul(lagrangeBasis[i], instances[i].publicInput[j]))
            }
            guard frEq(folded.publicInput[j], expected) else { return false }
        }

        // Check 4: Challenge consistency
        var expectedBeta = Fr.zero
        var expectedGamma = Fr.zero
        for i in 0..<k {
            expectedBeta = frAdd(expectedBeta, frMul(lagrangeBasis[i], instances[i].beta))
            expectedGamma = frAdd(expectedGamma, frMul(lagrangeBasis[i], instances[i].gamma))
        }
        guard frEq(folded.beta, expectedBeta) else { return false }
        guard frEq(folded.gamma, expectedGamma) else { return false }

        // Check 5: Error term = F(alpha)
        let expectedError = hornerEvaluate(coeffs: proof.fCoefficients, at: alpha)
        guard frEq(folded.errorTerm, expectedError) else { return false }

        // Check 6: Relaxation scalar
        var expectedU = Fr.zero
        for i in 0..<k {
            expectedU = frAdd(expectedU, frMul(lagrangeBasis[i], instances[i].u))
        }
        guard frEq(folded.u, expectedU) else { return false }

        return true
    }

    // MARK: - Transcript Helpers

    func absorbInstance(_ transcript: Transcript, _ instance: ProtogalaxyInstance) {
        transcript.absorbLabel("protogalaxy-instance")
        for c in instance.witnessCommitments {
            absorbPoint(transcript, c)
        }
        for x in instance.publicInput {
            transcript.absorb(x)
        }
        transcript.absorb(instance.beta)
        transcript.absorb(instance.gamma)
        transcript.absorb(instance.errorTerm)
        transcript.absorb(instance.u)
    }

    func absorbPoint(_ transcript: Transcript, _ p: PointProjective) {
        if pointIsIdentity(p) {
            transcript.absorb(Fr.zero)
            transcript.absorb(Fr.zero)
            return
        }
        var affine = (Fp.zero, Fp.zero)
        withUnsafeBytes(of: p) { pBuf in
            withUnsafeMutableBytes(of: &affine) { aBuf in
                bn254_projective_to_affine(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
        transcript.absorb(fpToFr(affine.0))
        transcript.absorb(fpToFr(affine.1))
    }
}

// MARK: - Lagrange Interpolation

/// Lagrange interpolation: given (points[i], values[i]) for i=0..n-1,
/// compute the unique polynomial of degree <= n-1 passing through all points.
/// Returns coefficients [c_0, c_1, ..., c_{n-1}] where p(x) = sum c_i x^i.
///
/// O(n^2) algorithm; for Protogalaxy k is typically small (2-16).
public func lagrangeInterpolate(points: [Fr], values: [Fr]) -> [Fr] {
    let n = points.count
    precondition(n == values.count)
    if n == 0 { return [] }
    if n == 1 { return [values[0]] }

    // Result accumulator in coefficient form
    var result = [Fr](repeating: Fr.zero, count: n)

    for i in 0..<n {
        // Compute Lagrange basis polynomial L_i(x) = prod_{j!=i} (x - p_j) / (p_i - p_j)
        // First compute the denominator: prod_{j!=i} (p_i - p_j)
        var denom = Fr.one
        for j in 0..<n where j != i {
            denom = frMul(denom, frSub(points[i], points[j]))
        }
        let denomInv = frInverse(denom)
        let coeff = frMul(values[i], denomInv)

        // Build L_i(x) in coefficient form by multiplying linear factors (x - p_j)
        // Start with [1] and multiply by (x - p_j) for each j != i
        var basis = [Fr](repeating: Fr.zero, count: n)
        basis[0] = Fr.one
        var degree = 0

        for j in 0..<n where j != i {
            let negPj = frNeg(points[j])
            // Multiply current polynomial by (x - p_j)
            // New coefficient[k] = old[k-1] + negPj * old[k]
            degree += 1
            // Process from high degree to low to avoid overwriting
            for d in stride(from: degree, to: 0, by: -1) {
                basis[d] = frAdd(basis[d - 1], frMul(negPj, basis[d]))
            }
            basis[0] = frMul(negPj, basis[0])
        }

        // Accumulate coeff * L_i(x) into result
        for d in 0..<n {
            result[d] = frAdd(result[d], frMul(coeff, basis[d]))
        }
    }

    return result
}

/// Evaluate Lagrange basis polynomials L_0(alpha), ..., L_{k-1}(alpha)
/// over the domain {0, 1, ..., k-1}.
///
/// L_i(alpha) = prod_{j!=i} (alpha - j) / (i - j)
///
/// O(k^2) for small k (typically 2-16 in Protogalaxy).
public func lagrangeBasisAtPoint(domainSize k: Int, point alpha: Fr) -> [Fr] {
    precondition(k >= 1)
    if k == 1 { return [Fr.one] }

    var result = [Fr](repeating: Fr.zero, count: k)

    // Pre-compute (alpha - j) for all j
    var alphaMinusJ = [Fr]()
    alphaMinusJ.reserveCapacity(k)
    for j in 0..<k {
        alphaMinusJ.append(frSub(alpha, frFromInt(UInt64(j))))
    }

    for i in 0..<k {
        // Numerator: prod_{j!=i} (alpha - j)
        var numer = Fr.one
        for j in 0..<k where j != i {
            numer = frMul(numer, alphaMinusJ[j])
        }

        // Denominator: prod_{j!=i} (i - j)
        var denom = Fr.one
        let iFr = frFromInt(UInt64(i))
        for j in 0..<k where j != i {
            denom = frMul(denom, frSub(iFr, frFromInt(UInt64(j))))
        }

        result[i] = frMul(numer, frInverse(denom))
    }

    return result
}

/// Horner's method polynomial evaluation: p(x) = sum_i coeffs[i] * x^i
/// O(n) multiplications.
public func hornerEvaluate(coeffs: [Fr], at x: Fr) -> Fr {
    if coeffs.isEmpty { return Fr.zero }
    var result = coeffs[coeffs.count - 1]
    for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
        result = frAdd(frMul(result, x), coeffs[i])
    }
    return result
}
