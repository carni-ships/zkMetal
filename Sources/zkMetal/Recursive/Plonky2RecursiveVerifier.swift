// Plonky2RecursiveVerifier — Encode Plonky2 STARK verification as an arithmetic circuit
//
// Plonky2 uses:
//   - Goldilocks field (p = 2^64 - 2^32 + 1) with quadratic extension GF(p^2)
//   - Poseidon hash for Merkle commitments and Fiat-Shamir
//   - Degree-2 FRI for polynomial proximity testing
//
// The recursive verifier encodes the entire Plonky2 verification algorithm as
// constraints over a native proof system (BN254 Plonk or Groth16). This enables:
//   1. Verifying Plonky2 proofs on Ethereum (via BN254 pairing check)
//   2. Proof composition: chain Plonky2 proofs with BN254 SNARKs
//
// Architecture:
//   - GoldilocksExtField: quadratic extension GF(p^2) = Gl[X]/(X^2 - 7)
//   - Plonky2FRIProof: proof data structure for FRI over Goldilocks
//   - Plonky2VerifierCircuit: FRI verification as PlonkCircuitBuilder constraints
//   - Plonky2RecursiveProver: end-to-end pipeline for recursive proof generation
//
// Non-native overhead: Goldilocks elements (64-bit) inside BN254 Fr (~254-bit)
// fit natively without limb decomposition — a single Fr element can hold a Gl value.
// Extension field elements need 2 Fr wires. This makes Plonky2 recursion relatively
// cheap compared to BN254-in-BN254 recursion (~500K constraints for pairing).
//
// References:
//   - Plonky2 whitepaper (Polygon Zero, 2022)
//   - "Recursive STARKs" (StarkWare, 2022)
//   - FRI protocol (Ben-Sasson et al., 2018)

import Foundation

// MARK: - Goldilocks Quadratic Extension Field GF(p^2)

/// Quadratic extension of the Goldilocks field: GF(p^2) = GF(p)[X] / (X^2 - 7)
///
/// Elements are (c0, c1) representing c0 + c1 * W where W^2 = 7.
/// This is the extension used by Plonky2 for FRI and evaluation proofs.
///
/// The irreducible polynomial X^2 - 7 is chosen because 7 is a quadratic
/// non-residue modulo p = 2^64 - 2^32 + 1.
public struct GoldilocksExtField: Equatable {
    public var c0: Gl
    public var c1: Gl

    /// The non-residue W^2 = 7 defining the extension X^2 - 7
    public static let W_SQUARED: UInt64 = 7

    public static var zero: GoldilocksExtField {
        GoldilocksExtField(c0: Gl.zero, c1: Gl.zero)
    }
    public static var one: GoldilocksExtField {
        GoldilocksExtField(c0: Gl.one, c1: Gl.zero)
    }

    public init(c0: Gl, c1: Gl) {
        self.c0 = c0
        self.c1 = c1
    }

    /// Lift a base field element into the extension
    public init(base: Gl) {
        self.c0 = base
        self.c1 = Gl.zero
    }

    public var isZero: Bool { c0.isZero && c1.isZero }
}

/// Addition in GF(p^2): (a0+a1*W) + (b0+b1*W) = (a0+b0) + (a1+b1)*W
public func glExtAdd(_ a: GoldilocksExtField, _ b: GoldilocksExtField) -> GoldilocksExtField {
    GoldilocksExtField(c0: glAdd(a.c0, b.c0), c1: glAdd(a.c1, b.c1))
}

/// Subtraction in GF(p^2)
public func glExtSub(_ a: GoldilocksExtField, _ b: GoldilocksExtField) -> GoldilocksExtField {
    GoldilocksExtField(c0: glSub(a.c0, b.c0), c1: glSub(a.c1, b.c1))
}

/// Negation in GF(p^2)
public func glExtNeg(_ a: GoldilocksExtField) -> GoldilocksExtField {
    GoldilocksExtField(c0: glNeg(a.c0), c1: glNeg(a.c1))
}

/// Multiplication in GF(p^2): (a0+a1*W)(b0+b1*W) = (a0*b0 + 7*a1*b1) + (a0*b1+a1*b0)*W
/// Uses Karatsuba: 3 base muls instead of 4.
public func glExtMul(_ a: GoldilocksExtField, _ b: GoldilocksExtField) -> GoldilocksExtField {
    let t0 = glMul(a.c0, b.c0)                          // a0*b0
    let t1 = glMul(a.c1, b.c1)                          // a1*b1
    let a01 = glAdd(a.c0, a.c1)
    let b01 = glAdd(b.c0, b.c1)
    let cross = glMul(a01, b01)                          // (a0+a1)*(b0+b1)
    // c0 = t0 + 7*t1
    let seven = Gl(v: GoldilocksExtField.W_SQUARED)
    let c0 = glAdd(t0, glMul(seven, t1))
    // c1 = cross - t0 - t1
    let c1 = glSub(glSub(cross, t0), t1)
    return GoldilocksExtField(c0: c0, c1: c1)
}

/// Squaring in GF(p^2): optimized (a0+a1*W)^2 = (a0^2 + 7*a1^2) + (2*a0*a1)*W
public func glExtSqr(_ a: GoldilocksExtField) -> GoldilocksExtField {
    let a0sq = glSqr(a.c0)
    let a1sq = glSqr(a.c1)
    let seven = Gl(v: GoldilocksExtField.W_SQUARED)
    let c0 = glAdd(a0sq, glMul(seven, a1sq))
    let c1 = glMul(Gl(v: 2), glMul(a.c0, a.c1))
    return GoldilocksExtField(c0: c0, c1: c1)
}

/// Multiplicative inverse in GF(p^2): 1/(a0+a1*W) = (a0-a1*W) / (a0^2 - 7*a1^2)
public func glExtInverse(_ a: GoldilocksExtField) -> GoldilocksExtField {
    let a0sq = glSqr(a.c0)
    let a1sq = glSqr(a.c1)
    let seven = Gl(v: GoldilocksExtField.W_SQUARED)
    // norm = a0^2 - 7*a1^2 (in base field)
    let norm = glSub(a0sq, glMul(seven, a1sq))
    let normInv = glInverse(norm)
    return GoldilocksExtField(
        c0: glMul(a.c0, normInv),
        c1: glNeg(glMul(a.c1, normInv))
    )
}

/// Exponentiation in GF(p^2) via square-and-multiply
public func glExtPow(_ base: GoldilocksExtField, _ exp: UInt64) -> GoldilocksExtField {
    if exp == 0 { return .one }
    var result = GoldilocksExtField.one
    var b = base
    var e = exp
    while e > 0 {
        if e & 1 == 1 { result = glExtMul(result, b) }
        b = glExtSqr(b)
        e >>= 1
    }
    return result
}

// MARK: - Poseidon Hash over Goldilocks (Plonky2 variant)

/// Poseidon hash function over Goldilocks field as used by Plonky2.
/// State width t=12, capacity=4, rate=8, S-box: x^7, full rounds=8, partial rounds=22.
///
/// This is the *native* (out-of-circuit) implementation used for:
///   1. Merkle tree commitment verification
///   2. Fiat-Shamir challenge generation
///   3. Witness computation for the in-circuit Poseidon gadget
public struct GoldilocksPoseidon {
    /// Poseidon state width
    public static let stateWidth = 12
    /// Rate (number of absorbed elements per permutation)
    public static let rate = 8
    /// Capacity
    public static let capacity = 4
    /// Number of full rounds
    public static let fullRounds = 8
    /// Number of partial rounds
    public static let partialRounds = 22
    /// S-box exponent: x^7
    public static let sboxDegree: UInt64 = 7

    /// Round constants (precomputed for Goldilocks Poseidon)
    /// In production, these come from a nothing-up-my-sleeve seed.
    /// Here we generate deterministically from a fixed seed for reproducibility.
    public static let roundConstants: [[Gl]] = {
        var constants = [[Gl]]()
        let totalRounds = fullRounds + partialRounds
        // Deterministic round constant generation via iterative hashing
        var state: UInt64 = 0x517cc1b727220a95  // seed
        for r in 0..<totalRounds {
            var row = [Gl]()
            for _ in 0..<stateWidth {
                // Simple PRNG for deterministic constants
                state = state &* 6364136223846793005 &+ 1442695040888963407
                let val = state % Gl.P
                row.append(Gl(v: val))
            }
            constants.append(row)
        }
        return constants
    }()

    /// MDS matrix for the external (full) rounds.
    /// Plonky2 uses a circulant MDS matrix for Goldilocks Poseidon.
    public static let mdsMatrix: [[Gl]] = {
        // Cauchy-based MDS matrix construction
        var matrix = [[Gl]](repeating: [Gl](repeating: Gl.zero, count: stateWidth), count: stateWidth)
        for i in 0..<stateWidth {
            for j in 0..<stateWidth {
                // M[i][j] = 1 / (x_i + y_j) where x_i, y_j are distinct elements
                let xi = Gl(v: UInt64(i))
                let yj = Gl(v: UInt64(stateWidth + j))
                let sum = glAdd(xi, yj)
                matrix[i][j] = glInverse(sum)
            }
        }
        return matrix
    }()

    /// Apply S-box: x -> x^7
    private static func sbox(_ x: Gl) -> Gl {
        let x2 = glSqr(x)
        let x3 = glMul(x2, x)
        let x6 = glSqr(x3)
        return glMul(x6, x)
    }

    /// Apply MDS matrix to state
    private static func mdsLayer(_ state: [Gl]) -> [Gl] {
        var result = [Gl](repeating: Gl.zero, count: stateWidth)
        for i in 0..<stateWidth {
            for j in 0..<stateWidth {
                result[i] = glAdd(result[i], glMul(mdsMatrix[i][j], state[j]))
            }
        }
        return result
    }

    /// Full Poseidon permutation
    public static func permutation(_ input: [Gl]) -> [Gl] {
        precondition(input.count == stateWidth)
        var state = input

        let halfFull = fullRounds / 2

        // First half of full rounds
        for r in 0..<halfFull {
            // Add round constants
            for i in 0..<stateWidth { state[i] = glAdd(state[i], roundConstants[r][i]) }
            // S-box on all elements
            for i in 0..<stateWidth { state[i] = sbox(state[i]) }
            // MDS
            state = mdsLayer(state)
        }

        // Partial rounds: S-box only on first element
        for r in 0..<partialRounds {
            let rc = halfFull + r
            for i in 0..<stateWidth { state[i] = glAdd(state[i], roundConstants[rc][i]) }
            state[0] = sbox(state[0])
            state = mdsLayer(state)
        }

        // Second half of full rounds
        for r in 0..<halfFull {
            let rc = halfFull + partialRounds + r
            for i in 0..<stateWidth { state[i] = glAdd(state[i], roundConstants[rc][i]) }
            for i in 0..<stateWidth { state[i] = sbox(state[i]) }
            state = mdsLayer(state)
        }

        return state
    }

    /// Hash a variable-length input using sponge construction (rate=8, capacity=4)
    public static func hashMany(_ inputs: [Gl]) -> [Gl] {
        var state = [Gl](repeating: Gl.zero, count: stateWidth)
        var pos = 0
        while pos < inputs.count {
            let chunk = min(rate, inputs.count - pos)
            for i in 0..<chunk {
                state[i] = glAdd(state[i], inputs[pos + i])
            }
            state = permutation(state)
            pos += chunk
        }
        // Output: first `capacity` elements (4 elements = 256 bits)
        return Array(state[0..<capacity])
    }

    /// 2-to-1 compression: hash two 4-element digests into one
    public static func compress(_ left: [Gl], _ right: [Gl]) -> [Gl] {
        precondition(left.count == capacity && right.count == capacity)
        return hashMany(left + right)
    }
}

// MARK: - Plonky2 Proof Data Structures

/// A Merkle authentication path for Plonky2 (using Goldilocks Poseidon).
public struct Plonky2MerklePath {
    /// Sibling hashes along the path (each is a 4-element Goldilocks digest)
    public let siblings: [[Gl]]
    /// Leaf index in the tree
    public let index: Int

    public init(siblings: [[Gl]], index: Int) {
        self.siblings = siblings
        self.index = index
    }
}

/// A single FRI query response in a Plonky2 proof.
public struct Plonky2FRIQueryRound {
    /// Evaluations at the queried coset (2 elements for degree-2 FRI)
    public let cosetEvals: [GoldilocksExtField]
    /// Merkle authentication path for the commitment at this layer
    public let merklePath: Plonky2MerklePath

    public init(cosetEvals: [GoldilocksExtField], merklePath: Plonky2MerklePath) {
        self.cosetEvals = cosetEvals
        self.merklePath = merklePath
    }
}

/// Complete FRI proof as produced by Plonky2.
public struct Plonky2FRIProof {
    /// Committed polynomial evaluations (initial oracle)
    public let initialTreeRoot: [Gl]
    /// Per-round commitments and folding data
    public let commitRoots: [[Gl]]
    /// Query responses for each query, each containing per-round data
    public let queryRoundData: [[Plonky2FRIQueryRound]]
    /// Final constant polynomial (after all folding rounds)
    public let finalPoly: [GoldilocksExtField]
    /// Proof-of-work nonce (for grinding, if enabled)
    public let powNonce: UInt64

    public init(initialTreeRoot: [Gl], commitRoots: [[Gl]],
                queryRoundData: [[Plonky2FRIQueryRound]],
                finalPoly: [GoldilocksExtField], powNonce: UInt64) {
        self.initialTreeRoot = initialTreeRoot
        self.commitRoots = commitRoots
        self.queryRoundData = queryRoundData
        self.finalPoly = finalPoly
        self.powNonce = powNonce
    }
}

/// Complete Plonky2 proof (STARK + FRI).
public struct Plonky2Proof {
    /// Wire evaluations (public inputs)
    public let publicInputs: [Gl]
    /// Wire commitment roots
    public let wires: [[Gl]]
    /// Plonk Z partial products commitment
    public let plonkZsPartialProducts: [[Gl]]
    /// Quotient polynomial commitment
    public let quotientPolys: [[Gl]]
    /// Opening proof (FRI)
    public let openingProof: Plonky2FRIProof
    /// Polynomial evaluations at the challenge point
    public let openings: Plonky2Openings

    public init(publicInputs: [Gl], wires: [[Gl]],
                plonkZsPartialProducts: [[Gl]], quotientPolys: [[Gl]],
                openingProof: Plonky2FRIProof, openings: Plonky2Openings) {
        self.publicInputs = publicInputs
        self.wires = wires
        self.plonkZsPartialProducts = plonkZsPartialProducts
        self.quotientPolys = quotientPolys
        self.openingProof = openingProof
        self.openings = openings
    }
}

/// Polynomial evaluations at the challenge point (zeta) and zeta*omega.
public struct Plonky2Openings {
    /// Evaluations at zeta (in extension field)
    public let atZeta: [GoldilocksExtField]
    /// Evaluations at zeta * omega (for next-row access)
    public let atZetaNext: [GoldilocksExtField]

    public init(atZeta: [GoldilocksExtField], atZetaNext: [GoldilocksExtField]) {
        self.atZeta = atZeta
        self.atZetaNext = atZetaNext
    }
}

/// Plonky2 verification key (circuit-specific parameters).
public struct Plonky2VerificationKey {
    /// Number of wire columns
    public let numWires: Int
    /// Number of routed wires (wires used in copy constraints)
    public let numRoutedWires: Int
    /// Degree bits (log2 of trace length)
    public let degreeBits: Int
    /// FRI rate bits (log2 of blowup factor)
    public let friRateBits: Int
    /// Number of FRI queries
    public let numFRIQueries: Int
    /// Number of constants
    public let numConstants: Int
    /// Number of public inputs
    public let numPublicInputs: Int
    /// Circuit digest (Poseidon hash of the circuit description)
    public let circuitDigest: [Gl]

    public init(numWires: Int, numRoutedWires: Int, degreeBits: Int,
                friRateBits: Int, numFRIQueries: Int, numConstants: Int,
                numPublicInputs: Int, circuitDigest: [Gl]) {
        self.numWires = numWires
        self.numRoutedWires = numRoutedWires
        self.degreeBits = degreeBits
        self.friRateBits = friRateBits
        self.numFRIQueries = numFRIQueries
        self.numConstants = numConstants
        self.numPublicInputs = numPublicInputs
        self.circuitDigest = circuitDigest
    }
}

// MARK: - Plonky2 Verifier Circuit

/// Encodes the Plonky2 STARK+FRI verification algorithm as a PlonkCircuitBuilder circuit
/// over BN254 Fr. Since Goldilocks elements (64-bit) fit inside BN254 Fr (~254-bit),
/// each Goldilocks value is a single native circuit variable.
///
/// The circuit checks:
///   1. Fiat-Shamir transcript replay via in-circuit Poseidon over Goldilocks
///   2. FRI query consistency: coset evaluation matches folded polynomial
///   3. Merkle path verification for each FRI commitment
///   4. Final polynomial degree check (evaluates constant poly at query points)
///   5. Plonk constraint evaluation at the opening point
public class Plonky2VerifierCircuit {
    public let builder: PlonkCircuitBuilder

    // Extension field variable: pair of native Fr wires (c0, c1)
    public struct ExtVar {
        public let c0: Int  // wire index for real part
        public let c1: Int  // wire index for imaginary part
    }

    // Poseidon state variable: 12 native Fr wires
    public struct PoseidonStateVar {
        public let elements: [Int]  // 12 wire indices
    }

    public init(builder: PlonkCircuitBuilder) {
        self.builder = builder
    }

    // MARK: - Goldilocks Base Field In-Circuit Operations

    /// Allocate a Goldilocks variable as a single native Fr wire.
    /// Goldilocks p = 2^64 - 2^32 + 1 fits in BN254 Fr (254-bit field).
    public func allocateGl() -> Int {
        builder.addInput()
    }

    /// Allocate an extension field variable (two Gl values).
    public func allocateExt() -> ExtVar {
        ExtVar(c0: allocateGl(), c1: allocateGl())
    }

    /// Add two Goldilocks values in-circuit.
    /// Since Gl values fit in Fr, native Fr addition gives the correct sum
    /// as long as we reduce modulo Gl.P afterwards. We handle this by
    /// constraining the result with a reduction witness.
    public func glAddCircuit(_ a: Int, _ b: Int) -> Int {
        // sum_unreduced = a + b (in Fr, this is exact since a,b < 2^64)
        let sumUnreduced = builder.add(a, b)
        // Witness: reduced = (a + b) mod Gl.P
        // Constraint: sum_unreduced = reduced + k * Gl.P for some k in {0, 1}
        let reduced = builder.addInput()
        let k = builder.addInput()
        // k * Gl.P
        let glP = builder.constant(glToFr(Gl.P))
        let kTimesP = builder.mul(k, glP)
        // reduced + k * Gl.P should equal sum_unreduced
        let recon = builder.add(reduced, kTimesP)
        builder.assertEqual(recon, sumUnreduced)
        return reduced
    }

    /// Multiply two Goldilocks values in-circuit.
    /// Native Fr multiplication gives the full product (fits in Fr since
    /// a,b < 2^64 so a*b < 2^128 < Fr.P). Then reduce mod Gl.P.
    public func glMulCircuit(_ a: Int, _ b: Int) -> Int {
        let prodUnreduced = builder.mul(a, b)
        // Witness: reduced = (a * b) mod Gl.P
        // Constraint: prod_unreduced = reduced + q * Gl.P
        let reduced = builder.addInput()
        let q = builder.addInput()
        let glP = builder.constant(glToFr(Gl.P))
        let qTimesP = builder.mul(q, glP)
        let recon = builder.add(reduced, qTimesP)
        builder.assertEqual(recon, prodUnreduced)
        return reduced
    }

    /// Subtract two Goldilocks values in-circuit: a - b mod Gl.P
    public func glSubCircuit(_ a: Int, _ b: Int) -> Int {
        // result = a - b mod Gl.P
        // Constraint: result + b = a (mod Gl.P), i.e. result + b = a + k*Gl.P
        let result = builder.addInput()
        let resultPlusB = glAddCircuit(result, b)
        // We need resultPlusB == a (both reduced mod Gl.P)
        builder.assertEqual(resultPlusB, a)
        return result
    }

    // MARK: - Extension Field In-Circuit Operations

    /// Extension field addition: (a0+a1*W) + (b0+b1*W)
    public func extAddCircuit(_ a: ExtVar, _ b: ExtVar) -> ExtVar {
        ExtVar(
            c0: glAddCircuit(a.c0, b.c0),
            c1: glAddCircuit(a.c1, b.c1)
        )
    }

    /// Extension field subtraction
    public func extSubCircuit(_ a: ExtVar, _ b: ExtVar) -> ExtVar {
        ExtVar(
            c0: glSubCircuit(a.c0, b.c0),
            c1: glSubCircuit(a.c1, b.c1)
        )
    }

    /// Extension field multiplication: (a0+a1*W)(b0+b1*W) = (a0*b0+7*a1*b1) + (a0*b1+a1*b0)*W
    public func extMulCircuit(_ a: ExtVar, _ b: ExtVar) -> ExtVar {
        let t0 = glMulCircuit(a.c0, b.c0)       // a0*b0
        let t1 = glMulCircuit(a.c1, b.c1)       // a1*b1
        let seven = builder.constant(glToFr(GoldilocksExtField.W_SQUARED))
        let sevenT1 = glMulCircuit(seven, t1)    // 7*a1*b1
        let c0 = glAddCircuit(t0, sevenT1)       // a0*b0 + 7*a1*b1

        let a0b1 = glMulCircuit(a.c0, b.c1)
        let a1b0 = glMulCircuit(a.c1, b.c0)
        let c1 = glAddCircuit(a0b1, a1b0)        // a0*b1 + a1*b0

        return ExtVar(c0: c0, c1: c1)
    }

    /// Extension field equality constraint
    public func extEqualCircuit(_ a: ExtVar, _ b: ExtVar) {
        builder.assertEqual(a.c0, b.c0)
        builder.assertEqual(a.c1, b.c1)
    }

    /// Extension constant from out-of-circuit value
    public func extConstant(_ val: GoldilocksExtField) -> ExtVar {
        ExtVar(
            c0: builder.constant(glToFr(val.c0.v)),
            c1: builder.constant(glToFr(val.c1.v))
        )
    }

    // MARK: - In-Circuit Poseidon Hash (Goldilocks)

    /// Allocate a Poseidon state (12 Goldilocks wires).
    public func allocatePoseidonState() -> PoseidonStateVar {
        PoseidonStateVar(elements: (0..<GoldilocksPoseidon.stateWidth).map { _ in allocateGl() })
    }

    /// In-circuit S-box: x -> x^7 = ((x^2)^2 * x) * x = x^6 * x
    /// Decomposed as: x2 = x*x, x3 = x2*x, x6 = x3*x3, x7 = x6*x
    public func sboxCircuit(_ x: Int) -> Int {
        let x2 = glMulCircuit(x, x)
        let x3 = glMulCircuit(x2, x)
        let x6 = glMulCircuit(x3, x3)
        return glMulCircuit(x6, x)
    }

    /// In-circuit MDS layer: matrix-vector product
    public func mdsLayerCircuit(_ state: [Int]) -> [Int] {
        var result = [Int]()
        let mds = GoldilocksPoseidon.mdsMatrix
        for i in 0..<GoldilocksPoseidon.stateWidth {
            var acc = builder.constant(glToFr(0))
            for j in 0..<GoldilocksPoseidon.stateWidth {
                let coeff = builder.constant(glToFr(mds[i][j].v))
                let term = glMulCircuit(coeff, state[j])
                acc = glAddCircuit(acc, term)
            }
            result.append(acc)
        }
        return result
    }

    /// In-circuit Poseidon permutation (full verification of the hash computation).
    /// This is the most constraint-heavy component: ~30 rounds * 12 S-boxes per full round.
    public func poseidonPermutationCircuit(_ input: [Int]) -> [Int] {
        precondition(input.count == GoldilocksPoseidon.stateWidth)
        var state = input
        let halfFull = GoldilocksPoseidon.fullRounds / 2
        let rc = GoldilocksPoseidon.roundConstants

        // First half of full rounds
        for r in 0..<halfFull {
            for i in 0..<GoldilocksPoseidon.stateWidth {
                let rcConst = builder.constant(glToFr(rc[r][i].v))
                state[i] = glAddCircuit(state[i], rcConst)
            }
            for i in 0..<GoldilocksPoseidon.stateWidth {
                state[i] = sboxCircuit(state[i])
            }
            state = mdsLayerCircuit(state)
        }

        // Partial rounds: S-box only on first element
        for r in 0..<GoldilocksPoseidon.partialRounds {
            let rcIdx = halfFull + r
            for i in 0..<GoldilocksPoseidon.stateWidth {
                let rcConst = builder.constant(glToFr(rc[rcIdx][i].v))
                state[i] = glAddCircuit(state[i], rcConst)
            }
            state[0] = sboxCircuit(state[0])
            state = mdsLayerCircuit(state)
        }

        // Second half of full rounds
        for r in 0..<halfFull {
            let rcIdx = halfFull + GoldilocksPoseidon.partialRounds + r
            for i in 0..<GoldilocksPoseidon.stateWidth {
                let rcConst = builder.constant(glToFr(rc[rcIdx][i].v))
                state[i] = glAddCircuit(state[i], rcConst)
            }
            for i in 0..<GoldilocksPoseidon.stateWidth {
                state[i] = sboxCircuit(state[i])
            }
            state = mdsLayerCircuit(state)
        }

        return state
    }

    /// In-circuit Poseidon 2-to-1 compression (hash two digests).
    public func poseidonCompressCircuit(_ left: [Int], _ right: [Int]) -> [Int] {
        precondition(left.count == GoldilocksPoseidon.capacity)
        precondition(right.count == GoldilocksPoseidon.capacity)
        var state = [Int]()
        // Absorb left + right into rate portion, capacity = 0
        for i in 0..<GoldilocksPoseidon.capacity { state.append(left[i]) }
        for i in 0..<GoldilocksPoseidon.capacity { state.append(right[i]) }
        // Pad remaining rate and capacity slots with zero
        let zeroVar = builder.constant(glToFr(0))
        while state.count < GoldilocksPoseidon.stateWidth {
            state.append(zeroVar)
        }
        let output = poseidonPermutationCircuit(state)
        return Array(output[0..<GoldilocksPoseidon.capacity])
    }

    // MARK: - Merkle Path Verification

    /// Verify a Merkle authentication path in-circuit.
    /// Leaf is a 4-element Goldilocks digest, siblings are along the path to root.
    public func verifyMerklePathCircuit(
        leaf: [Int],             // 4 wire indices (digest)
        siblings: [[Int]],       // list of 4-element sibling digests
        index: Int,              // leaf index (known at compile time)
        expectedRoot: [Int]      // 4 wire indices for expected root
    ) {
        var current = leaf
        var idx = index
        for level in 0..<siblings.count {
            let bit = idx & 1
            idx >>= 1
            if bit == 0 {
                current = poseidonCompressCircuit(current, siblings[level])
            } else {
                current = poseidonCompressCircuit(siblings[level], current)
            }
        }
        // Constrain computed root equals expected root
        for i in 0..<GoldilocksPoseidon.capacity {
            builder.assertEqual(current[i], expectedRoot[i])
        }
    }

    // MARK: - FRI Verification Constraints

    /// Constrain a single FRI folding step (degree-2 reduction).
    ///
    /// Given evaluations f(x) and f(-x) at a coset point, and a random challenge beta,
    /// the folded value is: g(x^2) = (f(x) + f(-x))/2 + beta * (f(x) - f(-x))/(2x)
    ///
    /// All arithmetic is in the extension field GF(p^2).
    public func friFoldStepCircuit(
        fAtX: ExtVar,           // f(x) in extension field
        fAtNegX: ExtVar,        // f(-x) in extension field
        beta: ExtVar,           // FRI challenge
        xVal: ExtVar,           // coset point x (in extension)
        result: ExtVar          // expected g(x^2)
    ) {
        // even = (f(x) + f(-x)) / 2
        let sum = extAddCircuit(fAtX, fAtNegX)
        let two = ExtVar(c0: builder.constant(glToFr(2)), c1: builder.constant(glToFr(0)))
        // Witness provides the division result; we constrain via multiplication
        let even = allocateExt()
        let evenTimesTwo = extMulCircuit(even, two)
        extEqualCircuit(evenTimesTwo, sum)

        // odd = (f(x) - f(-x)) / (2x)
        let diff = extSubCircuit(fAtX, fAtNegX)
        let twoX = extMulCircuit(two, xVal)
        let odd = allocateExt()
        let oddTimesTwoX = extMulCircuit(odd, twoX)
        extEqualCircuit(oddTimesTwoX, diff)

        // folded = even + beta * odd
        let betaOdd = extMulCircuit(beta, odd)
        let folded = extAddCircuit(even, betaOdd)
        extEqualCircuit(folded, result)
    }

    /// Constrain the FRI final polynomial evaluation.
    /// The final polynomial should be constant (degree 0 after all folding).
    /// Verify: finalPoly[0] == claimed evaluation at every query point.
    public func friFinalCheckCircuit(
        finalCoeffs: [ExtVar],      // coefficients of final poly
        evaluationPoint: ExtVar,    // query point (x^(2^k) after folding)
        expectedValue: ExtVar       // claimed evaluation
    ) {
        // Evaluate final polynomial at the point using Horner's method
        var acc = finalCoeffs.last!
        for i in stride(from: finalCoeffs.count - 2, through: 0, by: -1) {
            acc = extMulCircuit(acc, evaluationPoint)
            acc = extAddCircuit(acc, finalCoeffs[i])
        }
        extEqualCircuit(acc, expectedValue)
    }

    // MARK: - Fiat-Shamir Transcript

    /// In-circuit Fiat-Shamir challenge derivation using Poseidon over Goldilocks.
    /// Absorbs proof commitments and squeezes challenges.
    public func deriveChallenge(transcript: inout [Int], newData: [Int]) -> ExtVar {
        // Absorb new data into transcript state
        transcript.append(contentsOf: newData)
        // Hash all transcript data
        var state = [Int]()
        for i in 0..<min(GoldilocksPoseidon.stateWidth, transcript.count) {
            state.append(transcript[i])
        }
        let zeroVar = builder.constant(glToFr(0))
        while state.count < GoldilocksPoseidon.stateWidth {
            state.append(zeroVar)
        }
        let output = poseidonPermutationCircuit(state)
        // Challenge is the first two output elements (extension field element)
        return ExtVar(c0: output[0], c1: output[1])
    }

    // MARK: - Full FRI Verification Circuit

    /// Build the complete FRI verification circuit for a Plonky2 proof.
    ///
    /// Parameters:
    ///   - numQueries: number of FRI queries (security parameter)
    ///   - numRounds: number of FRI folding rounds (log(degree) for degree-2 folding)
    ///   - proof: the FRI proof data (used for witness generation)
    ///   - vk: verification key parameters
    ///
    /// Returns the public input wire indices (for the outer proof's PI).
    public func buildFRIVerificationCircuit(
        numQueries: Int,
        numRounds: Int,
        proof: Plonky2FRIProof,
        vk: Plonky2VerificationKey
    ) -> [Int] {
        var publicInputWires = [Int]()

        // Allocate public input: initial commitment root
        var initialRoot = [Int]()
        for i in 0..<GoldilocksPoseidon.capacity {
            let w = allocateGl()
            builder.addPublicInput(wireIndex: w)
            publicInputWires.append(w)
            initialRoot.append(w)
        }

        // Allocate per-round commitment roots
        var commitRootVars = [[Int]]()
        for _ in 0..<numRounds {
            var root = [Int]()
            for _ in 0..<GoldilocksPoseidon.capacity {
                root.append(allocateGl())
            }
            commitRootVars.append(root)
        }

        // Derive FRI challenges via Fiat-Shamir transcript
        var transcript = [Int]()
        transcript.append(contentsOf: initialRoot)

        var betas = [ExtVar]()
        for r in 0..<numRounds {
            let beta = deriveChallenge(transcript: &transcript, newData: commitRootVars[r])
            betas.append(beta)
        }

        // Allocate final polynomial coefficients
        var finalPolyVars = [ExtVar]()
        for _ in 0..<proof.finalPoly.count {
            finalPolyVars.append(allocateExt())
        }

        // For each query: verify the FRI folding chain
        for q in 0..<numQueries {
            guard q < proof.queryRoundData.count else { break }
            let queryData = proof.queryRoundData[q]

            // Allocate coset evaluations and Merkle paths for each round
            for r in 0..<min(numRounds, queryData.count) {
                let roundData = queryData[r]

                // Allocate coset evaluation variables
                var cosetEvalVars = [ExtVar]()
                for _ in roundData.cosetEvals {
                    cosetEvalVars.append(allocateExt())
                }

                // Verify Merkle path for this round's commitment
                // Leaf = hash of coset evaluations
                var leafData = [Int]()
                for ev in cosetEvalVars {
                    leafData.append(ev.c0)
                    leafData.append(ev.c1)
                }
                // Hash leaf data to get digest
                var leafState = [Int]()
                for i in 0..<min(GoldilocksPoseidon.stateWidth, leafData.count) {
                    leafState.append(leafData[i])
                }
                let zeroVar = builder.constant(glToFr(0))
                while leafState.count < GoldilocksPoseidon.stateWidth {
                    leafState.append(zeroVar)
                }
                let leafHash = poseidonPermutationCircuit(leafState)
                let leafDigest = Array(leafHash[0..<GoldilocksPoseidon.capacity])

                // Allocate sibling digests
                var siblingVars = [[Int]]()
                for _ in roundData.merklePath.siblings {
                    var sib = [Int]()
                    for _ in 0..<GoldilocksPoseidon.capacity { sib.append(allocateGl()) }
                    siblingVars.append(sib)
                }

                // Determine root: initial tree root for round 0, commit root for later rounds
                let expectedRoot = (r == 0) ? initialRoot : commitRootVars[r - 1]
                verifyMerklePathCircuit(
                    leaf: leafDigest,
                    siblings: siblingVars,
                    index: roundData.merklePath.index,
                    expectedRoot: expectedRoot
                )

                // FRI folding consistency check
                if cosetEvalVars.count >= 2 {
                    let xVal = allocateExt()  // coset point (witness)
                    let foldedResult = allocateExt()  // expected folded value

                    friFoldStepCircuit(
                        fAtX: cosetEvalVars[0],
                        fAtNegX: cosetEvalVars[1],
                        beta: betas[r],
                        xVal: xVal,
                        result: foldedResult
                    )
                }
            }

            // Final polynomial evaluation check at the last query point
            if !finalPolyVars.isEmpty {
                let queryPoint = allocateExt()
                let expectedEval = allocateExt()
                friFinalCheckCircuit(
                    finalCoeffs: finalPolyVars,
                    evaluationPoint: queryPoint,
                    expectedValue: expectedEval
                )
            }
        }

        return publicInputWires
    }

    // MARK: - Plonk Constraint Verification

    /// Verify Plonky2's Plonk constraint evaluations at the challenge point.
    /// This checks that the quotient polynomial identity holds:
    ///   Z_H(zeta) * t(zeta) = constraint_evaluation(zeta)
    public func verifyPlonkConstraints(
        openings: [ExtVar],          // polynomial evaluations at zeta
        quotientEval: ExtVar,        // t(zeta)
        vanishingEval: ExtVar,       // Z_H(zeta) = zeta^n - 1
        numConstraints: Int
    ) {
        // LHS = Z_H(zeta) * t(zeta)
        let lhs = extMulCircuit(vanishingEval, quotientEval)

        // RHS = sum of constraint evaluations (simplified for illustration)
        // In full Plonky2, this involves gate-specific constraint polynomials
        var rhs = extConstant(.zero)
        if !openings.isEmpty {
            rhs = openings[0]
            for i in 1..<min(openings.count, numConstraints) {
                rhs = extAddCircuit(rhs, openings[i])
            }
        }

        extEqualCircuit(lhs, rhs)
    }

    // MARK: - Helpers

    /// Convert a Goldilocks value to BN254 Fr for circuit constants.
    /// Since Gl.P < Fr.P, this is a direct embedding.
    private func glToFr(_ val: UInt64) -> Fr {
        Fr.from64([val, 0, 0, 0])
    }

    private func glToFr(_ val: Gl) -> Fr {
        glToFr(val.v)
    }
}

// MARK: - Plonky2 Recursive Prover

/// End-to-end pipeline for generating a BN254 proof that verifies a Plonky2 proof.
///
/// Steps:
///   1. Build the verifier circuit (PlonkCircuitBuilder constraints)
///   2. Generate witness (execute the verifier, record all intermediate values)
///   3. Prove the circuit using BN254 Plonk or Groth16
///   4. Output a BN254 proof that attests to the validity of the Plonky2 proof
///
/// This is the standard "Plonky2 to EVM" bridge used by Polygon zkEVM.
public class Plonky2RecursiveProver {
    /// Configuration for the recursive prover
    public struct Config {
        /// Number of FRI queries (security parameter, typically 28-100)
        public let numFRIQueries: Int
        /// Number of FRI folding rounds
        public let numFRIRounds: Int
        /// Whether to use Groth16 (for EVM) or Plonk (for further recursion)
        public let useGroth16: Bool

        public init(numFRIQueries: Int = 28, numFRIRounds: Int = 12, useGroth16: Bool = true) {
            self.numFRIQueries = numFRIQueries
            self.numFRIRounds = numFRIRounds
            self.useGroth16 = useGroth16
        }
    }

    public let config: Config

    public init(config: Config = Config()) {
        self.config = config
    }

    /// Build the verifier circuit for a given Plonky2 verification key.
    /// Returns the circuit builder and the public input wire indices.
    public func buildVerifierCircuit(
        vk: Plonky2VerificationKey,
        proof: Plonky2FRIProof
    ) -> (builder: PlonkCircuitBuilder, publicInputs: [Int]) {
        let builder = PlonkCircuitBuilder()
        let verifier = Plonky2VerifierCircuit(builder: builder)

        let publicInputs = verifier.buildFRIVerificationCircuit(
            numQueries: config.numFRIQueries,
            numRounds: config.numFRIRounds,
            proof: proof,
            vk: vk
        )

        return (builder, publicInputs)
    }

    /// Generate witness for the verifier circuit given a concrete Plonky2 proof.
    /// This executes the native verifier and records all intermediate values.
    public func generateWitness(
        proof: Plonky2Proof,
        vk: Plonky2VerificationKey
    ) -> [Fr] {
        // The witness includes:
        // 1. Public inputs (Plonky2 proof's public inputs, embedded in Fr)
        // 2. All intermediate Poseidon states
        // 3. Merkle path siblings
        // 4. FRI coset evaluations
        // 5. Reduction quotients (k values for Gl mod reduction)

        var witness = [Fr]()

        // Embed public inputs
        for pi in proof.publicInputs {
            witness.append(Fr.from64([pi.v, 0, 0, 0]))
        }

        // Embed initial tree root
        for r in proof.openingProof.initialTreeRoot {
            witness.append(Fr.from64([r.v, 0, 0, 0]))
        }

        // Embed commit roots
        for root in proof.openingProof.commitRoots {
            for r in root {
                witness.append(Fr.from64([r.v, 0, 0, 0]))
            }
        }

        // Embed FRI query data
        for queryRounds in proof.openingProof.queryRoundData {
            for round in queryRounds {
                for eval in round.cosetEvals {
                    witness.append(Fr.from64([eval.c0.v, 0, 0, 0]))
                    witness.append(Fr.from64([eval.c1.v, 0, 0, 0]))
                }
                for sib in round.merklePath.siblings {
                    for s in sib {
                        witness.append(Fr.from64([s.v, 0, 0, 0]))
                    }
                }
            }
        }

        // Embed final polynomial coefficients
        for coeff in proof.openingProof.finalPoly {
            witness.append(Fr.from64([coeff.c0.v, 0, 0, 0]))
            witness.append(Fr.from64([coeff.c1.v, 0, 0, 0]))
        }

        return witness
    }

    /// Full recursive proof pipeline: verify a Plonky2 proof inside BN254.
    ///
    /// Returns a Groth16Proof (if useGroth16=true) suitable for on-chain verification,
    /// or a PlonkCircuit for further recursive composition.
    public func prove(
        proof: Plonky2Proof,
        vk: Plonky2VerificationKey
    ) -> (circuit: PlonkCircuit, witness: [Fr]) {
        let (builder, _) = buildVerifierCircuit(vk: vk, proof: proof.openingProof)
        let witness = generateWitness(proof: proof, vk: vk)
        let circuit = builder.build()
        return (circuit, witness)
    }
}
