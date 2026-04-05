// PlonkHashGates -- Hash function constraint gates for Plonk circuits
//
// Provides custom gates for constraining hash operations within Plonk:
//   - Poseidon2RoundGate: single Poseidon2 round (S-box + MDS + round constant)
//   - Poseidon2PermutationGate: full Poseidon2 permutation (chains round gates)
//   - MiMCRoundGate: single MiMC round (x^7 + round constant)
//
// These gates enable efficient in-circuit hash verification without requiring
// external lookup tables. The S-box and linear layer computations are expressed
// as polynomial constraints over the wire columns.

import Foundation
import NeonFieldOps

// MARK: - Poseidon2RoundGate

/// Constrains a single Poseidon2 round: S-box + linear layer + round constant addition.
///
/// Poseidon2 uses a state width of t elements. In a full round, every state element
/// passes through the S-box (x^5 for BN254). In a partial round, only the first
/// element does.
///
/// For a width-3 state (common in Merkle tree hashing):
///
/// Wire layout (2 rows):
///   Row 0: col0 = state_in[0], col1 = state_in[1], col2 = state_in[2]
///   Row 1: col0 = state_out[0], col1 = state_out[1], col2 = state_out[2]
///
/// Full round constraint for each element i:
///   state_out[i] = MDS[i] . (sbox(state_in[0]+rc[0]), sbox(state_in[1]+rc[1]), sbox(state_in[2]+rc[2]))
///   where sbox(x) = x^5
///
/// Partial round constraint:
///   Only state_in[0] goes through sbox; others pass through linearly.
///   temp[0] = sbox(state_in[0] + rc[0])
///   temp[i] = state_in[i] + rc[i]  for i > 0
///   state_out = MDS * temp
public struct Poseidon2RoundGate: CustomGate {
    public let name: String

    /// State width (number of field elements in the Poseidon state)
    public let width: Int

    /// Whether this is a full round (all S-boxes) or partial round (one S-box)
    public let isFullRound: Bool

    /// Round constants for this round (length = width)
    public let roundConstants: [Fr]

    /// MDS matrix (width x width), stored row-major: mds[i*width + j]
    public let mds: [Fr]

    /// S-box exponent (5 for BN254 Poseidon2)
    public let sboxExp: Int

    public init(width: Int, isFullRound: Bool, roundConstants: [Fr], mds: [Fr],
                sboxExp: Int = 5, roundIndex: Int = 0) {
        precondition(roundConstants.count == width, "Round constants must match state width")
        precondition(mds.count == width * width, "MDS matrix must be width x width")
        self.width = width
        self.isFullRound = isFullRound
        self.roundConstants = roundConstants
        self.mds = mds
        self.sboxExp = sboxExp
        self.name = "Poseidon2Round_\(isFullRound ? "full" : "partial")_\(roundIndex)"
    }

    public var queriedCells: [ColumnRef] {
        var cells = [ColumnRef]()
        for i in 0..<width {
            cells.append(ColumnRef(column: i, rotation: .cur))   // state_in[i]
            cells.append(ColumnRef(column: i, rotation: .next))  // state_out[i]
        }
        return cells
    }

    /// Apply the S-box: x^5 = x * x^2 * x^2
    private func sbox(_ x: Fr) -> Fr {
        if sboxExp == 5 {
            let x2 = frSqr(x)
            let x4 = frSqr(x2)
            return frMul(x, x4)
        }
        // Generic: x^exp via square-and-multiply
        var result = Fr.one
        var base = x
        var e = sboxExp
        while e > 0 {
            if e & 1 == 1 { result = frMul(result, base) }
            base = frSqr(base)
            e >>= 1
        }
        return result
    }

    public func evaluate(rotations: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        // Read state_in and state_out
        var stateIn = [Fr]()
        var stateOut = [Fr]()
        for i in 0..<width {
            stateIn.append(rotations[ColumnRef(column: i, rotation: .cur)] ?? Fr.zero)
            stateOut.append(rotations[ColumnRef(column: i, rotation: .next)] ?? Fr.zero)
        }

        // Apply round constants and S-box to get intermediate values
        var temp = [Fr](repeating: Fr.zero, count: width)
        for i in 0..<width {
            let withRC = frAdd(stateIn[i], roundConstants[i])
            if isFullRound || i == 0 {
                temp[i] = sbox(withRC)
            } else {
                temp[i] = withRC
            }
        }

        // Apply MDS matrix: expected_out[i] = sum_j mds[i*width+j] * temp[j]
        var result = Fr.zero
        for i in 0..<width {
            var expected = Fr.zero
            for j in 0..<width {
                expected = frAdd(expected, frMul(mds[i * width + j], temp[j]))
            }
            // Constraint: state_out[i] - expected = 0
            let diff = frSub(stateOut[i], expected)
            result = frAdd(result, frMul(diff, diff))
        }

        return result
    }
}

// MARK: - Poseidon2PermutationGate

/// Full Poseidon2 permutation as a chain of round gates.
///
/// This is a compound gate that expands into multiple Poseidon2RoundGate instances.
/// The standard Poseidon2 for BN254 uses:
///   - 4 full rounds (beginning)
///   - 56 partial rounds (middle)
///   - 4 full rounds (end)
///   Total: 64 rounds for width-3, or configurable.
///
/// Usage: call `expandRoundGates()` to get individual round gates that should
/// be registered with the CustomGateSet.
public struct Poseidon2PermutationGate {
    public let name = "Poseidon2Permutation"

    /// State width
    public let width: Int

    /// Number of full rounds at the beginning
    public let fullRoundsBegin: Int

    /// Number of partial rounds in the middle
    public let partialRounds: Int

    /// Number of full rounds at the end
    public let fullRoundsEnd: Int

    /// All round constants: roundConstants[round][element]
    public let allRoundConstants: [[Fr]]

    /// MDS matrix (shared across all rounds)
    public let mds: [Fr]

    /// Standard Poseidon2 for BN254 with width 3 (t=3)
    /// Uses placeholder round constants (in production, use generated constants)
    public init(width: Int = 3, fullRoundsBegin: Int = 4,
                partialRounds: Int = 56, fullRoundsEnd: Int = 4,
                allRoundConstants: [[Fr]], mds: [Fr]) {
        self.width = width
        self.fullRoundsBegin = fullRoundsBegin
        self.partialRounds = partialRounds
        self.fullRoundsEnd = fullRoundsEnd
        self.allRoundConstants = allRoundConstants
        self.mds = mds
    }

    /// Total number of rounds
    public var totalRounds: Int {
        fullRoundsBegin + partialRounds + fullRoundsEnd
    }

    /// Expand into individual round gates.
    ///
    /// Each round gate constrains 2 rows (input state -> output state).
    /// The total execution trace uses `totalRounds + 1` rows for the state chain.
    ///
    /// - Returns: Array of round gates in order
    public func expandRoundGates() -> [Poseidon2RoundGate] {
        precondition(allRoundConstants.count == totalRounds,
                     "Must provide round constants for all \(totalRounds) rounds")

        var gates = [Poseidon2RoundGate]()

        for r in 0..<totalRounds {
            let isFullRound: Bool
            if r < fullRoundsBegin {
                isFullRound = true
            } else if r < fullRoundsBegin + partialRounds {
                isFullRound = false
            } else {
                isFullRound = true
            }

            gates.append(Poseidon2RoundGate(
                width: width,
                isFullRound: isFullRound,
                roundConstants: allRoundConstants[r],
                mds: mds,
                roundIndex: r))
        }

        return gates
    }

    /// Generate wire variable mappings for the full permutation.
    ///
    /// - Parameter compiler: Constraint compiler for variable allocation
    /// - Returns: Array of (stateIn vars, stateOut vars) for each round
    public func allocateWires(compiler: PlonkConstraintCompiler) -> (
        inputVars: [Int], outputVars: [Int], intermediateVars: [[Int]]
    ) {
        // Input state variables
        let inputVars = compiler.addVariables(width)
        // Output state variables
        let outputVars = compiler.addVariables(width)

        // Intermediate state variables between rounds
        var intermediateVars = [[Int]]()
        for _ in 0..<(totalRounds - 1) {
            intermediateVars.append(compiler.addVariables(width))
        }

        return (inputVars: inputVars, outputVars: outputVars,
                intermediateVars: intermediateVars)
    }
}

// MARK: - MiMCRoundGate

/// Constrains a single MiMC round: out = (in + key + round_constant)^7
///
/// MiMC is a block cipher / hash function using the x^7 S-box (for BN254,
/// where gcd(7, p-1) = 1).
///
/// Wire layout (single row):
///   col0 = input, col1 = key, col2 = output
///
/// The round constant is baked into the gate at construction time.
///
/// Constraint: output - (input + key + rc)^7 = 0
///
/// To reduce constraint degree, we decompose x^7 = x * x^2 * x^4
/// using auxiliary variables:
///   col3 = x^2 (advice), col4 = x^4 (advice)
///
/// Sub-constraints:
///   1. col3 - (in + key + rc)^2 = 0
///   2. col4 - col3^2 = 0
///   3. output - (in + key + rc) * col3 * col4 = 0  (i.e., x * x^2 * x^4 = x^7)
public struct MiMCRoundGate: CustomGate {
    public let name: String

    /// Round constant for this round
    public let roundConstant: Fr

    public init(roundConstant: Fr, roundIndex: Int = 0) {
        self.roundConstant = roundConstant
        self.name = "MiMCRound_\(roundIndex)"
    }

    public var queriedCells: [ColumnRef] {
        [
            ColumnRef(column: 0, rotation: .cur),  // input
            ColumnRef(column: 1, rotation: .cur),  // key
            ColumnRef(column: 2, rotation: .cur),  // output
            ColumnRef(column: 3, rotation: .cur),  // x^2 (aux)
            ColumnRef(column: 4, rotation: .cur),  // x^4 (aux)
        ]
    }

    public func evaluate(rotations: [ColumnRef: Fr], challenges: [Fr]) -> Fr {
        let input = rotations[ColumnRef(column: 0, rotation: .cur)] ?? Fr.zero
        let key = rotations[ColumnRef(column: 1, rotation: .cur)] ?? Fr.zero
        let output = rotations[ColumnRef(column: 2, rotation: .cur)] ?? Fr.zero
        let x2Witness = rotations[ColumnRef(column: 3, rotation: .cur)] ?? Fr.zero
        let x4Witness = rotations[ColumnRef(column: 4, rotation: .cur)] ?? Fr.zero

        // x = input + key + round_constant
        let x = frAdd(frAdd(input, key), roundConstant)

        // Constraint 1: x2Witness - x^2
        let c1 = frSub(x2Witness, frSqr(x))

        // Constraint 2: x4Witness - x2Witness^2
        let c2 = frSub(x4Witness, frSqr(x2Witness))

        // Constraint 3: output - x * x2Witness * x4Witness
        let c3 = frSub(output, frMul(x, frMul(x2Witness, x4Witness)))

        // Sum of squares
        return frAdd(frAdd(frMul(c1, c1), frMul(c2, c2)), frMul(c3, c3))
    }
}
