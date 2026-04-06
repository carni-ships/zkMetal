// XHash: STARK-friendly hash functions for the Goldilocks field
// Based on the XHash specification (ePrint 2023/1045) and the production
// RPX implementation from Polygon Miden (github.com/0xMiden/crypto).
//
// XHash12: 12-element state, rate=8, capacity=4, full S-box layers
//   Permutation: (FB)(E)(FB)(E)(FB)(E)(M)  — 7 logical rounds
//   - FB: MDS -> ARK1 -> x^7 -> MDS -> ARK2 -> x^(1/7)
//   - E:  ARK1 -> cubic extension x^7 on 4 Fp3 triplets
//   - M:  MDS -> ARK1
//
// XHash8: Same structure but extension rounds apply x^7 only to
//   the first 8 elements (rate portion) while identity on capacity.
//
// Field: Goldilocks (p = 2^64 - 2^32 + 1)
// S-box degree: 7
// Security target: 128-bit collision resistance

import Foundation

// MARK: - Configuration

public enum XHashConfig {
    public static let stateWidth = 12
    public static let rate = 8
    public static let capacity = 4
    public static let numRounds = 7  // 3 FB + 3 E + 1 M
    public static let sboxDegree: UInt64 = 7
}

// MARK: - MDS Matrix (12x12 circulant)
// First row: [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8]
// From Miden RPO/RPX specification.

private let MDS_FIRST_ROW: [UInt64] = [7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8]

private let MDS_MATRIX: [[UInt64]] = {
    let n = 12
    var m = [[UInt64]](repeating: [UInt64](repeating: 0, count: n), count: n)
    for row in 0..<n {
        for col in 0..<n {
            m[row][col] = MDS_FIRST_ROW[(col + n - row) % n]
        }
    }
    return m
}()

// MARK: - Round Constants (from Miden RPX production implementation)
// ARK1: applied in the first half of each round
// ARK2: applied in the second half of FB rounds

private let ARK1: [[UInt64]] = [
    // Round 0 (FB)
    [5789762306288267392, 6522564764413701783, 17809893479458208203,
     107145243989736508, 6388978042437517382, 15844067734406016715,
     9975000513555218239, 3344984123768313364, 9959189626657347191,
     12960773468763563665, 9602914297752488475, 16657542370200465908],
    // Round 1 (E)
    [12987190162843096997, 653957632802705281, 4441654670647621225,
     4038207883745915761, 5613464648874830118, 13222989726778338773,
     3037761201230264149, 16683759727265180203, 8337364536491240715,
     3227397518293416448, 8110510111539674682, 2872078294163232137],
    // Round 2 (FB)
    [18072785500942327487, 6200974112677013481, 17682092219085884187,
     10599526828986756440, 975003873302957338, 8264241093196931281,
     10065763900435475170, 2181131744534710197, 6317303992309418647,
     1401440938888741532, 8884468225181997494, 13066900325715521532],
    // Round 3 (E)
    [5674685213610121970, 5759084860419474071, 13943282657648897737,
     1352748651966375394, 17110913224029905221, 1003883795902368422,
     4141870621881018291, 8121410972417424656, 14300518605864919529,
     13712227150607670181, 17021852944633065291, 6252096473787587650],
    // Round 4 (FB)
    [4887609836208846458, 3027115137917284492, 9595098600469470675,
     10528569829048484079, 7864689113198939815, 17533723827845969040,
     5781638039037710951, 17024078752430719006, 109659393484013511,
     7158933660534805869, 2955076958026921730, 7433723648458773977],
    // Round 5 (E)
    [16308865189192447297, 11977192855656444890, 12532242556065780287,
     14594890931430968898, 7291784239689209784, 5514718540551361949,
     10025733853830934803, 7293794580341021693, 6728552937464861756,
     6332385040983343262, 13277683694236792804, 2600778905124452676],
    // Round 6 (M)
    [7123075680859040534, 1034205548717903090, 7717824418247931797,
     3019070937878604058, 11403792746066867460, 10280580802233112374,
     337153209462421218, 13333398568519923717, 3596153696935337464,
     8104208463525993784, 14345062289456085693, 17036731477169661256],
]

private let ARK2: [[UInt64]] = [
    // Round 0 (FB second half)
    [6077062762357204287, 15277620170502011191, 5358738125714196705,
     14233283787297595718, 13792579614346651365, 11614812331536767105,
     14871063686742261166, 10148237148793043499, 4457428952329675767,
     15590786458219172475, 10063319113072092615, 14200078843431360086],
    // Round 1 (E — ARK2 not used for E rounds, placeholder)
    [6202948458916099932, 17690140365333231091, 3595001575307484651,
     373995945117666487, 1235734395091296013, 14172757457833931602,
     707573103686350224, 15453217512188187135, 219777875004506018,
     17876696346199469008, 17731621626449383378, 2897136237748376248],
    // Round 2 (FB second half)
    [8023374565629191455, 15013690343205953430, 4485500052507912973,
     12489737547229155153, 9500452585969030576, 2054001340201038870,
     12420704059284934186, 355990932618543755, 9071225051243523860,
     12766199826003448536, 9045979173463556963, 12934431667190679898],
    // Round 3 (E — placeholder)
    [18389244934624494276, 16731736864863925227, 4440209734760478192,
     17208448209698888938, 8739495587021565984, 17000774922218161967,
     13533282547195532087, 525402848358706231, 16987541523062161972,
     5466806524462797102, 14512769585918244983, 10973956031244051118],
    // Round 4 (FB second half)
    [6982293561042362913, 14065426295947720331, 16451845770444974180,
     7139138592091306727, 9012006439959783127, 14619614108529063361,
     1394813199588124371, 4635111139507788575, 16217473952264203365,
     10782018226466330683, 6844229992533662050, 7446486531695178711],
    // Round 5 (E — placeholder)
    [3736792340494631448, 577852220195055341, 6689998335515779805,
     13886063479078013492, 14358505101923202168, 7744142531772274164,
     16135070735728404443, 12290902521256031137, 12059913662657709804,
     16456018495793751911, 4571485474751953524, 17200392109565783176],
    // Round 6 (M — ARK2 not used, placeholder)
    [17130398059294018733, 519782857322261988, 9625384390925085478,
     1664893052631119222, 7629576092524553570, 3485239601103661425,
     9755891797164033838, 15218148195153269027, 16460604813734957368,
     9643968136937729763, 3611348709641382851, 18256379591337759196],
]

// MARK: - S-box: x^7 over Goldilocks

@inline(__always)
private func glSbox(_ x: Gl) -> Gl {
    let x2 = glSqr(x)
    let x3 = glMul(x2, x)
    let x6 = glSqr(x3)
    return glMul(x6, x)
}

// MARK: - Inverse S-box: x^(1/7) over Goldilocks
// 7^(-1) mod (p-1) = 10540996611094048183
// Uses the optimized addition chain from Miden.

@inline(__always)
private func glInvSbox(_ x: Gl) -> Gl {
    // Compute x^10540996611094048183 using the Miden addition chain.
    // 10540996611094048183 = 0x9249249249249247
    // Binary: 1001001001001001001001001001000110110110110110110110110110110111
    //
    // Chain: t1 = x^2, t2 = x^4, then accumulate via squaring+multiplying

    let t1 = glSqr(x)                           // x^2
    let t2 = glSqr(t1)                           // x^4

    // x^(100100) = x^36
    var t3 = t2
    for _ in 0..<3 { t3 = glSqr(t3) }           // x^32
    t3 = glMul(t3, t2)                            // x^36 = x^(100100)

    // x^(100100100100) = x^(0x924)
    var t4 = t3
    for _ in 0..<6 { t4 = glSqr(t4) }
    t4 = glMul(t4, t3)

    // x^(100100100100100100100100)
    var t5 = t4
    for _ in 0..<12 { t5 = glSqr(t5) }
    t5 = glMul(t5, t4)

    // x^(100100100100100100100100100100)
    var t6 = t5
    for _ in 0..<6 { t6 = glSqr(t6) }
    t6 = glMul(t6, t3)

    // x^(1001001001001001001001001001000100100100100100100100100100100)
    var t7 = t6
    for _ in 0..<31 { t7 = glSqr(t7) }
    t7 = glMul(t7, t6)

    // Final: x^(1001001001001001001001001001000110110110110110110110110110110111)
    var a = glSqr(t7)
    a = glMul(a, t6)
    a = glSqr(a)
    a = glSqr(a)
    let b = glMul(glMul(t1, t2), x)  // x^7
    return glMul(a, b)
}

// MARK: - MDS Matrix Multiplication

@inline(__always)
private func glMdsMultiply(_ state: inout [Gl]) {
    var result = [Gl](repeating: Gl.zero, count: 12)
    for i in 0..<12 {
        var acc = Gl.zero
        for j in 0..<12 {
            acc = glAdd(acc, glMul(state[j], Gl(v: MDS_MATRIX[i][j])))
        }
        result[i] = acc
    }
    state = result
}

// MARK: - Add Round Constants

@inline(__always)
private func glAddConstants(_ state: inout [Gl], _ constants: [UInt64]) {
    for i in 0..<12 {
        state[i] = glAdd(state[i], Gl(v: constants[i]))
    }
}

// MARK: - Cubic Extension Field: Fp3 = Fp[x]/(x^3 - x - 1)
// Used for the E (extension) rounds in XHash12.
// Element: a + b*phi + c*phi^2 where phi^3 = phi + 1

@inline(__always)
private func fp3Mul(_ a: (Gl, Gl, Gl), _ b: (Gl, Gl, Gl)) -> (Gl, Gl, Gl) {
    let a0b0 = glMul(a.0, b.0)
    let a1b1 = glMul(a.1, b.1)
    let a2b2 = glMul(a.2, b.2)

    let sum01a = glAdd(a.0, a.1)
    let sum01b = glAdd(b.0, b.1)
    let a0b0_a0b1_a1b0_a1b1 = glMul(sum01a, sum01b)

    let sum02a = glAdd(a.0, a.2)
    let sum02b = glAdd(b.0, b.2)
    let a0b0_a0b2_a2b0_a2b2 = glMul(sum02a, sum02b)

    let sum12a = glAdd(a.1, a.2)
    let sum12b = glAdd(b.1, b.2)
    let a1b1_a1b2_a2b1_a2b2 = glMul(sum12a, sum12b)

    let a0b0_minus_a1b1 = glSub(a0b0, a1b1)

    // c0 = a0b0 + a1b2 + a2b1 = (a1b1 + a1b2 + a2b1 + a2b2) + (a0b0 - a1b1) - a2b2
    let c0 = glSub(glAdd(a1b1_a1b2_a2b1_a2b2, a0b0_minus_a1b1), a2b2)

    // c1 = a0b1 + a1b0 + a1b2 + a2b1 + a2b2
    let double_a1b1 = glAdd(a1b1, a1b1)
    let c1 = glSub(glSub(glAdd(a0b0_a0b1_a1b0_a1b1, a1b1_a1b2_a2b1_a2b2), double_a1b1), a0b0)

    // c2 = a0b2 + a1b1 + a2b0 + a2b2 = (a0b0 + a0b2 + a2b0 + a2b2) - (a0b0 - a1b1)
    let c2 = glSub(a0b0_a0b2_a2b0_a2b2, a0b0_minus_a1b1)

    return (c0, c1, c2)
}

@inline(__always)
private func fp3Square(_ a: (Gl, Gl, Gl)) -> (Gl, Gl, Gl) {
    let a2sq = glSqr(a.2)
    let a1a2 = glMul(a.1, a.2)
    let double_a1a2 = glAdd(a1a2, a1a2)

    let c0 = glAdd(glSqr(a.0), double_a1a2)

    let a0a1 = glMul(a.0, a.1)
    let c1 = glAdd(glAdd(a0a1, a1a2), glAdd(glAdd(a0a1, a1a2), a2sq))

    let a0a2 = glMul(a.0, a.2)
    let c2 = glAdd(glAdd(a0a2, a0a2), glAdd(glSqr(a.1), a2sq))

    return (c0, c1, c2)
}

/// Compute x^7 in Fp3: x -> x^2 -> x^3 -> x^6 -> x^7
@inline(__always)
private func fp3Power7(_ a: (Gl, Gl, Gl)) -> (Gl, Gl, Gl) {
    let a2 = fp3Square(a)
    let a3 = fp3Mul(a2, a)
    let a6 = fp3Square(a3)
    return fp3Mul(a6, a)
}

// MARK: - XHash12 Permutation

/// Applies the XHash12 permutation to a 12-element Goldilocks state.
/// Structure: (FB)(E)(FB)(E)(FB)(E)(M)
public func xhash12Permutation(_ input: [Gl]) -> [Gl] {
    precondition(input.count == 12)
    var state = input

    // Round 0: FB
    applyFBRound(&state, round: 0)
    // Round 1: E
    applyExtRound(&state, round: 1, partial: false)
    // Round 2: FB
    applyFBRound(&state, round: 2)
    // Round 3: E
    applyExtRound(&state, round: 3, partial: false)
    // Round 4: FB
    applyFBRound(&state, round: 4)
    // Round 5: E
    applyExtRound(&state, round: 5, partial: false)
    // Round 6: M (final linear layer)
    glMdsMultiply(&state)
    glAddConstants(&state, ARK1[6])

    return state
}

/// In-place XHash12 permutation.
public func xhash12Permutation(state: inout [Gl]) {
    precondition(state.count == 12)

    applyFBRound(&state, round: 0)
    applyExtRound(&state, round: 1, partial: false)
    applyFBRound(&state, round: 2)
    applyExtRound(&state, round: 3, partial: false)
    applyFBRound(&state, round: 4)
    applyExtRound(&state, round: 5, partial: false)
    glMdsMultiply(&state)
    glAddConstants(&state, ARK1[6])
}

// MARK: - XHash8 Permutation
// Same as XHash12 but extension rounds only apply cubic S-box to
// rate elements (indices 0..7), leaving capacity (8..11) unchanged.

/// Applies the XHash8 permutation to a 12-element Goldilocks state.
public func xhash8Permutation(_ input: [Gl]) -> [Gl] {
    precondition(input.count == 12)
    var state = input

    applyFBRound(&state, round: 0)
    applyExtRound(&state, round: 1, partial: true)
    applyFBRound(&state, round: 2)
    applyExtRound(&state, round: 3, partial: true)
    applyFBRound(&state, round: 4)
    applyExtRound(&state, round: 5, partial: true)
    glMdsMultiply(&state)
    glAddConstants(&state, ARK1[6])

    return state
}

/// In-place XHash8 permutation.
public func xhash8Permutation(state: inout [Gl]) {
    precondition(state.count == 12)

    applyFBRound(&state, round: 0)
    applyExtRound(&state, round: 1, partial: true)
    applyFBRound(&state, round: 2)
    applyExtRound(&state, round: 3, partial: true)
    applyFBRound(&state, round: 4)
    applyExtRound(&state, round: 5, partial: true)
    glMdsMultiply(&state)
    glAddConstants(&state, ARK1[6])
}

// MARK: - Round Functions

/// FB round: MDS -> ARK1 -> x^7 -> MDS -> ARK2 -> x^(1/7)
@inline(__always)
private func applyFBRound(_ state: inout [Gl], round: Int) {
    // Forward half
    glMdsMultiply(&state)
    glAddConstants(&state, ARK1[round])
    for i in 0..<12 { state[i] = glSbox(state[i]) }

    // Backward half
    glMdsMultiply(&state)
    glAddConstants(&state, ARK2[round])
    for i in 0..<12 { state[i] = glInvSbox(state[i]) }
}

/// E round: ARK1 -> cubic ext x^7 on 4 triplets
/// If partial=true, only apply to first 8 elements (XHash8 mode).
@inline(__always)
private func applyExtRound(_ state: inout [Gl], round: Int, partial: Bool) {
    glAddConstants(&state, ARK1[round])

    // Decompose state into 4 Fp3 elements and apply x^7
    let numTriplets = partial ? 2 : 4  // XHash8: 2 triplets (6 elems), XHash12: 4
    let numElements = partial ? 8 : 12

    // For XHash8, we apply to first 8 elements but group as:
    // triplet 0: [0,1,2], triplet 1: [3,4,5], and elements [6,7] get individual x^7
    if partial {
        // Triplets 0 and 1
        for t in 0..<2 {
            let base = t * 3
            let ext = fp3Power7((state[base], state[base+1], state[base+2]))
            state[base] = ext.0
            state[base+1] = ext.1
            state[base+2] = ext.2
        }
        // Elements 6 and 7: apply scalar x^7
        state[6] = glSbox(state[6])
        state[7] = glSbox(state[7])
        // Elements 8..11 (capacity): identity — no S-box
    } else {
        // XHash12: all 4 triplets
        for t in 0..<4 {
            let base = t * 3
            let ext = fp3Power7((state[base], state[base+1], state[base+2]))
            state[base] = ext.0
            state[base+1] = ext.1
            state[base+2] = ext.2
        }
    }
}

// MARK: - XHash12 Hash (Sponge Construction)

/// Hash two 4-element Goldilocks digests using XHash12 (2-to-1 compression).
/// Rate=8, capacity=4. Output: first 4 elements of the squeezed state.
public func xhash12Merge(left: [Gl], right: [Gl]) -> [Gl] {
    precondition(left.count == 4 && right.count == 4)
    var state = [Gl](repeating: Gl.zero, count: 12)
    for i in 0..<4 { state[i] = left[i] }
    for i in 0..<4 { state[i + 4] = right[i] }
    xhash12Permutation(state: &state)
    return Array(state[0..<4])
}

/// Hash a variable-length array of Goldilocks elements using XHash12 sponge.
/// Rate=8, capacity=4. Returns first 4 elements as digest.
public func xhash12Hash(_ inputs: [Gl]) -> [Gl] {
    if inputs.isEmpty { return [Gl](repeating: Gl.zero, count: 4) }

    var state = [Gl](repeating: Gl.zero, count: 12)
    var i = 0
    while i < inputs.count {
        // Absorb up to 8 elements into rate portion
        for j in 0..<8 {
            if i + j < inputs.count {
                state[j] = glAdd(state[j], inputs[i + j])
            }
        }
        xhash12Permutation(state: &state)
        i += 8
    }
    return Array(state[0..<4])
}

/// Hash a single element using XHash12.
public func xhash12HashSingle(_ x: Gl) -> [Gl] {
    return xhash12Hash([x])
}

// MARK: - XHash8 Hash (Sponge Construction)

/// Hash two 4-element Goldilocks digests using XHash8 (2-to-1 compression).
public func xhash8Merge(left: [Gl], right: [Gl]) -> [Gl] {
    precondition(left.count == 4 && right.count == 4)
    var state = [Gl](repeating: Gl.zero, count: 12)
    for i in 0..<4 { state[i] = left[i] }
    for i in 0..<4 { state[i + 4] = right[i] }
    xhash8Permutation(state: &state)
    return Array(state[0..<4])
}

/// Hash a variable-length array of Goldilocks elements using XHash8 sponge.
public func xhash8Hash(_ inputs: [Gl]) -> [Gl] {
    if inputs.isEmpty { return [Gl](repeating: Gl.zero, count: 4) }

    var state = [Gl](repeating: Gl.zero, count: 12)
    var i = 0
    while i < inputs.count {
        for j in 0..<8 {
            if i + j < inputs.count {
                state[j] = glAdd(state[j], inputs[i + j])
            }
        }
        xhash8Permutation(state: &state)
        i += 8
    }
    return Array(state[0..<4])
}

/// Hash a single element using XHash8.
public func xhash8HashSingle(_ x: Gl) -> [Gl] {
    return xhash8Hash([x])
}
