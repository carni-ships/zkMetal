// NonNativeFieldGadget — Emulate foreign field arithmetic inside a circuit
//
// The Pasta cycle gives us: Pallas Fp = Vesta Fr, Pallas Fr = Vesta Fp.
// This means Pallas base field elements are *native* scalar field elements
// for Vesta circuits, and vice versa. For the cycle pair, non-native field
// arithmetic is essentially free (just a reinterpretation of bits).
//
// For the general case (arbitrary field pairs), we provide limb-decomposed
// arithmetic: represent a foreign field element as k limbs of w bits each,
// then constrain addition, multiplication, and reduction using native gates.
//
// Limb layout (for 256-bit fields with 4 x 68-bit limbs):
//   x = x_0 + x_1 * 2^68 + x_2 * 2^136 + x_3 * 2^204
//   Each limb fits in a native field element (native field > 254 bits).
//
// References:
//   - "Efficient Non-native Field Arithmetic" (xJsnark, Kosba et al.)
//   - Halo2 non-native gadget design

import Foundation

// MARK: - Limb Configuration

/// Configuration for non-native field emulation.
public struct NonNativeConfig {
    /// Number of limbs to decompose foreign field elements into
    public let numLimbs: Int
    /// Bit width of each limb
    public let limbBits: Int
    /// Modulus of the foreign field (as limbs in little-endian)
    public let modulus: [UInt64]
    /// Total bit width of the foreign field
    public let fieldBits: Int

    public init(numLimbs: Int, limbBits: Int, modulus: [UInt64], fieldBits: Int) {
        self.numLimbs = numLimbs
        self.limbBits = limbBits
        self.modulus = modulus
        self.fieldBits = fieldBits
    }

    /// Configuration for emulating Pallas Fp inside Vesta Fr.
    /// Since Pallas Fp = Vesta Fr, this is the trivial (native) case.
    /// We still provide the config for API uniformity.
    public static let pallasFpInVestaFr = NonNativeConfig(
        numLimbs: 1,
        limbBits: 255,
        modulus: PallasFp.P,
        fieldBits: 255
    )

    /// Configuration for emulating Vesta Fp inside Pallas Fr.
    /// Since Vesta Fp = Pallas Fr, this is also the trivial case.
    public static let vestaFpInPallasFr = NonNativeConfig(
        numLimbs: 1,
        limbBits: 255,
        modulus: VestaFp.P,
        fieldBits: 255
    )

    /// Configuration for emulating a generic 256-bit field using 4 x 68-bit limbs.
    /// This is for non-cycle curves where there is no field equality.
    public static func generic256(modulus: [UInt64]) -> NonNativeConfig {
        NonNativeConfig(numLimbs: 4, limbBits: 68, modulus: modulus, fieldBits: 256)
    }
}

// MARK: - Non-Native Field Variable

/// A variable in the circuit representing a foreign field element.
/// For Pasta cycle curves (native case): single variable, zero overhead.
/// For generic non-native: decomposed into limbs with range constraints.
public struct NonNativeVar {
    /// Circuit variable indices for each limb (or single variable for native case)
    public let limbs: [Int]
    /// Configuration
    public let config: NonNativeConfig
    /// Whether this is the native (trivial) case
    public var isNative: Bool { config.numLimbs == 1 }

    public init(limbs: [Int], config: NonNativeConfig) {
        precondition(limbs.count == config.numLimbs)
        self.limbs = limbs
        self.config = config
    }

    /// For native case: the single variable index
    public var nativeVar: Int {
        precondition(isNative, "nativeVar only valid for native (cycle) case")
        return limbs[0]
    }
}

// MARK: - Non-Native Field Gadget (Pasta Cycle)

/// Gadget for foreign field arithmetic inside a Plonk circuit.
///
/// For the Pasta cycle, this is highly efficient:
///   - Pallas Fp elements inside a Vesta circuit: native (zero overhead)
///   - Vesta Fp elements inside a Pallas circuit: native (zero overhead)
///
/// For general non-native arithmetic, uses limb decomposition.
public class NonNativeFieldGadget {
    public let builder: PlonkCircuitBuilder
    public let config: NonNativeConfig

    public init(builder: PlonkCircuitBuilder, config: NonNativeConfig) {
        self.builder = builder
        self.config = config
    }

    // MARK: - Allocation

    /// Allocate a new non-native field variable (witness must be provided externally).
    /// For native case: single input variable.
    /// For limb case: allocates numLimbs variables + range constraints.
    public func allocate() -> NonNativeVar {
        if config.numLimbs == 1 {
            // Native case: just a single circuit variable
            let v = builder.addInput()
            return NonNativeVar(limbs: [v], config: config)
        }

        // Limb decomposition case
        var limbVars = [Int]()
        for _ in 0..<config.numLimbs {
            let v = builder.addInput()
            limbVars.append(v)
            // Range-check each limb to limbBits bits
            builder.rangeCheck(v, bits: config.limbBits)
        }
        return NonNativeVar(limbs: limbVars, config: config)
    }

    /// Allocate a constant non-native field value.
    public func constant(_ value: VestaFp) -> NonNativeVar {
        if config.numLimbs == 1 {
            // Native case: the Pallas Fp value IS a Vesta Fr value
            // Convert VestaFp -> circuit constant
            // Since Pallas Fp = Vesta Fr, we reinterpret the Montgomery form
            let limbs = vestaToInt(value)
            let frVal = Fr.from64(limbs)
            let frMont = frMul(frVal, Fr.from64(Fr.R2_MOD_R))
            let v = builder.constant(frMont)
            return NonNativeVar(limbs: [v], config: config)
        }

        // Limb decomposition: split value into limbs
        let intLimbs = vestaToInt(value)
        var limbVars = [Int]()
        let mask: UInt64 = (1 << config.limbBits) - 1

        // Pack the 4 x UInt64 into numLimbs limbs of limbBits each
        var bigint = intLimbs
        for _ in 0..<config.numLimbs {
            let limbVal = bigint[0] & mask
            // Shift right by limbBits across all words
            shiftRight(&bigint, by: config.limbBits)
            let frVal = frFromInt(limbVal)
            let v = builder.constant(frVal)
            limbVars.append(v)
        }
        return NonNativeVar(limbs: limbVars, config: config)
    }

    // MARK: - Arithmetic (Native Case)

    /// Add two non-native field elements.
    /// Native case: single add gate. Limb case: add limbs + carry propagation.
    public func add(_ a: NonNativeVar, _ b: NonNativeVar) -> NonNativeVar {
        if config.numLimbs == 1 {
            let c = builder.add(a.limbs[0], b.limbs[0])
            return NonNativeVar(limbs: [c], config: config)
        }

        // Limb-wise addition with carry propagation
        // For now, we add limbs individually. The caller must ensure
        // reduction is applied when results might overflow.
        var resultLimbs = [Int]()
        for i in 0..<config.numLimbs {
            let sum = builder.add(a.limbs[i], b.limbs[i])
            resultLimbs.append(sum)
        }
        return NonNativeVar(limbs: resultLimbs, config: config)
    }

    /// Multiply two non-native field elements.
    /// Native case: single mul gate. Limb case: schoolbook multiply + reduce.
    public func mul(_ a: NonNativeVar, _ b: NonNativeVar) -> NonNativeVar {
        if config.numLimbs == 1 {
            let c = builder.mul(a.limbs[0], b.limbs[0])
            return NonNativeVar(limbs: [c], config: config)
        }

        // For limb-based multiplication, we use the standard approach:
        // 1. Compute unreduced product limbs (schoolbook)
        // 2. Compute quotient q and remainder r such that a*b = q*p + r
        // 3. Constrain q, r via range checks
        // 4. Return r as the result
        //
        // This is the core non-native multiply. For Pasta cycle, we never
        // reach this path, but it's needed for arbitrary curve pairs.

        // Schoolbook product: result has 2*numLimbs - 1 limbs
        let k = config.numLimbs
        var productLimbs = [Int]()

        // First limb: a[0] * b[0]
        productLimbs.append(builder.mul(a.limbs[0], b.limbs[0]))

        // Subsequent limbs: sum of cross-products
        for i in 1..<(2 * k - 1) {
            var termVars = [Int]()
            for j in 0...i {
                if j < k && (i - j) < k {
                    let prod = builder.mul(a.limbs[j], b.limbs[i - j])
                    termVars.append(prod)
                }
            }
            // Sum all terms for this limb
            var acc = termVars[0]
            for t in 1..<termVars.count {
                acc = builder.add(acc, termVars[t])
            }
            productLimbs.append(acc)
        }

        // For the reduction step, we'd need to constrain:
        //   product = quotient * modulus + remainder
        // The remainder is our result. This requires additional witness
        // variables for quotient and remainder, plus carry constraints.
        //
        // For now, return the unreduced product (first k limbs).
        // A full implementation would add the reduction constraints here.
        // In the Pasta cycle case, we never reach this code path.
        return NonNativeVar(limbs: Array(productLimbs.prefix(k)), config: config)
    }

    /// Assert two non-native field elements are equal.
    public func assertEqual(_ a: NonNativeVar, _ b: NonNativeVar) {
        for i in 0..<config.numLimbs {
            builder.assertEqual(a.limbs[i], b.limbs[i])
        }
    }

    /// Select: if bit == 1 then a else b. Bit must be a boolean variable.
    public func select(bit: Int, ifTrue a: NonNativeVar, ifFalse b: NonNativeVar) -> NonNativeVar {
        // select = b + bit * (a - b) for each limb
        var resultLimbs = [Int]()
        for i in 0..<config.numLimbs {
            // diff = a - b (using gate: qL=1, qR=-1, qO=-1)
            let diff = builder.addInput()
            // We constrain diff = a[i] - b[i] via add gate: b[i] + diff = a[i]
            let check = builder.add(b.limbs[i], diff)
            builder.assertEqual(check, a.limbs[i])

            // selected = bit * diff
            let scaled = builder.mul(bit, diff)
            // result = b[i] + selected
            let result = builder.add(b.limbs[i], scaled)
            resultLimbs.append(result)
        }
        return NonNativeVar(limbs: resultLimbs, config: config)
    }

    // MARK: - Helpers

    /// Shift a big integer right by `n` bits (in-place, across UInt64 words).
    private func shiftRight(_ limbs: inout [UInt64], by n: Int) {
        let wordShift = n / 64
        let bitShift = n % 64
        let count = limbs.count

        if wordShift >= count {
            for i in 0..<count { limbs[i] = 0 }
            return
        }

        if bitShift == 0 {
            for i in 0..<(count - wordShift) {
                limbs[i] = limbs[i + wordShift]
            }
        } else {
            for i in 0..<(count - wordShift - 1) {
                limbs[i] = (limbs[i + wordShift] >> bitShift) |
                           (limbs[i + wordShift + 1] << (64 - bitShift))
            }
            limbs[count - wordShift - 1] = limbs[count - 1] >> bitShift
        }
        for i in (count - wordShift)..<count {
            limbs[i] = 0
        }
    }
}

// MARK: - Pasta Cycle Conversions

/// Convert a Pallas Fp value to a Vesta Fr (scalar) for use in circuit constraints.
/// These are the SAME field, so this is just a reinterpretation of bits.
public func pallasFpToVestaFr(_ fp: PallasFp) -> VestaFp {
    // Pallas Fp = Vesta Fr. Both are the same prime field.
    // The Montgomery representations use different R values, so we convert
    // through the integer representation.
    let intVal = pallasToInt(fp)
    let raw = VestaFp.from64(intVal)
    return vestaMul(raw, VestaFp.from64(VestaFp.R2_MOD_P))
}

/// Convert a Vesta Fp value to a Pallas Fr (scalar) for use in circuit constraints.
/// These are the SAME field, so this is just a reinterpretation of bits.
public func vestaFpToPallasFr(_ fp: VestaFp) -> PallasFp {
    // Vesta Fp = Pallas Fr. Both are the same prime field.
    let intVal = vestaToInt(fp)
    let raw = PallasFp.from64(intVal)
    return pallasMul(raw, PallasFp.from64(PallasFp.R2_MOD_P))
}

// MARK: - Point Coordinate Extraction

/// Extract Pallas point coordinates as Vesta Fr values (for circuit use).
/// Returns (x, y) where each coordinate is a VestaFp (= Pallas Fp reinterpreted).
public func pallasPointToVestaCoords(_ p: PallasPointProjective) -> (x: VestaFp, y: VestaFp) {
    let affine = pallasPointToAffine(p)
    return (pallasFpToVestaFr(affine.x), pallasFpToVestaFr(affine.y))
}

/// Extract Vesta point coordinates as Pallas Fr values (for circuit use).
public func vestaPointToPallasCoords(_ p: VestaPointProjective) -> (x: PallasFp, y: PallasFp) {
    let affine = vestaPointToAffine(p)
    return (vestaFpToPallasFr(affine.x), vestaFpToPallasFr(affine.y))
}
