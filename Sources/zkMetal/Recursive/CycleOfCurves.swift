// CycleOfCurves -- Curve cycle pairs for recursive proof composition
//
// A cycle of curves is a pair (E1, E2) where:
//   E1.Fr = E2.Fq  and  E1.Fq = E2.Fr
//
// This means arithmetic on E1's scalar field is *native* in an E2 circuit,
// and vice versa. This property is essential for efficient recursive SNARKs:
// the verifier circuit for an E1-proof runs natively in an E2-circuit without
// expensive non-native field emulation.
//
// Supported cycles:
//   1. BN254 <-> Grumpkin  (used by Aztec/Noir, Halo2)
//   2. Pallas <-> Vesta    (used by Zcash Orchard, Nova)
//
// References:
//   - "A cycle of pairing-friendly elliptic curves" (Guillevic 2019)
//   - Pasta curves specification (Bowe et al. 2019)
//   - Aztec Noir docs: BN254/Grumpkin cycle

import Foundation
import NeonFieldOps

// MARK: - Curve Cycle Protocol

/// A cycle of two elliptic curves where each curve's scalar field equals the
/// other's base field. This enables efficient recursive proof composition.
public protocol CurveCycle {
    /// The primary curve's scalar field element type
    associatedtype PrimaryFr
    /// The secondary curve's scalar field element type (= Primary base field)
    associatedtype SecondaryFr

    /// Name of this cycle (for diagnostics)
    static var name: String { get }

    /// Convert a primary scalar field element to a secondary base field element.
    /// This is a bitwise reinterpretation (same underlying prime, different
    /// Montgomery domains).
    static func primaryFrToSecondaryFq(_ x: PrimaryFr) -> SecondaryFr

    /// Convert a secondary scalar field element to a primary base field element.
    static func secondaryFrToPrimaryFq(_ x: SecondaryFr) -> PrimaryFr
}

// MARK: - BN254 <-> Grumpkin Cycle

/// The BN254/Grumpkin cycle of curves.
///
/// BN254:   y^2 = x^3 + 3     (pairing-friendly, base=Fp, scalar=Fr)
/// Grumpkin: y^2 = x^3 - 17   (inner curve, base=Fr, scalar=Fp)
///
/// Key relationships:
///   BN254 Fr = Grumpkin Fq  (Grumpkin's base field)
///   BN254 Fq = Grumpkin Fr  (Grumpkin's scalar field)
///
/// This cycle is used by Aztec's Noir proving system for recursive proofs.
/// BN254 provides cheap pairing verification on Ethereum L1, while Grumpkin
/// handles the recursive accumulation step.
public enum BN254GrumpkinCycle: CurveCycle {
    public typealias PrimaryFr = Fr      // BN254 scalar field
    public typealias SecondaryFr = Fp    // Grumpkin scalar field = BN254 base field

    public static let name = "BN254/Grumpkin"

    /// Convert BN254 Fr -> Grumpkin Fq (they are the same field).
    /// Both are the same prime, but may have different Montgomery R constants,
    /// so we convert through integer representation.
    public static func primaryFrToSecondaryFq(_ x: Fr) -> Fp {
        // BN254 Fr -> integer -> Grumpkin Fq (= BN254 Fr reinterpreted)
        // Grumpkin's base field IS BN254 Fr, so the Fr struct is already native.
        // For circuit use, we need to re-encode as the secondary field's Montgomery form.
        //
        // Since Grumpkin base field = BN254 Fr, and GrumpkinPointProjective uses Fr
        // for coordinates, this is actually an identity operation on the type level.
        // The conversion to Fp (BN254 base / Grumpkin scalar) goes through integers.
        let intVal = frToInt(x)
        let raw = Fp.from64(intVal)
        return fpMul(raw, Fp.from64(Fp.R2_MOD_P))
    }

    /// Convert Grumpkin Fr (= BN254 Fp) -> BN254 Fq (= BN254 Fr).
    public static func secondaryFrToPrimaryFq(_ x: Fp) -> Fr {
        let intVal = fpToInt(x)
        let raw = Fr.from64(intVal)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }

    // MARK: - BN254 Point <-> Grumpkin Coordinate Helpers

    /// Extract BN254 G1 point coordinates as Grumpkin-native field elements.
    ///
    /// BN254 G1 points have coordinates in Fp (BN254 base field).
    /// Grumpkin's scalar field is Fp, so these are native scalars in a Grumpkin circuit.
    /// However, for Grumpkin *circuit constraints* (which operate over Grumpkin's base = Fr),
    /// we need non-native field emulation for the Fp coordinates.
    ///
    /// Returns (x, y) as Fp values (native Grumpkin scalars).
    public static func bn254PointToGrumpkinScalars(_ p: PointProjective) -> (x: Fp, y: Fp) {
        guard let affine = pointToAffine(p) else {
            return (Fp.zero, Fp.zero)
        }
        // BN254 affine coordinates are already in Fp = Grumpkin scalar field
        return (affine.x, affine.y)
    }

    /// Extract Grumpkin point coordinates as BN254-native Fr elements.
    ///
    /// Grumpkin points have coordinates in Fr (BN254 scalar field).
    /// BN254's scalar field is Fr, so these are native in a BN254 circuit.
    /// This is the key efficiency win: verifying Grumpkin curve operations
    /// inside a BN254 circuit is cheap because the coordinates are native.
    public static func grumpkinPointToBN254Scalars(_ p: GrumpkinPointProjective) -> (x: Fr, y: Fr) {
        let affine = grumpkinPointToAffine(p)
        // Grumpkin affine coordinates are in Fr = BN254 scalar field (native)
        return (affine.x, affine.y)
    }

    // MARK: - Non-Native Field Configuration

    /// Configuration for emulating BN254 Fp inside a BN254 Fr circuit.
    /// Since Fp != Fr (different primes), this requires limb decomposition.
    /// 4 x 68-bit limbs cover the 254-bit Fp modulus.
    public static var fpInFrConfig: NonNativeConfig {
        NonNativeConfig.generic256(modulus: Fp.P)
    }

    /// Configuration for emulating BN254 Fr inside a Grumpkin Fr (= Fp) circuit.
    /// Since BN254 Fr != BN254 Fp (different primes), this requires limb decomposition.
    public static var frInFpConfig: NonNativeConfig {
        NonNativeConfig.generic256(modulus: Fr.P)
    }
}

// MARK: - Pallas <-> Vesta Cycle

/// The Pallas/Vesta (Pasta) cycle of curves.
///
/// Pallas: y^2 = x^3 + 5   (base = PallasFp, scalar = VestaFp)
/// Vesta:  y^2 = x^3 + 5   (base = VestaFp,  scalar = PallasFp)
///
/// Key relationships:
///   Pallas Fp = Vesta Fq  (Vesta's scalar field)
///   Vesta Fp  = Pallas Fq (Pallas's scalar field)
///
/// The Pasta cycle is elegant: both curves have the same equation (y^2=x^3+5)
/// and their fields are swapped. This means Pallas point coordinates are native
/// in Vesta circuits and vice versa, with ZERO non-native overhead.
///
/// Used by: Zcash Orchard, Nova/SuperNova, Lurk.
public enum PallasVestaCycle: CurveCycle {
    public typealias PrimaryFr = VestaFp   // Pallas scalar = Vesta base
    public typealias SecondaryFr = PallasFp // Vesta scalar = Pallas base

    public static let name = "Pallas/Vesta"

    /// Convert Pallas Fr (= VestaFp) -> Vesta Fq (= PallasFp).
    /// Since Pallas Fr = Vesta Fp and Vesta Fq = Pallas Fp, this converts
    /// between the two fields (which are the same prime with different R).
    public static func primaryFrToSecondaryFq(_ x: VestaFp) -> PallasFp {
        vestaFpToPallasFr(x)
    }

    /// Convert Vesta Fr (= PallasFp) -> Pallas Fq (= VestaFp).
    public static func secondaryFrToPrimaryFq(_ x: PallasFp) -> VestaFp {
        pallasFpToVestaFr(x)
    }

    // MARK: - Coordinate Extraction

    /// Extract Pallas point coordinates as Vesta-native field elements.
    /// Pallas Fp = Vesta Fr, so coordinates are native with zero overhead.
    public static func pallasPointToVestaNative(_ p: PallasPointProjective) -> (x: VestaFp, y: VestaFp) {
        pallasPointToVestaCoords(p)
    }

    /// Extract Vesta point coordinates as Pallas-native field elements.
    /// Vesta Fp = Pallas Fr, so coordinates are native with zero overhead.
    public static func vestaPointToPallasNative(_ p: VestaPointProjective) -> (x: PallasFp, y: PallasFp) {
        vestaPointToPallasCoords(p)
    }
}

// MARK: - Cross-Curve Verification Helper

/// Helper for verifying curve operations on one curve using arithmetic native
/// to the other curve in the cycle.
///
/// The key insight: given a cycle (E1, E2), we can verify E1 curve operations
/// inside an E2 circuit because E1's coordinates live in E2's scalar field.
/// This avoids the ~100x overhead of non-native field emulation.
///
/// Usage pattern (BN254/Grumpkin cycle):
///   1. Prover computes Grumpkin EC operations natively
///   2. Prover provides the results as witness to a BN254 circuit
///   3. BN254 circuit verifies Grumpkin operations using native Fr arithmetic
///      (Grumpkin coords are in BN254 Fr)
///   4. The BN254 proof is verified via pairing (cheap on Ethereum)
///
/// Usage pattern (Pallas/Vesta cycle):
///   1. Prover creates a Pallas IPA proof
///   2. Builds a Vesta circuit that verifies the Pallas proof
///   3. Creates a Vesta IPA proof of the verifier circuit
///   4. Repeat (alternating curves) for IVC
public class CrossCurveVerifier {

    // MARK: - Grumpkin Point Verification in BN254 Circuit

    /// Build R1CS constraints that verify a Grumpkin point addition
    /// inside a BN254 circuit. Since Grumpkin coordinates are BN254 Fr,
    /// all arithmetic is native -- no non-native field emulation needed.
    ///
    /// Constrains: P3 = P1 + P2 on Grumpkin (y^2 = x^3 - 17)
    ///
    /// The constraint uses the standard short Weierstrass addition formula:
    ///   lambda = (y2 - y1) / (x2 - x1)
    ///   x3 = lambda^2 - x1 - x2
    ///   y3 = lambda * (x1 - x3) - y1
    ///
    /// Returns R1CS entries for the addition constraints.
    /// Variable layout: [1, x1, y1, x2, y2, x3, y3, lambda, ...]
    public static func constrainGrumpkinAdd(
        builder: PlonkCircuitBuilder,
        x1: Int, y1: Int,
        x2: Int, y2: Int,
        lambdaVar: Int
    ) -> (x3: Int, y3: Int) {
        // Constraint 1: lambda * (x2 - x1) = y2 - y1
        let lx2 = builder.mul(lambdaVar, x2)
        let lx1 = builder.mul(lambdaVar, x1)
        let lhs = builder.add(lx2, y1)
        let rhs = builder.add(lx1, y2)
        builder.assertEqual(lhs, rhs)

        // Constraint 2: x3 = lambda^2 - x1 - x2
        let lambdaSq = builder.mul(lambdaVar, lambdaVar)
        let x1x2 = builder.add(x1, x2)
        let x3 = builder.addInput()
        let x3Sum = builder.add(x3, x1x2)
        builder.assertEqual(x3Sum, lambdaSq)

        // Constraint 3: y3 = lambda * (x1 - x3) - y1
        let y3 = builder.addInput()
        let y3y1 = builder.add(y3, y1)
        let lx3 = builder.mul(lambdaVar, x3)
        let lx1y = builder.mul(lambdaVar, x1)
        let checkLhs = builder.add(y3y1, lx3)
        builder.assertEqual(checkLhs, lx1y)

        return (x3, y3)
    }

    /// Build constraints that verify a Grumpkin point doubling in a BN254 circuit.
    ///
    /// Constrains: P3 = 2 * P1 on Grumpkin (y^2 = x^3 - 17, a=0)
    ///   lambda = 3 * x1^2 / (2 * y1)
    ///   x3 = lambda^2 - 2*x1
    ///   y3 = lambda * (x1 - x3) - y1
    public static func constrainGrumpkinDouble(
        builder: PlonkCircuitBuilder,
        x1: Int, y1: Int,
        lambdaVar: Int
    ) -> (x3: Int, y3: Int) {
        // Constraint 1: lambda * 2*y1 = 3*x1^2
        let x1Sq = builder.mul(x1, x1)
        let x1Sq2 = builder.add(x1Sq, x1Sq)
        let x1Sq3 = builder.add(x1Sq2, x1Sq)
        let y1_2 = builder.add(y1, y1)
        let lambdaTimesY = builder.mul(lambdaVar, y1_2)
        builder.assertEqual(lambdaTimesY, x1Sq3)

        // Constraint 2: x3 = lambda^2 - 2*x1
        let lambdaSq = builder.mul(lambdaVar, lambdaVar)
        let x1Double = builder.add(x1, x1)
        let x3 = builder.addInput()
        let x3Sum = builder.add(x3, x1Double)
        builder.assertEqual(x3Sum, lambdaSq)

        // Constraint 3: y3 = lambda * (x1 - x3) - y1
        let y3 = builder.addInput()
        let y3y1 = builder.add(y3, y1)
        let lx3 = builder.mul(lambdaVar, x3)
        let lx1 = builder.mul(lambdaVar, x1)
        let checkLhs = builder.add(y3y1, lx3)
        builder.assertEqual(checkLhs, lx1)

        return (x3, y3)
    }

    // MARK: - On-Curve Check

    /// Constrain that (x, y) lies on the Grumpkin curve: y^2 = x^3 - 17.
    /// All arithmetic is native BN254 Fr (Grumpkin base field).
    public static func constrainOnGrumpkinCurve(
        builder: PlonkCircuitBuilder,
        x: Int, y: Int
    ) {
        // y^2
        let ySq = builder.mul(y, y)
        // x^3
        let xSq = builder.mul(x, x)
        let xCube = builder.mul(xSq, x)
        // x^3 - 17: use addConstant with -17
        let neg17 = frSub(Fr.zero, frFromInt(17))
        let rhs = builder.addConstant(xCube, neg17)
        builder.assertEqual(ySq, rhs)
    }

    /// Constrain that (x, y) lies on the Pallas curve: y^2 = x^3 + 5.
    /// In a Vesta circuit, Pallas coordinates are native VestaFp elements.
    /// (This is used by the existing CycleFoldEngine.)
    public static func constrainOnPallasCurve(
        builder: PlonkCircuitBuilder,
        x: Int, y: Int
    ) {
        let ySq = builder.mul(y, y)
        let xSq = builder.mul(x, x)
        let xCube = builder.mul(xSq, x)
        let five = frFromInt(5)
        let rhs = builder.addConstant(xCube, five)
        builder.assertEqual(ySq, rhs)
    }

    // MARK: - Cycle Diagnostics

    /// Estimate constraint count for verifying an EC operation in the partner curve.
    /// Point addition: ~6 mul + ~6 add + 3 equality = ~15 gates
    /// Point doubling: ~7 mul + ~6 add + 3 equality = ~16 gates
    /// On-curve check: ~3 mul + 1 add + 1 equality = ~5 gates
    public static func estimateNativeCycleGates(operation: CycleCurveOp) -> Int {
        switch operation {
        case .pointAdd: return 15
        case .pointDouble: return 16
        case .onCurveCheck: return 5
        case .scalarMul(let bits): return bits * 16  // double-and-add per bit
        }
    }

    /// Estimate constraint count for the same operation WITHOUT cycle benefits
    /// (non-native field emulation, ~4 limbs, each mul costs ~4^2 = 16 native muls).
    public static func estimateNonNativeGates(operation: CycleCurveOp) -> Int {
        let nativeGates = estimateNativeCycleGates(operation: operation)
        // Non-native overhead: ~16x for multiplication, ~4x for addition
        return nativeGates * 12  // approximate average overhead
    }
}

/// Operations that can be verified across a curve cycle.
public enum CycleCurveOp {
    case pointAdd
    case pointDouble
    case onCurveCheck
    case scalarMul(bits: Int)
}

// MARK: - Cycle Selection

/// Select the appropriate curve cycle for a given use case.
public enum CycleSelection {
    /// BN254/Grumpkin: best for Ethereum verification (BN254 precompile)
    case bn254Grumpkin
    /// Pallas/Vesta: best for recursive IPA proofs (no trusted setup)
    case pallasVesta

    /// Human-readable name
    public var name: String {
        switch self {
        case .bn254Grumpkin: return BN254GrumpkinCycle.name
        case .pallasVesta: return PallasVestaCycle.name
        }
    }

    /// Whether pairing-based verification is available on the primary curve.
    /// BN254 has pairings; Pallas does not.
    public var hasPairing: Bool {
        switch self {
        case .bn254Grumpkin: return true
        case .pallasVesta: return false
        }
    }

    /// Scalar field size in bits for the primary curve.
    public var scalarFieldBits: Int {
        switch self {
        case .bn254Grumpkin: return 254
        case .pallasVesta: return 255
        }
    }
}
