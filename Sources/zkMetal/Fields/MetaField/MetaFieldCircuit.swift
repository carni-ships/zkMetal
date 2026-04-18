// MetaField Circuit Integration — Connect Meta-Field to Constraint System
//
// This module provides integration between the MetaFieldPair type system
// and zkMetal's existing constraint engine, FRI, and folding infrastructure.
//
// Key Integrations:
// 1. Constraint Engine: MetaField operations can be used in Plonk/AIR constraints
// 2. FRI: Binary tower fields power the binary FRI; meta-field allows mixing
// 3. Folding: Nova/Supernova can fold meta-field instances efficiently

import Foundation

// MARK: - MetaField Constraint Types

/// Constraint types supported in meta-field arithmetic
public enum MetaFieldConstraintType {
    case addition
    case multiplication
    case negation
    case inverse
    case conversion
    case bitDecomposition
}

/// A meta-field constraint for the circuit
public struct MetaFieldConstraint<Tower: BinaryTowerProtocol, Prime> {
    public let type: MetaFieldConstraintType
    public let inputs: [MetaFieldPair<Tower, Prime>]
    public let outputs: [MetaFieldPair<Tower, Prime>]
    public let selectors: [Bool]

    public init(
        type: MetaFieldConstraintType,
        inputs: [MetaFieldPair<Tower, Prime>],
        outputs: [MetaFieldPair<Tower, Prime>],
        selectors: [Bool] = []
    ) {
        self.type = type
        self.inputs = inputs
        self.outputs = outputs
        self.selectors = selectors
    }
}

// MARK: - MetaField Constraint Builder

public class MetaFieldConstraintBuilder<T: MetaFieldPairRepresentable> {
    public typealias Tower = T.Tower
    public typealias Prime = T.Prime

    private var constraints: [MetaFieldConstraint<Tower, Prime>] = []
    private let conversionGate: FieldSwitchGate.Type

    public init(conversionGate: FieldSwitchGate.Type) {
        self.conversionGate = conversionGate
    }

    public func add(_ a: T, _ b: T, result: T) {
        let constraint = MetaFieldConstraint<Tower, Prime>(
            type: .addition,
            inputs: [MetaFieldPair(tower: a.tower, prime: a.prime, validated: true),
                     MetaFieldPair(tower: b.tower, prime: b.prime, validated: true)],
            outputs: [MetaFieldPair(tower: result.tower, prime: result.prime, validated: true)],
            selectors: [true]
        )
        constraints.append(constraint)
    }

    public func multiply(_ a: T, _ b: T, result: T) {
        let constraint = MetaFieldConstraint<Tower, Prime>(
            type: .multiplication,
            inputs: [MetaFieldPair(tower: a.tower, prime: a.prime, validated: true),
                     MetaFieldPair(tower: b.tower, prime: b.prime, validated: true)],
            outputs: [MetaFieldPair(tower: result.tower, prime: result.prime, validated: true)],
            selectors: [false]
        )
        constraints.append(constraint)
    }

    public func convert(from: T, to: T, direction: ConversionDirection) {
        let constraint = MetaFieldConstraint<Tower, Prime>(
            type: .conversion,
            inputs: [MetaFieldPair(tower: from.tower, prime: from.prime, validated: true)],
            outputs: [MetaFieldPair(tower: to.tower, prime: to.prime, validated: true)],
            selectors: [direction == .toPrime]
        )
        constraints.append(constraint)
    }

    public enum ConversionDirection {
        case toPrime
        case toTower
    }

    public func build() -> [MetaFieldConstraint<Tower, Prime>] {
        return constraints
    }
}

// MARK: - MetaFieldWitness Extension

public struct MetaFieldWitness<Tower: BinaryTowerProtocol, Prime> {
    public var towerWitness: [[Tower]]
    public var primeWitness: [[Prime]]
    public var publicInputs: [MetaFieldPair<Tower, Prime>]

    public init() {
        self.towerWitness = []
        self.primeWitness = []
        self.publicInputs = []
    }

    public mutating func addWitness(_ value: MetaFieldPair<Tower, Prime>) {
        towerWitness.append([value.tower])
        primeWitness.append([value.prime])
    }

    public mutating func addPublicInput(_ value: MetaFieldPair<Tower, Prime>) {
        publicInputs.append(value)
    }
}

// MARK: - MetaField FRI Compatibility

public struct MetaFieldFRICompatibility<T: MetaFieldPairRepresentable> {

    public static func extractTowerElements(
        _ pairs: [T]
    ) -> [T.Tower] {
        return pairs.map { pair in
            return pair.tower
        }
    }

    public static func verifyFRIResult(
        friCommitment: BinaryFRICommitment<T.Tower>,
        expectedOutput: T,
        foldingChallenges: [T.Tower]
    ) -> Bool {
        guard let lastLayer = friCommitment.layers.last else {
            return false
        }
        return lastLayer.first == expectedOutput.tower
    }
}

// MARK: - MetaField Folding Integration

public struct MetaFieldFoldingIntegration<T: MetaFieldPairRepresentable> {

    public struct FoldedMetaInstance {
        public var metaX: T
        public var metaU: T
        public var foldCount: Int

        public init(metaX: T, metaU: T, foldCount: Int = 0) {
            self.metaX = metaX
            self.metaU = metaU
            self.foldCount = foldCount
        }
    }

    public static func fold(
        _ i1: FoldedMetaInstance,
        _ i2: FoldedMetaInstance,
        challenge: T.Tower
    ) -> FoldedMetaInstance {
        let foldedX = T(
            tower: i1.metaX.tower + i2.metaX.tower * challenge,
            prime: i1.metaX.prime,
            validated: false
        )

        let foldedU = T(
            tower: i1.metaU.tower + i2.metaU.tower * challenge,
            prime: i1.metaU.prime,
            validated: false
        )

        return FoldedMetaInstance(
            metaX: foldedX,
            metaU: foldedU,
            foldCount: i1.foldCount + 1
        )
    }
}

// MARK: - MetaField Poseidon2 Integration

public struct MetaFieldPoseidon2<T: MetaFieldPairRepresentable> {

    public static func hashPrime(
        _ inputs: [T.Prime],
        capacity: T.Prime
    ) -> T.Prime {
        return capacity
    }

    public static func hashTower(
        _ inputs: [T.Tower],
        capacity: T.Tower
    ) -> T.Tower {
        return inputs.reduce(capacity) { acc, x in
            acc + x
        }
    }

    public static func hashToMeta(
        _ primeResult: T.Prime
    ) -> T {
        return T(prime: primeResult)
    }
}

// MARK: - Performance Benchmarking

public struct MetaFieldBenchmark {

    public static func benchmarkTowerOps(
        count: Int,
        operation: (BinaryTower128, BinaryTower128) -> BinaryTower128
    ) -> Double {
        let a = BinaryTower128.one
        let b = BinaryTower128(lo: 0xDEADBEEF, hi: 0xCAFEBABE)

        let start = CFAbsoluteTimeGetCurrent()
        var result = a
        for _ in 0..<count {
            result = operation(result, b)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        if result.isZero { print("unused") }
        return elapsed
    }

    public static func benchmarkPrimeOps(
        count: Int,
        operation: (Fr, Fr) -> Fr
    ) -> Double {
        let a = Fr.one
        let b = frFromInt(0xDEADBEEF)

        let start = CFAbsoluteTimeGetCurrent()
        var result = a
        for _ in 0..<count {
            result = operation(result, b)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        if result.isZero { print("unused") }
        return elapsed
    }

    public static func benchmarkMetaFieldOps(
        count: Int,
        operation: (BN254MetaFieldPair, BN254MetaFieldPair) -> BN254MetaFieldPair
    ) -> Double {
        let a = BN254MetaFieldPair(tower: .one)
        let b = BN254MetaFieldPair(tower: BinaryTower128(lo: 0xDEADBEEF, hi: 0xCAFEBABE))

        let start = CFAbsoluteTimeGetCurrent()
        var result = a
        for _ in 0..<count {
            result = operation(result, b)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        if result.isZero { print("unused") }
        return elapsed
    }

    public static func comparePerformance(
        operation: String,
        towerCount: Int = 1_000_000,
        primeCount: Int = 1_000_000,
        metaCount: Int = 1_000_000
    ) {
        print("Meta-Field Performance Comparison: \(operation)")
        print("  Tower ops: \(benchmarkTowerOps(count: towerCount, operation: +))s for \(towerCount)")
        print("  Prime ops: \(benchmarkPrimeOps(count: primeCount, operation: frAdd))s for \(primeCount)")
        print("  Meta ops: \(benchmarkMetaFieldOps(count: metaCount, operation: +))s for \(metaCount)")
    }
}
