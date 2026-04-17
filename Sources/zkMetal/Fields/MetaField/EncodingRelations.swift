// Encoding Relations for Tower-Prime Meta-Fields
//
// This module defines the mathematical encoding relations that enable transparent
// composition of tower-native and prime-native operations within the meta-field
// framework.

import Foundation

// MARK: - Encoding Relation Types

/// Represents an encoding relation between tower and prime representations.
public struct EncodingRelation<Tower: BinaryTowerProtocol, Prime> {
    /// Check if tower and prime representations are equivalent
    public let isValid: (Tower, Prime) -> Bool

    /// The bit-length of the tower representation
    public let towerBits: Int

    /// The bit-length of the prime representation
    public let primeBits: Int

    public init(
        towerBits: Int,
        primeBits: Int,
        isValid: @escaping (Tower, Prime) -> Bool
    ) {
        self.towerBits = towerBits
        self.primeBits = primeBits
        self.isValid = isValid
    }
}

// MARK: - BN254 Encoding Relation

public struct BN254EncodingRelation {
    public typealias Tower = BinaryTower128
    public typealias Prime = Fr

    public static let towerBits = 128
    public static let primeBits = 254

    public static func isValid(_ tower: Tower, _ prime: Prime) -> Bool {
        let reconstructed = BN254MetaFieldPair.towerToPrime(tower)
        return frEq(reconstructed, prime)
    }

    public static let relation = EncodingRelation<Tower, Prime>(
        towerBits: towerBits,
        primeBits: primeBits,
        isValid: isValid
    )
}

// MARK: - Mixed-Radix Encoding

public struct MixedRadixEncoding {
    public let maxValue: Int
    public let radices: [Int]

    public init(radices: [Int]) {
        self.radices = radices
        var max = 1
        for r in radices {
            max *= r
        }
        self.maxValue = max
    }

    public func encode(_ value: Int, asBits bitsPerLimb: Int) -> [UInt8] {
        precondition(value < maxValue, "Value \(value) exceeds max \(maxValue)")

        var remaining = value
        var bits = [UInt8]()
        for radix in radices {
            bits.append(UInt8(remaining % radix))
            remaining /= radix
        }
        while bits.count < bitsPerLimb {
            bits.append(0)
        }
        return bits
    }

    public func decode(_ bits: [UInt8]) -> Int {
        var result = 0
        var multiplier = 1
        for (i, bit) in bits.enumerated() {
            precondition(i < radices.count, "Too many bits")
            precondition(bit < UInt8(radices[i]), "Invalid digit for position \(i)")
            result += Int(bit) * multiplier
            multiplier *= radices[i]
        }
        return result
    }
}

// MARK: - BN254 Composition Laws

public struct BN254CompositionLaws {
    public let encoding = BN254EncodingRelation.relation

    public init() {}

    public func encode(_ tower: BinaryTower128) -> Fr {
        BN254MetaFieldPair.towerToPrime(tower)
    }

    public func decode(_ prime: Fr) -> BinaryTower128 {
        BN254MetaFieldPair.primeToTower(prime)
    }

    public func verify(_ tower: BinaryTower128, _ prime: Fr) -> Bool {
        BN254EncodingRelation.isValid(tower, prime)
    }
}

// MARK: - Transparency Proof System

public struct TransparencyProof<Tower: BinaryTowerProtocol, Prime> {

    public struct RepresentationProof {
        public let towerCommitment: Tower
        public let primeCommitment: Prime
        public let equivalenceProof: [UInt8]

        public init(towerCommitment: Tower, primeCommitment: Prime, equivalenceProof: [UInt8]) {
            self.towerCommitment = towerCommitment
            self.primeCommitment = primeCommitment
            self.equivalenceProof = equivalenceProof
        }
    }

    public struct OperationProof {
        public let inputProofs: [RepresentationProof]
        public let outputProofs: [RepresentationProof]
        public let consistencyProof: [UInt8]
    }

    public static func proveConversion(
        _ tower: Tower,
        to prime: Prime,
        usingTranscript transcript: inout [UInt8]
    ) -> RepresentationProof {
        return RepresentationProof(
            towerCommitment: tower,
            primeCommitment: prime,
            equivalenceProof: []
        )
    }

    public static func verifyConversion(
        _ proof: RepresentationProof,
        relation: EncodingRelation<Tower, Prime>
    ) -> Bool {
        relation.isValid(proof.towerCommitment, proof.primeCommitment)
    }
}

// MARK: - Encoding Efficiency Analysis

public struct EncodingEfficiency {
    public let towerToPrimeCost: Double
    public let primeToTowerCost: Double
    public let compressionRatio: Double
    public let hasEfficientEncoding: Bool

    public static let bn254 = EncodingEfficiency(
        towerToPrimeCost: 1.0,
        primeToTowerCost: 1.0,
        compressionRatio: 128.0 / 254.0,
        hasEfficientEncoding: true
    )
}

// MARK: - Native Operation Selection

public struct OperationSelector<Tower: BinaryTowerProtocol, Prime> {

    public static func towerNativeOps() -> Set<String> {
        return [
            "xor", "and", "not", "shift", "bitwise",
            "hash", "merkle", "commitment"
        ]
    }

    public static func primeNativeOps() -> Set<String> {
        return [
            "add", "mul", "mod", "exp", "inverse",
            "curve_add", "pairing", "scalar_mul"
        ]
    }

    public static func neutralOps() -> Set<String> {
        return [
            "copy", "zero", "one", "compare", "select"
        ]
    }

    public static func select(
        operations: [(name: String, count: Int)]
    ) -> MetaFieldPair<Tower, Prime>.Representation {
        var towerScore = 0
        var primeScore = 0

        let towerOps = towerNativeOps()
        let primeOps = primeNativeOps()

        for (name, count) in operations {
            if towerOps.contains(name) {
                towerScore += count
            } else if primeOps.contains(name) {
                primeScore += count
            }
        }

        if towerScore > primeScore * 2 {
            return .tower
        } else if primeScore > towerScore * 2 {
            return .prime
        }
        return .both
    }
}
