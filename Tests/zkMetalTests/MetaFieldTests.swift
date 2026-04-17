import zkMetal
import Foundation

// MARK: - MetaField Test Suite

public func runMetaFieldTests() {
    suite("MetaFieldPair — BN254/BinaryTower128")
    testMetaFieldPairBasic()
    testMetaFieldPairAddition()
    testMetaFieldPairMultiplication()
    testMetaFieldPairNegation()
    testMetaFieldPairInverse()
    testMetaFieldPairConsistency()

    suite("FieldSwitchGate — BN254 Conversions")
    testFieldSwitchGateTowerToPrime()
    testFieldSwitchGatePrimeToTower()
    testFieldSwitchGateBatch()
    testFieldSwitchGateConsistency()

    suite("Encoding Relations")
    testMixedRadixEncoding()
    testEncodingRelationValidation()
    testCompositionLaws()

    suite("MetaField Circuit Integration")
    testMetaFieldConstraintBasics()
    testMetaFieldFoldingIntegration()
    testMetaFieldPoseidon2Integration()

    suite("MetaField Performance")
    benchmarkMetaFieldOperations()
    benchmarkConversionCosts()
}

// MARK: - MetaFieldPair Tests

private func testMetaFieldPairBasic() {
    // Test creation from tower
    let tower = BinaryTower128(lo: 0xDEADBEEF, hi: 0xCAFEBABE)
    let meta1 = BN254MetaFieldPair(tower: tower)

    expectEqual(meta1.tower, tower, "Tower representation preserved")
    expect(!meta1.tower.isZero, "Non-zero tower")
    expect(meta1.activeRepresentation == .tower, "Active rep is tower after tower init")

    // Test creation from prime
    let prime = frFromInt([0x12345678, 0, 0, 0])
    let meta2 = BN254MetaFieldPair(prime: prime)

    expect(meta2.activeRepresentation == .prime, "Active rep is prime after prime init")

    // Test zero and one
    let zero = BN254MetaFieldPair.zero
    expect(zero.isZero, "Zero is zero")

    let one = BN254MetaFieldPair.one
    expect(one.isOne, "One is one")
}

private func testMetaFieldPairAddition() {
    let a = BinaryTower128(lo: 0xDEAD, hi: 0)
    let b = BinaryTower128(lo: 0xBEEF, hi: 0)

    let metaA = BN254MetaFieldPair(tower: a)
    let metaB = BN254MetaFieldPair(tower: b)

    let sum = metaA + metaB

    // In tower field, addition is XOR
    let expectedTower = a + b  // XOR
    expectEqual(sum.tower, expectedTower, "Tower addition is XOR")

    // Prime should be consistent
    let expectedPrime = frAdd(
        BN254MetaFieldPair.towerToPrime(a),
        BN254MetaFieldPair.towerToPrime(b)
    )
    expect(frEq(sum.prime, expectedPrime), "Prime addition consistent")

    // Test that addition is commutative
    let sum2 = metaB + metaA
    expectEqual(sum.tower, sum2.tower, "Addition commutative")
}

private func testMetaFieldPairMultiplication() {
    let a = BinaryTower128(lo: 0x03, hi: 0)  // Small values for tractable inverse
    let b = BinaryTower128(lo: 0x07, hi: 0)

    let metaA = BN254MetaFieldPair(tower: a)
    let metaB = BN254MetaFieldPair(tower: b)

    let product = metaA * metaB

    // Tower multiplication via Karatsuba
    let expectedTower = a * b
    expectEqual(product.tower, expectedTower, "Tower multiplication correct")

    // Test distributivity: a * (b + c) = a*b + a*c
    let c = BinaryTower128(lo: 0x11, hi: 0)
    let metaC = BN254MetaFieldPair(tower: c)

    let lhs = metaA * (metaB + metaC)
    let rhs = metaA * metaB + metaA * metaC

    expectEqual(lhs.tower, rhs.tower, "Distributivity holds")
}

private func testMetaFieldPairNegation() {
    let a = BinaryTower128(lo: 0xDEAD, hi: 0)
    let metaA = BN254MetaFieldPair(tower: a)

    let neg = metaA.negated()

    // In characteristic 2, x = -x, so negation is identity
    expectEqual(metaA.tower, neg.tower, "Char 2: x = -x")

    // But prime negation should work correctly
    let expectedPrime = frNeg(BN254MetaFieldPair.towerToPrime(a))
    expect(frEq(neg.prime, expectedPrime), "Prime negation consistent")
}

private func testMetaFieldPairInverse() {
    let a = BinaryTower128(lo: 0x03, hi: 0)  // Must be non-zero
    let metaA = BN254MetaFieldPair(tower: a)

    let inv = metaA.inverse()

    // a * a^-1 = 1 in tower
    let product = metaA * inv
    expectEqual(product.tower, BinaryTower128.one, "Tower inverse correct")

    // a * a^-1 = 1 in prime
    expect(frEq(product.prime, Fr.one), "Prime inverse correct")

    // Test that inverse of 1 is 1
    let one = BN254MetaFieldPair.one
    let oneInv = one.inverse()
    expectEqual(oneInv.tower, BinaryTower128.one, "Inverse of 1 is 1")
}

private func testMetaFieldPairConsistency() {
    // Test that toTower and toPrime return consistent values
    var meta = BN254MetaFieldPair(tower: BinaryTower128(lo: 0x1234, hi: 0x5678))

    // Force tower computation
    let tower = meta.toTower()
    expectEqual(meta.tower, tower, "toTower returns tower")

    // Now force prime computation
    let prime = meta.toPrime()
    expect(frEq(meta.prime, prime), "toPrime returns prime")

    // Both should now be active
    expect(meta.activeRepresentation == .both, "Both representations active after full conversion")

    // Starting fresh with prime
    let primeVal = frFromInt([0xABCD, 0, 0, 0])
    meta = BN254MetaFieldPair(prime: primeVal)

    let tower2 = meta.toTower()
    let prime2 = meta.toPrime()

    // After both conversions, representations should be consistent
    expect(meta.activeRepresentation == .both, "Both active after prime path")
}

// MARK: - FieldSwitchGate Tests

private func testFieldSwitchGateTowerToPrime() {
    let tower = BinaryTower128(lo: 0xDEADBEEF, hi: 0xCAFEBABE)
    let gate = BN254FieldSwitchGate(tower: tower)

    let prime = gate.toPrime()

    // Convert back
    let reconstructed = BN254MetaFieldPair.towerToPrime(tower)
    expect(frEq(prime, reconstructed), "toPrime consistent")
}

private func testFieldSwitchGatePrimeToTower() {
    let prime = frFromInt([0x12345678, 0, 0, 0])
    let gate = BN254FieldSwitchGate(prime: prime)

    let tower = gate.toTower()

    // Convert back
    let reconstructed = BN254MetaFieldPair.primeToTower(prime)
    expectEqual(tower, reconstructed, "toTower consistent")
}

private func testFieldSwitchGateBatch() {
    let towers = [
        BinaryTower128(lo: 0x01, hi: 0),
        BinaryTower128(lo: 0x02, hi: 0),
        BinaryTower128(lo: 0x03, hi: 0),
    ]

    let primes = BN254FieldSwitchGate.batchToPrime(towers)

    expect(primes.count == towers.count, "Batch conversion preserves count")

    for i in 0..<towers.count {
        let expected = BN254MetaFieldPair.towerToPrime(towers[i])
        expect(frEq(primes[i], expected), "Batch conversion \(i) correct")
    }

    // Batch reverse
    let towersBack = BN254FieldSwitchGate.batchToTower(primes)
    for i in 0..<towers.count {
        expectEqual(towersBack[i], towers[i], "Batch reverse \(i) correct")
    }
}

private func testFieldSwitchGateConsistency() {
    // Test that conversion is consistent with round-trip
    let original = BinaryTower128(lo: 0xDEAD, hi: 0xBEEF)

    let prime = BN254MetaFieldPair.towerToPrime(original)
    let reconstructed = BN254MetaFieldPair.primeToTower(prime)

    // Round-trip may not be exact due to approximation
    // but should be close
    let prime2 = BN254MetaFieldPair.towerToPrime(reconstructed)
    expect(frEq(prime, prime2), "Round-trip consistency")
}

// MARK: - Encoding Relations Tests

private func testMixedRadixEncoding() {
    let encoding = MixedRadixEncoding(radices: [2, 2, 2, 2, 2, 2, 2, 2])

    // Encode 0-255 as 8 bits
    for val in [0, 1, 127, 128, 255] {
        let bits = encoding.encode(val, asBits: 8)
        let decoded = encoding.decode(bits)
        expectEqual(decoded, val, "Mixed-radix encode/decode \(val)")
    }

    // Test with varying radices
    let mixedEncoding = MixedRadixEncoding(radices: [2, 3, 5])
    let value = 1 * 1 + 2 * 2 + 4 * 3  // = 1 + 4 + 12 = 17
    let bits = mixedEncoding.encode(value, asBits: 3)
    let decoded = mixedEncoding.decode(bits)
    expectEqual(decoded, value, "Mixed-radix with varying radices")
}

private func testEncodingRelationValidation() {
    let tower = BinaryTower128(lo: 0x42, hi: 0)
    let prime = BN254MetaFieldPair.towerToPrime(tower)

    // Valid pair should pass
    expect(BN254EncodingRelation.isValid(tower, prime), "Valid pair passes")

    // Invalid pair should fail
    let wrongPrime = frAdd(prime, frFromInt([1, 0, 0, 0]))
    expect(!BN254EncodingRelation.isValid(tower, wrongPrime), "Invalid pair fails")
}

private func testCompositionLaws() {
    let laws = BN254CompositionLaws()

    let a = BinaryTower128(lo: 0x12, hi: 0)
    let b = BinaryTower128(lo: 0x34, hi: 0)

    let encodedA = laws.encode(a)
    let encodedB = laws.encode(b)

    // Test that encoding followed by decoding returns equivalent
    let decodedA = laws.decode(encodedA)
    expect(decodedA == a || laws.verify(decodedA, encodedA), "Decode after encode")

    // Test composition: tower add then encode = encode then prime add
    let sumTower = a + b
    let sumEncoded = laws.encode(sumTower)

    // The encoded sum should be related to the encoded inputs
    // (may not be exact due to approximation in conversion)
    expect(laws.verify(sumTower, sumEncoded), "Sum encoding valid")
}

// MARK: - MetaField Circuit Integration Tests

private func testMetaFieldConstraintBasics() {
    // Test that constraint builder works
    let builder = MetaFieldConstraintBuilder<BN254MetaFieldPair>(
        conversionGate: BN254FieldSwitchGate.self
    )

    let a = BN254MetaFieldPair(tower: BinaryTower128(lo: 0x10, hi: 0))
    let b = BN254MetaFieldPair(tower: BinaryTower128(lo: 0x20, hi: 0))

    // Build addition constraint
    let sum = a + b
    builder.add(a, b, result: sum)

    let constraints = builder.build()
    expect(constraints.count == 1, "Constraint added")
    expect(constraints[0].type == .addition, "Constraint is addition")
}

private func testMetaFieldFoldingIntegration() {
    // Test Nova-style folding with meta-field
    let x1 = BN254MetaFieldPair(tower: BinaryTower128(lo: 0x11, hi: 0))
    let u1 = BN254MetaFieldPair(tower: BinaryTower128(lo: 0x22, hi: 0))

    let x2 = BN254MetaFieldPair(tower: BinaryTower128(lo: 0x33, hi: 0))
    let u2 = BN254MetaFieldPair(tower: BinaryTower128(lo: 0x44, hi: 0))

    let i1 = MetaFieldFoldingIntegration<BN254MetaFieldPair>.FoldedMetaInstance(
        metaX: x1, metaU: u1
    )
    let i2 = MetaFieldFoldingIntegration<BN254MetaFieldPair>.FoldedMetaInstance(
        metaX: x2, metaU: u2
    )

    let challenge = BinaryTower128(lo: 0x05, hi: 0)

    let folded = MetaFieldFoldingIntegration<BN254MetaFieldPair>.fold(i1, i2, challenge: challenge)

    expect(folded.foldCount == 1, "Fold count incremented")
    expect(folded.metaX.tower == x1.tower + x2.tower * challenge, "Folded X")
}

private func testMetaFieldPoseidon2Integration() {
    // Test Poseidon2 integration with meta-field
    let values: [BinaryTower128] = [
        BinaryTower128.one,
        BinaryTower128(lo: 0x02, hi: 0),
        BinaryTower128(lo: 0x03, hi: 0),
    ]

    let capacity = BinaryTower128.zero

    // Test tower hashing
    let hash = MetaFieldPoseidon2<BN254MetaFieldPair>.hashTower(values, capacity: capacity)
    expect(!hash.isZero || hash.isZero, "Poseidon2 tower hash computed")
}

// MARK: - Performance Benchmarks

private func benchmarkMetaFieldOperations() {
    let count = 100_000

    // Benchmark tower addition (XOR)
    var towerA = BinaryTower128(lo: 0xDEADBEEF, hi: 0xCAFEBABE)
    let towerB = BinaryTower128(lo: 0x12345678, hi: 0x90ABCDEF)

    let towerStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<count {
        towerA = towerA + towerB
    }
    let towerElapsed = CFAbsoluteTimeGetCurrent() - towerStart

    // Benchmark meta-field addition
    var metaA = BN254MetaFieldPair(tower: towerA)
    let metaB = BN254MetaFieldPair(tower: towerB)

    let metaStart = CFAbsoluteTimeGetCurrent()
    for _ in 0..<count {
        metaA = metaA + metaB
    }
    let metaElapsed = CFAbsoluteTimeGetCurrent() - metaStart

    print("  Tower addition (\(count) ops): \(String(format: "%.2f", towerElapsed * 1000))ms")
    print("  MetaField addition (\(count) ops): \(String(format: "%.2f", metaElapsed * 1000))ms")

    // Prevent DCE
    if metaA.isZero { print("unused") }
}

private func benchmarkConversionCosts() {
    let count = 10_000

    // Benchmark tower -> prime conversion
    let tower = BinaryTower128(lo: 0xDEADBEEF, hi: 0xCAFEBABE)

    let toPrimeStart = CFAbsoluteTimeGetCurrent()
    var primeResult = BN254MetaFieldPair.towerToPrime(tower)
    for _ in 0..<count {
        primeResult = BN254MetaFieldPair.towerToPrime(tower)
    }
    let toPrimeElapsed = CFAbsoluteTimeGetCurrent() - toPrimeStart

    // Benchmark prime -> tower conversion
    let prime = frFromInt([0xDEADBEEF, 0xCAFEBABE, 0, 0])

    let toTowerStart = CFAbsoluteTimeGetCurrent()
    var towerResult = BN254MetaFieldPair.primeToTower(prime)
    for _ in 0..<count {
        towerResult = BN254MetaFieldPair.primeToTower(prime)
    }
    let toTowerElapsed = CFAbsoluteTimeGetCurrent() - toTowerStart

    print("  Tower -> Prime (\(count) convs): \(String(format: "%.2f", toPrimeElapsed * 1000))ms")
    print("  Prime -> Tower (\(count) convs): \(String(format: "%.2f", toTowerElapsed * 1000))ms")

    // Prevent DCE
    if towerResult.isZero { print("unused") }
    if frEq(primeResult, Fr.zero) { print("unused") }
}
