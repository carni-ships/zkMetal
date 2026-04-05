// EVM Precompile Tests
//
// Test vectors from Ethereum consensus tests plus edge cases.

import Foundation
import zkMetal

// MARK: - Hex Helpers

private func hexToBytes(_ hex: String) -> [UInt8] {
    let h = hex.hasPrefix("0x") ? String(hex.dropFirst(2)) : hex
    var bytes = [UInt8]()
    var idx = h.startIndex
    while idx < h.endIndex {
        let next = h.index(idx, offsetBy: 2)
        let byteStr = String(h[idx..<next])
        bytes.append(UInt8(byteStr, radix: 16)!)
        idx = next
    }
    return bytes
}

private func bytesToHex(_ bytes: [UInt8]) -> String {
    return bytes.map { String(format: "%02x", $0) }.joined()
}

/// Pad or truncate to exactly n bytes (left-pad with zeros for big-endian)
private func padLeft(_ data: [UInt8], to n: Int) -> [UInt8] {
    if data.count >= n { return Array(data.suffix(n)) }
    return [UInt8](repeating: 0, count: n - data.count) + data
}

// MARK: - BN254 Test Vectors

// Generator G1 = (1, 2)
private let BN254_G1_X = padLeft(hexToBytes("01"), to: 32)
private let BN254_G1_Y = padLeft(hexToBytes("02"), to: 32)

// 2*G1 (known result)
private let BN254_2G1_X = hexToBytes("030644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd3")
private let BN254_2G1_Y = hexToBytes("15ed738c0e0a7c92e7845f96b2ae9c0a68a6a449e3538fc7ff3ebf7a5a18a2c4")

// Known scalar multiplication: 2 * G1
private let SCALAR_TWO = padLeft(hexToBytes("02"), to: 32)

// Negation of G1: (1, p - 2)
private let BN254_NEG_G1_Y = hexToBytes("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd45")

// MARK: - Test Implementation

public func runEVMPrecompileTests() {
    suite("EVM Precompile: BN254 ecAdd (0x06)")

    // Test 1: G1 + G1 = 2*G1
    do {
        let input = BN254_G1_X + BN254_G1_Y + BN254_G1_X + BN254_G1_Y
        let result = EVMPrecompile06_ecAdd(input: input)
        expect(result != nil, "ecAdd(G1, G1) should succeed")
        if let r = result {
            expectEqual(r.count, 64, "ecAdd output should be 64 bytes")
            let xHex = bytesToHex(Array(r[0..<32]))
            let yHex = bytesToHex(Array(r[32..<64]))
            expectEqual(xHex, bytesToHex(BN254_2G1_X), "ecAdd(G1,G1).x == 2G1.x")
            expectEqual(yHex, bytesToHex(BN254_2G1_Y), "ecAdd(G1,G1).y == 2G1.y")
        }
    }

    // Test 2: G1 + O = G1 (identity)
    do {
        let zero64 = [UInt8](repeating: 0, count: 64)
        let input = BN254_G1_X + BN254_G1_Y + zero64
        let result = EVMPrecompile06_ecAdd(input: input)
        expect(result != nil, "ecAdd(G1, O) should succeed")
        if let r = result {
            expectEqual(bytesToHex(Array(r[0..<32])), bytesToHex(BN254_G1_X), "ecAdd(G1,O).x == G1.x")
            expectEqual(bytesToHex(Array(r[32..<64])), bytesToHex(BN254_G1_Y), "ecAdd(G1,O).y == G1.y")
        }
    }

    // Test 3: O + G1 = G1
    do {
        let zero64 = [UInt8](repeating: 0, count: 64)
        let input = zero64 + BN254_G1_X + BN254_G1_Y
        let result = EVMPrecompile06_ecAdd(input: input)
        expect(result != nil, "ecAdd(O, G1) should succeed")
        if let r = result {
            expectEqual(bytesToHex(Array(r[0..<32])), bytesToHex(BN254_G1_X), "ecAdd(O,G1).x == G1.x")
        }
    }

    // Test 4: O + O = O
    do {
        let input = [UInt8](repeating: 0, count: 128)
        let result = EVMPrecompile06_ecAdd(input: input)
        expect(result != nil, "ecAdd(O, O) should succeed")
        if let r = result {
            expect(r.allSatisfy({ $0 == 0 }), "ecAdd(O,O) should be identity")
        }
    }

    // Test 5: G1 + (-G1) = O
    do {
        let input = BN254_G1_X + BN254_G1_Y + BN254_G1_X + BN254_NEG_G1_Y
        let result = EVMPrecompile06_ecAdd(input: input)
        expect(result != nil, "ecAdd(G1, -G1) should succeed")
        if let r = result {
            expect(r.allSatisfy({ $0 == 0 }), "ecAdd(G1, -G1) should be identity")
        }
    }

    // Test 6: Invalid point (not on curve)
    do {
        let badX = padLeft(hexToBytes("01"), to: 32)
        let badY = padLeft(hexToBytes("03"), to: 32) // (1, 3) is not on curve
        let input = badX + badY + BN254_G1_X + BN254_G1_Y
        let result = EVMPrecompile06_ecAdd(input: input)
        expect(result == nil, "ecAdd with invalid point should fail")
    }

    // Test 7: Short input (padded with zeros)
    do {
        let result = EVMPrecompile06_ecAdd(input: [])
        expect(result != nil, "ecAdd with empty input should succeed (all zeros = O+O)")
        if let r = result {
            expect(r.allSatisfy({ $0 == 0 }), "ecAdd([]) should be identity")
        }
    }

    suite("EVM Precompile: BN254 ecMul (0x07)")

    // Test 1: 2 * G1 = 2G1
    do {
        let input = BN254_G1_X + BN254_G1_Y + SCALAR_TWO
        let result = EVMPrecompile07_ecMul(input: input)
        expect(result != nil, "ecMul(G1, 2) should succeed")
        if let r = result {
            expectEqual(bytesToHex(Array(r[0..<32])), bytesToHex(BN254_2G1_X), "ecMul(G1,2).x == 2G1.x")
            expectEqual(bytesToHex(Array(r[32..<64])), bytesToHex(BN254_2G1_Y), "ecMul(G1,2).y == 2G1.y")
        }
    }

    // Test 2: 0 * G1 = O
    do {
        let input = BN254_G1_X + BN254_G1_Y + [UInt8](repeating: 0, count: 32)
        let result = EVMPrecompile07_ecMul(input: input)
        expect(result != nil, "ecMul(G1, 0) should succeed")
        if let r = result {
            expect(r.allSatisfy({ $0 == 0 }), "ecMul(G1, 0) should be identity")
        }
    }

    // Test 3: 1 * G1 = G1
    do {
        let scalarOne = padLeft(hexToBytes("01"), to: 32)
        let input = BN254_G1_X + BN254_G1_Y + scalarOne
        let result = EVMPrecompile07_ecMul(input: input)
        expect(result != nil, "ecMul(G1, 1) should succeed")
        if let r = result {
            expectEqual(bytesToHex(Array(r[0..<32])), bytesToHex(BN254_G1_X), "ecMul(G1,1).x == G1.x")
            expectEqual(bytesToHex(Array(r[32..<64])), bytesToHex(BN254_G1_Y), "ecMul(G1,1).y == G1.y")
        }
    }

    // Test 4: s * O = O
    do {
        let input = [UInt8](repeating: 0, count: 64) + SCALAR_TWO
        let result = EVMPrecompile07_ecMul(input: input)
        expect(result != nil, "ecMul(O, 2) should succeed")
        if let r = result {
            expect(r.allSatisfy({ $0 == 0 }), "ecMul(O, 2) should be identity")
        }
    }

    // Test 5: Invalid point
    do {
        let badX = padLeft(hexToBytes("01"), to: 32)
        let badY = padLeft(hexToBytes("03"), to: 32)
        let input = badX + badY + SCALAR_TWO
        let result = EVMPrecompile07_ecMul(input: input)
        expect(result == nil, "ecMul with invalid point should fail")
    }

    suite("EVM Precompile: BN254 ecPairing (0x08)")

    // Test 1: Empty input -> success (trivial pairing = 1)
    do {
        let result = EVMPrecompile08_ecPairing(input: [])
        expect(result != nil, "ecPairing([]) should succeed")
        if let r = result {
            expectEqual(r.count, 32, "ecPairing output should be 32 bytes")
            expectEqual(r[31], 1, "empty pairing should return 1")
        }
    }

    // Test 2: Invalid length
    do {
        let result = EVMPrecompile08_ecPairing(input: [UInt8](repeating: 0, count: 100))
        expect(result == nil, "ecPairing with invalid length should fail")
    }

    // Test 3: One pair with all zeros (G1=O, G2=O) -> pairing = 1
    do {
        let input = [UInt8](repeating: 0, count: 192)
        let result = EVMPrecompile08_ecPairing(input: input)
        expect(result != nil, "ecPairing with zero pair should succeed")
        if let r = result {
            expectEqual(r[31], 1, "pairing(O, O) = 1")
        }
    }

    // Test 4: e(G1, G2) * e(-G1, G2) = 1
    // This is the canonical pairing check test
    do {
        // BN254 G2 generator:
        // x = (10857046999023057135944570762232829481370756359578518086990519993285655852781,
        //      11559732032986387107991004021392285783925812861821192530917403151452391805634)
        // y = (8495653923123431417604973247489272438418190587263600148770280649306958101930,
        //      4082367875863433681332203403145435568316851327593401208105741076214120093531)
        // BN254 G2 generator (EVM format: x_im || x_re || y_im || y_re)
        let g2XI = hexToBytes("198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2")
        let g2XR = hexToBytes("1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed")
        let g2YI = hexToBytes("090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b")
        let g2YR = hexToBytes("12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa")

        // Pad to 32 bytes each
        let g2XIp = padLeft(g2XI, to: 32)
        let g2XRp = padLeft(g2XR, to: 32)
        let g2YIp = padLeft(g2YI, to: 32)
        let g2YRp = padLeft(g2YR, to: 32)

        // Pair 1: (G1, G2)
        let pair1 = BN254_G1_X + BN254_G1_Y + g2XIp + g2XRp + g2YIp + g2YRp
        // Pair 2: (-G1, G2)
        let pair2 = BN254_G1_X + BN254_NEG_G1_Y + g2XIp + g2XRp + g2YIp + g2YRp

        let input = pair1 + pair2
        let result = EVMPrecompile08_ecPairing(input: input)
        expect(result != nil, "ecPairing(G1,G2,-G1,G2) should succeed")
        if let r = result {
            expectEqual(r[31], 1, "e(G1,G2)*e(-G1,G2) should equal 1")
        }
    }

    suite("EVM Precompile: BLS12-381 G1 Add (0x0A)")

    // BLS12-381 G1 generator in EVM format (64-byte padded Fp coordinates)
    // G1 = (x, y) where:
    // x = 0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
    // y = 0x08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1

    let bls12G1X = padLeft(hexToBytes("17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"), to: 64)
    let bls12G1Y = padLeft(hexToBytes("08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1"), to: 64)

    // Test 1: G1 + O = G1
    do {
        let zero128 = [UInt8](repeating: 0, count: 128)
        let input = bls12G1X + bls12G1Y + zero128
        let result = EVMPrecompile0A_bls12381G1Add(input: input)
        expect(result != nil, "BLS G1Add(G1, O) should succeed")
        if let r = result {
            expectEqual(bytesToHex(Array(r[0..<64])), bytesToHex(bls12G1X), "G1Add(G1,O).x == G1.x")
            expectEqual(bytesToHex(Array(r[64..<128])), bytesToHex(bls12G1Y), "G1Add(G1,O).y == G1.y")
        }
    }

    // Test 2: O + O = O
    do {
        let input = [UInt8](repeating: 0, count: 256)
        let result = EVMPrecompile0A_bls12381G1Add(input: input)
        expect(result != nil, "BLS G1Add(O, O) should succeed")
        if let r = result {
            expect(r.allSatisfy({ $0 == 0 }), "G1Add(O,O) should be identity")
        }
    }

    // Test 3: Wrong input length
    do {
        let result = EVMPrecompile0A_bls12381G1Add(input: [UInt8](repeating: 0, count: 100))
        expect(result == nil, "BLS G1Add with wrong length should fail")
    }

    // Test 4: G1 + G1 = 2*G1 (verify via scalar mul)
    do {
        let addInput = bls12G1X + bls12G1Y + bls12G1X + bls12G1Y
        let addResult = EVMPrecompile0A_bls12381G1Add(input: addInput)

        let mulInput = bls12G1X + bls12G1Y + padLeft(hexToBytes("02"), to: 32)
        let mulResult = EVMPrecompile0B_bls12381G1Mul(input: mulInput)

        expect(addResult != nil, "G1Add(G1,G1) should succeed")
        expect(mulResult != nil, "G1Mul(G1,2) should succeed")
        if let a = addResult, let m = mulResult {
            expectEqual(bytesToHex(a), bytesToHex(m), "G1+G1 == 2*G1")
        }
    }

    suite("EVM Precompile: BLS12-381 G1 Mul (0x0B)")

    // Test 1: 1 * G1 = G1
    do {
        let input = bls12G1X + bls12G1Y + padLeft(hexToBytes("01"), to: 32)
        let result = EVMPrecompile0B_bls12381G1Mul(input: input)
        expect(result != nil, "BLS G1Mul(G1, 1) should succeed")
        if let r = result {
            expectEqual(bytesToHex(Array(r[0..<64])), bytesToHex(bls12G1X), "G1Mul(G1,1).x == G1.x")
            expectEqual(bytesToHex(Array(r[64..<128])), bytesToHex(bls12G1Y), "G1Mul(G1,1).y == G1.y")
        }
    }

    // Test 2: 0 * G1 = O
    do {
        let input = bls12G1X + bls12G1Y + [UInt8](repeating: 0, count: 32)
        let result = EVMPrecompile0B_bls12381G1Mul(input: input)
        expect(result != nil, "BLS G1Mul(G1, 0) should succeed")
        if let r = result {
            expect(r.allSatisfy({ $0 == 0 }), "G1Mul(G1, 0) should be identity")
        }
    }

    // Test 3: Wrong input length
    do {
        let result = EVMPrecompile0B_bls12381G1Mul(input: [UInt8](repeating: 0, count: 100))
        expect(result == nil, "BLS G1Mul with wrong length should fail")
    }

    suite("EVM Precompile: BLS12-381 Pairing (0x10)")

    // Test 1: Empty input -> 1
    do {
        let result = EVMPrecompile10_bls12381Pairing(input: [])
        expect(result != nil, "BLS pairing([]) should succeed")
        if let r = result {
            expectEqual(r[31], 1, "empty BLS pairing should return 1")
        }
    }

    // Test 2: Invalid length
    do {
        let result = EVMPrecompile10_bls12381Pairing(input: [UInt8](repeating: 0, count: 100))
        expect(result == nil, "BLS pairing with invalid length should fail")
    }

    // Test 3: All-zero pair (O, O) -> 1
    do {
        let input = [UInt8](repeating: 0, count: 384)
        let result = EVMPrecompile10_bls12381Pairing(input: input)
        expect(result != nil, "BLS pairing(O, O) should succeed")
        if let r = result {
            expectEqual(r[31], 1, "BLS pairing(O, O) = 1")
        }
    }

    suite("EVM Precompile: Gas Metering")

    // Verify gas costs match EIP specs
    do {
        let runner = EVMPrecompileRunner()

        let addCall = EVMPrecompileCall(id: .bn254Add, input: [UInt8](repeating: 0, count: 128))
        expectEqual(runner.gasCost(for: addCall), 150, "ecAdd gas = 150")

        let mulCall = EVMPrecompileCall(id: .bn254Mul, input: [UInt8](repeating: 0, count: 96))
        expectEqual(runner.gasCost(for: mulCall), 6000, "ecMul gas = 6000")

        let pair1Call = EVMPrecompileCall(id: .bn254Pairing, input: [UInt8](repeating: 0, count: 192))
        expectEqual(runner.gasCost(for: pair1Call), 45000 + 34000, "ecPairing(1 pair) gas = 79000")

        let pair2Call = EVMPrecompileCall(id: .bn254Pairing, input: [UInt8](repeating: 0, count: 384))
        expectEqual(runner.gasCost(for: pair2Call), 45000 + 2 * 34000, "ecPairing(2 pairs) gas = 113000")

        let blsAddCall = EVMPrecompileCall(id: .bls12381G1Add, input: [UInt8](repeating: 0, count: 256))
        expectEqual(runner.gasCost(for: blsAddCall), 500, "BLS G1Add gas = 500")

        let blsMulCall = EVMPrecompileCall(id: .bls12381G1Mul, input: [UInt8](repeating: 0, count: 160))
        expectEqual(runner.gasCost(for: blsMulCall), 12000, "BLS G1Mul gas = 12000")

        let blsPairCall = EVMPrecompileCall(id: .bls12381Pairing, input: [UInt8](repeating: 0, count: 384))
        expectEqual(runner.gasCost(for: blsPairCall), 115000 + 23000, "BLS Pairing(1 pair) gas = 138000")
    }

    suite("EVM Precompile: Batch Runner")

    // Run a small batch and verify reporting
    do {
        let runner = EVMPrecompileRunner()
        let calls: [EVMPrecompileCall] = [
            EVMPrecompileCall(id: .bn254Add, input: [UInt8](repeating: 0, count: 128)),
            EVMPrecompileCall(id: .bn254Mul, input: BN254_G1_X + BN254_G1_Y + SCALAR_TWO),
            EVMPrecompileCall(id: .bn254Pairing, input: []),
        ]

        let batch = runner.executeBatch(calls)
        expectEqual(batch.results.count, 3, "batch should have 3 results")
        expectEqual(batch.successCount, 3, "all calls should succeed")
        expectEqual(batch.failureCount, 0, "no failures")
        expectEqual(batch.totalGasUsed, 150 + 6000 + 45000, "total gas matches sum")
        expect(batch.totalDurationMs > 0, "batch took nonzero time")
    }
}
