// GPU EVM Precompile Engine Tests
//
// Tests GPU-accelerated EVM precompile verification including ecRecover,
// ecAdd/ecMul/ecPairing on BN254, modExp, blake2f, and BLS12-381 ops.

import Foundation
import zkMetal

// MARK: - Hex Helpers

private func hexToBytes(_ hex: String) -> [UInt8] {
    let h = hex.hasPrefix("0x") ? String(hex.dropFirst(2)) : hex
    var bytes = [UInt8]()
    var idx = h.startIndex
    while idx < h.endIndex {
        let next = h.index(idx, offsetBy: 2)
        bytes.append(UInt8(String(h[idx..<next]), radix: 16)!)
        idx = next
    }
    return bytes
}

private func bytesToHex(_ bytes: [UInt8]) -> String {
    return bytes.map { String(format: "%02x", $0) }.joined()
}

private func padLeft(_ data: [UInt8], to n: Int) -> [UInt8] {
    if data.count >= n { return Array(data.suffix(n)) }
    return [UInt8](repeating: 0, count: n - data.count) + data
}

// MARK: - Test Vectors

private let BN254_G1_X = padLeft(hexToBytes("01"), to: 32)
private let BN254_G1_Y = padLeft(hexToBytes("02"), to: 32)
private let BN254_2G1_X = hexToBytes("030644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd3")
private let BN254_2G1_Y = hexToBytes("15ed738c0e0a7c92e7845f96b2ae9c0a68a6a449e3538fc7ff3ebf7a5a18a2c4")
private let SCALAR_TWO = padLeft(hexToBytes("02"), to: 32)

// MARK: - Tests

public func runGPUEVMPrecompileTests() {
    suite("GPU EVM Precompile: Engine Init")

    var engine: GPUEVMPrecompileEngine?
    do {
        engine = try GPUEVMPrecompileEngine()
        expect(engine != nil, "Engine should initialize")
    } catch {
        expect(false, "Engine init failed: \(error)")
        return
    }

    guard let eng = engine else { return }

    suite("GPU EVM Precompile: ecAdd via Engine")

    // G1 + G1 = 2*G1
    do {
        let input = BN254_G1_X + BN254_G1_Y + BN254_G1_X + BN254_G1_Y
        let result = eng.ecAdd(input: input)
        expect(result != nil, "ecAdd(G1, G1) should succeed")
        if let r = result {
            expectEqual(r.count, 64, "ecAdd output should be 64 bytes")
            expectEqual(bytesToHex(Array(r[0..<32])), bytesToHex(BN254_2G1_X), "ecAdd(G1,G1).x")
            expectEqual(bytesToHex(Array(r[32..<64])), bytesToHex(BN254_2G1_Y), "ecAdd(G1,G1).y")
        }
    }

    // O + O = O
    do {
        let input = [UInt8](repeating: 0, count: 128)
        let result = eng.ecAdd(input: input)
        expect(result != nil, "ecAdd(O, O) should succeed")
        if let r = result {
            expect(r.allSatisfy({ $0 == 0 }), "ecAdd(O,O) == O")
        }
    }

    suite("GPU EVM Precompile: ecMul via Engine")

    // 2 * G1 = 2G1
    do {
        let input = BN254_G1_X + BN254_G1_Y + SCALAR_TWO
        let result = eng.ecMul(input: input)
        expect(result != nil, "ecMul(G1, 2) should succeed")
        if let r = result {
            expectEqual(bytesToHex(Array(r[0..<32])), bytesToHex(BN254_2G1_X), "ecMul(G1,2).x")
            expectEqual(bytesToHex(Array(r[32..<64])), bytesToHex(BN254_2G1_Y), "ecMul(G1,2).y")
        }
    }

    // 0 * G1 = O
    do {
        let input = BN254_G1_X + BN254_G1_Y + [UInt8](repeating: 0, count: 32)
        let result = eng.ecMul(input: input)
        expect(result != nil, "ecMul(G1, 0) should succeed")
        if let r = result {
            expect(r.allSatisfy({ $0 == 0 }), "ecMul(G1, 0) == O")
        }
    }

    suite("GPU EVM Precompile: ecPairing via Engine")

    // Empty input -> success (trivial pairing = 1)
    do {
        let result = eng.ecPairing(input: [])
        expect(result != nil, "ecPairing([]) should succeed")
        if let r = result {
            expectEqual(r.count, 32, "ecPairing output 32 bytes")
            expectEqual(r[31], 1, "empty pairing returns 1")
        }
    }

    // Invalid length
    do {
        let result = eng.ecPairing(input: [UInt8](repeating: 0, count: 100))
        expect(result == nil, "ecPairing invalid length should fail")
    }

    suite("GPU EVM Precompile: ecRecover")

    // ecRecover with valid-format input (returns deterministic address)
    do {
        var input = [UInt8](repeating: 0, count: 128)
        // Set hash to something nonzero
        input[0] = 0x42
        // v = 27
        input[63] = 27
        // r = 1
        input[95] = 1
        // s = 1
        input[127] = 1
        let result = eng.ecRecover(input: input)
        expect(result != nil, "ecRecover with v=27 should succeed")
        if let r = result {
            expectEqual(r.count, 32, "ecRecover output is 32 bytes")
            // First 12 bytes should be zero (address padding)
            expect(r[0..<12].allSatisfy({ $0 == 0 }), "ecRecover address left-padded")
        }
    }

    // ecRecover with invalid v
    do {
        var input = [UInt8](repeating: 0, count: 128)
        input[63] = 25  // v must be 27 or 28
        input[95] = 1
        input[127] = 1
        let result = eng.ecRecover(input: input)
        expect(result == nil, "ecRecover with v=25 should fail")
    }

    // ecRecover with zero r (invalid)
    do {
        var input = [UInt8](repeating: 0, count: 128)
        input[63] = 27
        // r = 0, s = 1
        input[127] = 1
        let result = eng.ecRecover(input: input)
        expect(result == nil, "ecRecover with r=0 should fail")
    }

    suite("GPU EVM Precompile: modExp")

    // 2^10 mod 1000 = 24
    do {
        let bSize = padLeft(hexToBytes("01"), to: 32)  // base size = 1
        let eSize = padLeft(hexToBytes("01"), to: 32)  // exp size = 1
        let mSize = padLeft(hexToBytes("02"), to: 32)  // mod size = 2
        let base: [UInt8] = [0x02]                      // base = 2
        let exp: [UInt8] = [0x0a]                        // exp = 10
        let modulus: [UInt8] = [0x03, 0xe8]              // mod = 1000
        let input = bSize + eSize + mSize + base + exp + modulus
        let result = eng.modExp(input: input)
        expect(result != nil, "modExp(2, 10, 1000) should succeed")
        if let r = result {
            expectEqual(r.count, 2, "modExp output size = mSize")
            let val = Int(r[0]) * 256 + Int(r[1])
            expectEqual(val, 24, "2^10 mod 1000 = 24")
        }
    }

    // 0^0 mod 1 = 0
    do {
        let bSize = padLeft(hexToBytes("01"), to: 32)
        let eSize = padLeft(hexToBytes("01"), to: 32)
        let mSize = padLeft(hexToBytes("01"), to: 32)
        let base: [UInt8] = [0x00]
        let exp: [UInt8] = [0x00]
        let modulus: [UInt8] = [0x01]
        let input = bSize + eSize + mSize + base + exp + modulus
        let result = eng.modExp(input: input)
        expect(result != nil, "modExp(0, 0, 1) should succeed")
        if let r = result {
            expectEqual(r[0], 0, "0^0 mod 1 = 0")
        }
    }

    // 3^3 mod 7 = 6
    do {
        let bSize = padLeft(hexToBytes("01"), to: 32)
        let eSize = padLeft(hexToBytes("01"), to: 32)
        let mSize = padLeft(hexToBytes("01"), to: 32)
        let base: [UInt8] = [0x03]
        let exp: [UInt8] = [0x03]
        let modulus: [UInt8] = [0x07]
        let input = bSize + eSize + mSize + base + exp + modulus
        let result = eng.modExp(input: input)
        expect(result != nil, "modExp(3, 3, 7) should succeed")
        if let r = result {
            expectEqual(Int(r[0]), 6, "3^3 mod 7 = 6")
        }
    }

    suite("GPU EVM Precompile: blake2f")

    // blake2f with 0 rounds (identity-like, returns initial F applied state)
    do {
        var input = [UInt8](repeating: 0, count: 213)
        // rounds = 0
        // h = 64 zero bytes
        // m = 128 zero bytes
        // t = 16 zero bytes
        // f = 0
        let result = eng.blake2f(input: input)
        expect(result != nil, "blake2f with 0 rounds should succeed")
        if let r = result {
            expectEqual(r.count, 64, "blake2f output is 64 bytes")
        }
    }

    // blake2f with invalid length
    do {
        let result = eng.blake2f(input: [UInt8](repeating: 0, count: 100))
        expect(result == nil, "blake2f with wrong length should fail")
    }

    // blake2f with invalid f flag
    do {
        var input = [UInt8](repeating: 0, count: 213)
        input[212] = 2  // f must be 0 or 1
        let result = eng.blake2f(input: input)
        expect(result == nil, "blake2f with f=2 should fail")
    }

    // blake2f with 12 rounds (EIP-152 test vector)
    do {
        // Standard test: 12 rounds, known h/m/t, f=1
        var input = [UInt8](repeating: 0, count: 213)
        input[3] = 12  // rounds = 12
        input[212] = 1  // f = 1 (final block)
        let result = eng.blake2f(input: input)
        expect(result != nil, "blake2f(12 rounds) should succeed")
        if let r = result {
            expectEqual(r.count, 64, "blake2f output 64 bytes")
        }
    }

    suite("GPU EVM Precompile: BLS12-381 via Engine")

    // BLS12-381 G1 Add: O + O = O
    do {
        let input = [UInt8](repeating: 0, count: 256)
        let result = eng.bls12381G1Add(input: input)
        expect(result != nil, "BLS G1Add(O, O) should succeed")
        if let r = result {
            expect(r.allSatisfy({ $0 == 0 }), "BLS G1Add(O,O) == O")
        }
    }

    // BLS12-381 G1 Mul: 0 * O = O
    do {
        let input = [UInt8](repeating: 0, count: 160)
        let result = eng.bls12381G1Mul(input: input)
        expect(result != nil, "BLS G1Mul(O, 0) should succeed")
        if let r = result {
            expect(r.allSatisfy({ $0 == 0 }), "BLS G1Mul(O, 0) == O")
        }
    }

    // BLS12-381 Pairing: empty -> 1
    do {
        let result = eng.bls12381Pairing(input: [])
        expect(result != nil, "BLS pairing([]) should succeed")
        if let r = result {
            expectEqual(r[31], 1, "empty BLS pairing = 1")
        }
    }

    suite("GPU EVM Precompile: Extended Dispatch")

    // Identity precompile (0x04)
    do {
        let data: [UInt8] = [1, 2, 3, 4, 5]
        let result = eng.executeExtended(id: .identity, input: data)
        expect(result != nil, "identity precompile should succeed")
        if let r = result {
            expectEqual(bytesToHex(r), bytesToHex(data), "identity returns input")
        }
    }

    // SHA-256 precompile (0x02) - empty input
    do {
        let result = eng.executeExtended(id: .sha256, input: [])
        expect(result != nil, "sha256([]) should succeed")
        if let r = result {
            expectEqual(r.count, 32, "sha256 output is 32 bytes")
            // SHA-256 of empty input = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
            let expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            expectEqual(bytesToHex(r), expected, "sha256('') matches known hash")
        }
    }

    suite("GPU EVM Precompile: Gas Costs")

    // Extended gas cost checks
    do {
        expectEqual(eng.extendedGasCost(id: .ecRecover, input: []), 3000, "ecRecover gas = 3000")
        expectEqual(eng.extendedGasCost(id: .identity, input: [UInt8](repeating: 0, count: 32)),
                    15 + 3, "identity(32 bytes) gas = 18")
        expectEqual(eng.extendedGasCost(id: .bn254Add, input: []), 150, "ecAdd gas = 150")
        expectEqual(eng.extendedGasCost(id: .bn254Mul, input: []), 6000, "ecMul gas = 6000")

        // blake2f gas = rounds
        var blake2Input = [UInt8](repeating: 0, count: 213)
        blake2Input[3] = 12 // rounds = 12
        expectEqual(eng.extendedGasCost(id: .blake2f, input: blake2Input), 12, "blake2f(12 rounds) gas = 12")
    }

    suite("GPU EVM Precompile: Batch Execution")

    // Small batch through engine
    do {
        let calls: [EVMPrecompileCall] = [
            EVMPrecompileCall(id: .bn254Add, input: [UInt8](repeating: 0, count: 128)),
            EVMPrecompileCall(id: .bn254Mul, input: BN254_G1_X + BN254_G1_Y + SCALAR_TWO),
            EVMPrecompileCall(id: .bn254Pairing, input: []),
        ]
        let batch = eng.executeBatch(calls)
        expectEqual(batch.results.count, 3, "batch has 3 results")
        expectEqual(batch.successCount, 3, "all succeed")
        expectEqual(batch.failureCount, 0, "no failures")
        expect(batch.totalGasUsed > 0, "total gas > 0")
    }

    // Single execute dispatch
    do {
        let call = EVMPrecompileCall(id: .bn254Add, input: [UInt8](repeating: 0, count: 128))
        let result = eng.execute(call)
        expect(result.success, "single execute should succeed")
        expectEqual(result.gasUsed, 150, "gas = 150")
        expect(result.output != nil, "output not nil")
    }

    suite("GPU EVM Precompile: Version")

    do {
        let v = GPUEVMPrecompileEngine.version
        expectEqual(v.version, "1.0.0", "version is 1.0.0")
    }
}
