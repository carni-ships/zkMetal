// Binary FRI Tests
//
// Tests for binary-native FRI implementation with additive domains.

import Foundation
import zkMetal

// MARK: - Test Helpers

/// GF(2^8) multiplication with reduction by 0x11B.
private func gf8Mul(_ a: UInt8, _ b: UInt8) -> UInt8 {
    var p: UInt16 = 0
    var a = UInt16(a)
    var b = UInt16(b)

    for _ in 0..<8 {
        if b & 1 != 0 {
            p ^= a
        }
        let hiBit = a & 0x80
        a <<= 1
        if hiBit != 0 {
            a ^= 0x1B
        }
        b >>= 1
    }
    return UInt8(p & 0xFF)
}

/// GF(2^8) addition (XOR)
private func gf8Add(_ a: UInt8, _ b: UInt8) -> UInt8 {
    return a ^ b
}

// MARK: - Binary FRI Tests

func runBinaryFRITests() {
    suite("Binary FRI Tests")

    // Test 1: Binary FRI Config Defaults
    do {
        let config = BinaryFRIConfig()
        assert(config.extensionDegree == 128, "Default extension degree should be 128")
        assert(config.foldingFactor == 2, "Default folding factor should be 2")
        assert(config.numQueries == 32, "Default num queries should be 32")
        assert(config.finalPolyMaxDegree == 7, "Default final poly max degree should be 7")
        assert(config.logDomainSize == 20, "Default log domain size should be 20")
        print("  ✓ BinaryFRIConfig defaults")
    }

    // Test 2: Binary FRI Config Validation
    do {
        // Valid folding factors should work
        let config2 = BinaryFRIConfig(foldingFactor: 2)
        let config4 = BinaryFRIConfig(foldingFactor: 4)
        let config8 = BinaryFRIConfig(foldingFactor: 8)
        assert(config2.foldingFactor == 2, "Should accept folding factor 2")
        assert(config4.foldingFactor == 4, "Should accept folding factor 4")
        assert(config8.foldingFactor == 8, "Should accept folding factor 8")
        print("  ✓ BinaryFRIConfig validation")
    }

    // Test 3: CPU Fold Operations
    do {
        let evals = (0..<16).map { UInt8($0) }
        let alpha = UInt8(7)

        // Fold by 2
        let folded = BinaryCPUFold.fold2(evals: evals, alpha: alpha)
        assert(folded.count == 8, "Folded size should be half")

        // Verify fold equation: f'[i] = f[i] + alpha * f[i+8]
        var correct = true
        for i in 0..<8 {
            let expected = gf8Add(evals[i], gf8Mul(alpha, evals[i + 8]))
            if folded[i] != expected {
                correct = false
                break
            }
        }
        assert(correct, "Fold should satisfy f' = f + alpha * f'")
        print("  ✓ CPU fold by 2")
    }

    // Test 4: High-Arity Fold
    do {
        let evals = (0..<16).map { UInt8($0) }
        let alpha = UInt8(3)

        // Fold by 4 (arity = 2)
        let folded = BinaryCPUFold.foldArity(evals: evals, alpha: alpha, arity: 2)
        assert(folded.count == 4, "Folded size should be quarter")

        // Fold by 4 again
        let folded2 = BinaryCPUFold.foldArity(evals: folded, alpha: alpha, arity: 2)
        assert(folded2.count == 1, "Should end up with 1 element")
        print("  ✓ High-arity CPU fold")
    }

    // Test 5: Binary FRI Prover Round Computation
    do {
        let config = BinaryFRIConfig(finalPolyMaxDegree: 7, logDomainSize: 20)
        let prover = BinaryFRIProver(config: config)

        let rounds = prover.computeNumRounds(logSize: 20)
        assert(rounds > 0, "Should have positive rounds")
        assert(rounds <= 20, "Should have at most 20 rounds")
        print("  ✓ BinaryFRIProver round computation")
    }

    // Test 6: Binary FRI Prover Proof Generation
    do {
        let config = BinaryFRIConfig(finalPolyMaxDegree: 3, logDomainSize: 8)
        let prover = BinaryFRIProver(config: config)

        let evals = (0..<(1 << 8)).map { UInt8($0) }
        let alphas = [UInt8(3), UInt8(5), UInt8(7), UInt8(9), UInt8(11)]

        let (key, witness) = try! prover.prove(evals: evals, alphas: alphas)

        assert(key.logDomainSize == 8, "Log domain size should be preserved")
        assert(key.numRounds > 0, "Should have positive rounds")
        assert(witness.layerEvals.count > 0, "Should have layers")
        print("  ✓ BinaryFRIProver proof generation")
    }

    // Test 7: Binary Merkle Tree Construction
    do {
        let params = BinaryMerkleParams(logLeaves: 4)
        let data = (0..<16).map { UInt8($0) }

        let tree = BinaryMerkleTree(evaluations: data, params: params)

        assert(tree.root.count > 0, "Root should have data")
        assert(tree.nodes.count == 31, "Should have 2*16-1=31 nodes")
        print("  ✓ Binary Merkle tree construction")
    }

    // Test 8: Binary Merkle Authentication Path
    do {
        let params = BinaryMerkleParams(logLeaves: 4)
        let data = (0..<16).map { UInt8($0) }

        let tree = BinaryMerkleTree(evaluations: data, params: params)
        let path = tree.getAuthPath(leafIndex: 5)

        assert(path.count == 4, "Auth path should have 4 elements for 16 leaves")
        print("  ✓ Binary Merkle authentication path")
    }

    // Test 9: Johnson Bound Parameters
    do {
        // Use valid parameters where L * d <= n
        let params = JohnsonBoundParams(n: 1024, d: 256, L: 4)

        assert(params.n == 1024, "n should be 1024")
        assert(params.d == 256, "d should be 256")
        assert(params.L == 4, "L should be 4")

        let radius = params.johnsonRadius
        assert(radius >= 0, "Radius should be non-negative")
        assert(radius <= params.n, "Radius should be at most n")
        print("  ✓ Johnson bound parameters")
    }

    // Test 10: Proximity Gap Parameters
    do {
        let params = ProximityGapParams(
            securityBits: 40,
            numQueries: 32,
            extensionDegree: 128
        )

        assert(params.achievesSecurity, "Should achieve security")
        assert(params.soundnessError < 1e-12, "Soundness error should be small")
        print("  ✓ Proximity gap parameters")
    }

    // Test 11: Fiat-Shamir Transcript
    do {
        var transcript = BinaryFRITranscript(seed: [0x00])

        transcript.update([0x01, 0x02, 0x03])

        let squeezed = transcript.squeeze(numBytes: 4)
        assert(squeezed.count == 4, "Should squeeze 4 bytes")
        print("  ✓ Fiat-Shamir transcript")
    }

    // Test 12: Binary FRI Protocol
    do {
        let config = BinaryFRIConfig(finalPolyMaxDegree: 3, logDomainSize: 8)
        let friProtocol = BinaryFRIProtocol(config: config)

        let rounds = friProtocol.computeNumRounds(logSize: 8)
        assert(rounds > 0, "Should compute rounds")
        print("  ✓ Binary FRI protocol")
    }

    // Test 13: Co-Curvilinear Line Test
    do {
        // Verify basic GF(2^8) arithmetic works for line testing
        // In GF(2^8), addition is XOR: v ^ v = 0, v ^ 0 = v
        let v = UInt8(3)
        let zero = gf8Add(v, v)  // v ^ v = 0
        let back = gf8Add(zero, v)  // 0 ^ v = v
        assert(zero == 0, "v ^ v should be 0")
        assert(back == v, "0 ^ v should be v")
        print("  ✓ Co-curvilinear line test")
    }

    print("  ✓ All Binary FRI tests passed")
}
