// GPURangeProofProtocolEngine tests
//
// Verifies:
//   - Single range proof: prove + verify for various bit sizes
//   - Bit decomposition correctness (value = 0, max, mid-range)
//   - Commitment binding (different blinding -> different V)
//   - Verification rejects wrong commitment, wrong bit size
//   - Batch/aggregated range proofs for multiple values
//   - Batch verification rejects tampered proofs
//   - Inner product argument consistency
//   - Protocol transcript determinism
//   - Proof size computation
//   - GPU vs CPU fallback consistency
//   - Edge cases: single-bit values, power-of-two boundaries
//   - ProtocolGenerators generation and batch generation
//   - Config validation

import zkMetal
import Foundation

public func runGPURangeProofProtocolTests() {
    suite("GPU Range Proof Protocol Engine")

    testSingleProof8Bit()
    testSingleProof16Bit()
    testSingleProof32Bit()
    testValueZero()
    testValueMax8Bit()
    testValueMax16Bit()
    testValueOne()
    testDifferentBlindingsDifferentCommitments()
    testSameBlindingSameProof()
    testVerifyRejectsWrongCommitment()
    testVerifyRejectsBitSizeMismatch()
    testBatchProve2Values()
    testBatchProve4Values()
    testBatchVerifyRejectsTamperedCommitment()
    testBatchVerifyRejectsEmptyMismatch()
    testProofSizeComputation()
    testBatchProofSizeComputation()
    testProtocolGeneratorsGenerate()
    testProtocolGeneratorsBatchGenerate()
    testConfigDefaults()
    testConfigCustomBitSize()
    testConfigGPUDisabled()
    testCPUOnlyProveVerify()
    testTranscriptDeterminism()
    testIPAProofStructure()
    testBatchIPAProofStructure()
    testProtocolRangeProofFields()
    testBatchProtocolRangeProofFields()
    testMultipleBitSizes()
    testVersionReported()

    print("  GPU Range Proof Protocol engine version: \(GPURangeProofProtocolEngine.version.description)")
}

// MARK: - Helpers

private func frEq(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
           a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

private func makeEngine(gpu: Bool = true) -> GPURangeProofProtocolEngine {
    let config = RangeProofProtocolConfig(bitSize: 8, useGPU: gpu, gpuThreshold: 16)
    return GPURangeProofProtocolEngine(config: config)
}

private func makeEngineWithBitSize(_ n: Int, gpu: Bool = true) -> GPURangeProofProtocolEngine {
    let config = RangeProofProtocolConfig(bitSize: n, useGPU: gpu, gpuThreshold: 16)
    return GPURangeProofProtocolEngine(config: config)
}

// MARK: - Single Value Prove/Verify

private func testSingleProof8Bit() {
    let engine = makeEngine()
    let generators = ProtocolGenerators.generate(n: 8)
    let blinding = frFromInt(42)
    let value: UInt64 = 200

    let proof = engine.prove(value: value, blinding: blinding, generators: generators)

    // Verify commitment matches V = value*g + blinding*h
    let gProj = pointFromAffine(generators.g)
    let hProj = pointFromAffine(generators.h)
    let expectedV = pointAdd(cPointScalarMul(gProj, frFromInt(value)),
                             cPointScalarMul(hProj, blinding))
    expect(pointEqual(proof.V, expectedV), "8-bit: commitment correct")

    let valid = engine.verify(proof: proof, commitment: proof.V, generators: generators)
    expect(valid, "8-bit range proof: value 200 verifies")
}

private func testSingleProof16Bit() {
    let engine = makeEngineWithBitSize(16)
    let generators = ProtocolGenerators.generate(n: 16)
    let blinding = frFromInt(1337)
    let value: UInt64 = 50000

    let proof = engine.prove(value: value, blinding: blinding, generators: generators)
    let valid = engine.verify(proof: proof, commitment: proof.V, generators: generators)
    expect(valid, "16-bit range proof: value 50000 verifies")
    expectEqual(proof.bitSize, 16, "16-bit proof records correct bit size")
}

private func testSingleProof32Bit() {
    let engine = makeEngineWithBitSize(32)
    let generators = ProtocolGenerators.generate(n: 32)
    let blinding = frFromInt(9999)
    let value: UInt64 = 1_000_000

    let proof = engine.prove(value: value, blinding: blinding, generators: generators)
    let valid = engine.verify(proof: proof, commitment: proof.V, generators: generators)
    expect(valid, "32-bit range proof: value 1000000 verifies")
    expectEqual(proof.bitSize, 32, "32-bit proof records correct bit size")
}

// MARK: - Edge Cases: Value Boundaries

private func testValueZero() {
    let engine = makeEngine()
    let generators = ProtocolGenerators.generate(n: 8)
    let blinding = frFromInt(99)

    let proof = engine.prove(value: 0, blinding: blinding, generators: generators)
    let valid = engine.verify(proof: proof, commitment: proof.V, generators: generators)
    expect(valid, "8-bit range proof: value = 0 verifies")
}

private func testValueMax8Bit() {
    let engine = makeEngine()
    let generators = ProtocolGenerators.generate(n: 8)
    let blinding = frFromInt(77)
    let value: UInt64 = 255

    let proof = engine.prove(value: value, blinding: blinding, generators: generators)
    let valid = engine.verify(proof: proof, commitment: proof.V, generators: generators)
    expect(valid, "8-bit range proof: value = 255 (max) verifies")
}

private func testValueMax16Bit() {
    let engine = makeEngineWithBitSize(16)
    let generators = ProtocolGenerators.generate(n: 16)
    let blinding = frFromInt(88)
    let value: UInt64 = 65535

    let proof = engine.prove(value: value, blinding: blinding, generators: generators)
    let valid = engine.verify(proof: proof, commitment: proof.V, generators: generators)
    expect(valid, "16-bit range proof: value = 65535 (max) verifies")
}

private func testValueOne() {
    let engine = makeEngine()
    let generators = ProtocolGenerators.generate(n: 8)
    let blinding = frFromInt(11)

    let proof = engine.prove(value: 1, blinding: blinding, generators: generators)
    let valid = engine.verify(proof: proof, commitment: proof.V, generators: generators)
    expect(valid, "8-bit range proof: value = 1 verifies")
}

// MARK: - Commitment Binding

private func testDifferentBlindingsDifferentCommitments() {
    let engine = makeEngine()
    let generators = ProtocolGenerators.generate(n: 8)

    let proof1 = engine.prove(value: 42, blinding: frFromInt(1), generators: generators)
    let proof2 = engine.prove(value: 42, blinding: frFromInt(2), generators: generators)

    // Same value, different blinding -> different V commitments
    expect(!pointEqual(proof1.V, proof2.V),
           "Different blindings produce different commitments")

    // Both should verify
    let v1 = engine.verify(proof: proof1, commitment: proof1.V, generators: generators)
    let v2 = engine.verify(proof: proof2, commitment: proof2.V, generators: generators)
    expect(v1, "Proof with blinding=1 verifies")
    expect(v2, "Proof with blinding=2 verifies")
}

private func testSameBlindingSameProof() {
    let engine = makeEngine()
    let generators = ProtocolGenerators.generate(n: 8)
    let blinding = frFromInt(42)

    let proof1 = engine.prove(value: 100, blinding: blinding, generators: generators)
    let proof2 = engine.prove(value: 100, blinding: blinding, generators: generators)

    // Same value and blinding -> same commitment (deterministic)
    expect(pointEqual(proof1.V, proof2.V),
           "Same value and blinding produce same commitment")
    expect(pointEqual(proof1.A, proof2.A),
           "Deterministic: same A commitment")
    expect(pointEqual(proof1.S, proof2.S),
           "Deterministic: same S commitment")
}

// MARK: - Verification Rejection

private func testVerifyRejectsWrongCommitment() {
    let engine = makeEngine()
    let generators = ProtocolGenerators.generate(n: 8)
    let blinding = frFromInt(42)

    let proof = engine.prove(value: 100, blinding: blinding, generators: generators)

    // Create a wrong commitment (different value)
    let gProj = pointFromAffine(generators.g)
    let hProj = pointFromAffine(generators.h)
    let wrongV = pointAdd(cPointScalarMul(gProj, frFromInt(101)),
                          cPointScalarMul(hProj, blinding))

    let valid = engine.verify(proof: proof, commitment: wrongV, generators: generators)
    expect(!valid, "Rejects wrong commitment")
}

private func testVerifyRejectsBitSizeMismatch() {
    let engine = makeEngine()
    let gen8 = ProtocolGenerators.generate(n: 8)
    let gen16 = ProtocolGenerators.generate(n: 16)
    let blinding = frFromInt(42)

    let proof = engine.prove(value: 100, blinding: blinding, generators: gen8)

    // Try to verify 8-bit proof with 16-bit generators
    let valid = engine.verify(proof: proof, commitment: proof.V, generators: gen16)
    expect(!valid, "Rejects bit size mismatch (8 vs 16)")
}

// MARK: - Batch Prove/Verify

private func testBatchProve2Values() {
    let engine = makeEngine()
    let bitSize = 8
    let m = 2
    let generators = ProtocolGenerators.generateBatch(n: bitSize, count: m)
    let values: [UInt64] = [10, 200]
    let blindings = [frFromInt(1), frFromInt(2)]

    let batchProof = engine.batchProve(values: values, blindings: blindings,
                                        generators: generators, bitSize: bitSize)

    expectEqual(batchProof.count, 2, "Batch proof count = 2")
    expectEqual(batchProof.bitSize, 8, "Batch proof bitSize = 8")
    expectEqual(batchProof.commitments.count, 2, "2 commitments")

    // Verify each commitment matches V_i = v_i*g + gamma_i*h
    let gProj = pointFromAffine(generators.g)
    let hProj = pointFromAffine(generators.h)
    for i in 0..<m {
        let expectedV = pointAdd(cPointScalarMul(gProj, frFromInt(values[i])),
                                 cPointScalarMul(hProj, blindings[i]))
        expect(pointEqual(batchProof.commitments[i], expectedV),
               "Batch commitment \(i) correct")
    }

    let valid = engine.batchVerify(proof: batchProof,
                                    commitments: batchProof.commitments,
                                    generators: generators)
    expect(valid, "Batch verify 2 values passes")
}

private func testBatchProve4Values() {
    let engine = makeEngine()
    let bitSize = 8
    let m = 4
    let generators = ProtocolGenerators.generateBatch(n: bitSize, count: m)
    let values: [UInt64] = [0, 100, 200, 255]
    let blindings = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]

    let batchProof = engine.batchProve(values: values, blindings: blindings,
                                        generators: generators, bitSize: bitSize)

    expectEqual(batchProof.count, 4, "Batch proof count = 4")

    let valid = engine.batchVerify(proof: batchProof,
                                    commitments: batchProof.commitments,
                                    generators: generators)
    expect(valid, "Batch verify 4 values passes")
}

private func testBatchVerifyRejectsTamperedCommitment() {
    let engine = makeEngine()
    let bitSize = 8
    let m = 2
    let generators = ProtocolGenerators.generateBatch(n: bitSize, count: m)
    let values: [UInt64] = [50, 150]
    let blindings = [frFromInt(10), frFromInt(20)]

    let batchProof = engine.batchProve(values: values, blindings: blindings,
                                        generators: generators, bitSize: bitSize)

    // Tamper: swap commitments
    let tamperedCommitments = [batchProof.commitments[1], batchProof.commitments[0]]
    let valid = engine.batchVerify(proof: batchProof,
                                    commitments: tamperedCommitments,
                                    generators: generators)
    expect(!valid, "Batch verify rejects tampered commitments")
}

private func testBatchVerifyRejectsEmptyMismatch() {
    let engine = makeEngine()
    let bitSize = 8
    let m = 2
    let generators = ProtocolGenerators.generateBatch(n: bitSize, count: m)
    let values: [UInt64] = [10, 20]
    let blindings = [frFromInt(1), frFromInt(2)]

    let batchProof = engine.batchProve(values: values, blindings: blindings,
                                        generators: generators, bitSize: bitSize)

    // Wrong number of commitments
    let valid = engine.batchVerify(proof: batchProof,
                                    commitments: [batchProof.commitments[0]],
                                    generators: generators)
    expect(!valid, "Batch verify rejects count mismatch")
}

// MARK: - Proof Size

private func testProofSizeComputation() {
    let size8 = GPURangeProofProtocolEngine.proofSize(bitSize: 8)
    // log2(8) = 3, so 5 + 2*3 = 11 groups, 3 + 2 = 5 fields
    expectEqual(size8.groups, 11, "8-bit proof: 11 group elements")
    expectEqual(size8.fields, 5, "8-bit proof: 5 field elements")

    let size16 = GPURangeProofProtocolEngine.proofSize(bitSize: 16)
    // log2(16) = 4, so 5 + 2*4 = 13 groups
    expectEqual(size16.groups, 13, "16-bit proof: 13 group elements")
    expectEqual(size16.fields, 5, "16-bit proof: 5 field elements")

    let size32 = GPURangeProofProtocolEngine.proofSize(bitSize: 32)
    // log2(32) = 5, so 5 + 2*5 = 15 groups
    expectEqual(size32.groups, 15, "32-bit proof: 15 group elements")
}

private func testBatchProofSizeComputation() {
    let size = GPURangeProofProtocolEngine.batchProofSize(bitSize: 8, count: 4)
    // totalN = 32, log2(32) = 5
    // groups = 4 + 4 + 2*5 = 18, fields = 5
    expectEqual(size.groups, 18, "Batch 4x8-bit: 18 group elements")
    expectEqual(size.fields, 5, "Batch 4x8-bit: 5 field elements")
}

// MARK: - Generator Generation

private func testProtocolGeneratorsGenerate() {
    let gen = ProtocolGenerators.generate(n: 8)
    expectEqual(gen.n, 8, "Generators n = 8")
    expectEqual(gen.G.count, 8, "G has 8 points")
    expectEqual(gen.H.count, 8, "H has 8 points")

    // All generators should be distinct (check G[0] != G[1] and G[0] != H[0])
    let g0 = pointFromAffine(gen.G[0])
    let g1 = pointFromAffine(gen.G[1])
    let h0 = pointFromAffine(gen.H[0])
    expect(!pointEqual(g0, g1), "G[0] != G[1]")
    expect(!pointEqual(g0, h0), "G[0] != H[0]")

    // g, h, u should all be distinct
    let gP = pointFromAffine(gen.g)
    let hP = pointFromAffine(gen.h)
    let uP = pointFromAffine(gen.u)
    expect(!pointEqual(gP, hP), "g != h")
    expect(!pointEqual(gP, uP), "g != u")
    expect(!pointEqual(hP, uP), "h != u")
}

private func testProtocolGeneratorsBatchGenerate() {
    let gen = ProtocolGenerators.generateBatch(n: 8, count: 4)
    expectEqual(gen.n, 32, "Batch generators n = 32 (4 * 8)")
    expectEqual(gen.G.count, 32, "G has 32 points")
    expectEqual(gen.H.count, 32, "H has 32 points")
}

// MARK: - Configuration

private func testConfigDefaults() {
    let config = RangeProofProtocolConfig()
    expectEqual(config.bitSize, 32, "Default bitSize = 32")
    expect(config.useGPU, "Default useGPU = true")
    expectEqual(config.gpuThreshold, 16, "Default gpuThreshold = 16")
}

private func testConfigCustomBitSize() {
    let config = RangeProofProtocolConfig(bitSize: 16, useGPU: false, gpuThreshold: 8)
    expectEqual(config.bitSize, 16, "Custom bitSize = 16")
    expect(!config.useGPU, "Custom useGPU = false")
    expectEqual(config.gpuThreshold, 8, "Custom gpuThreshold = 8")
}

private func testConfigGPUDisabled() {
    let config = RangeProofProtocolConfig(bitSize: 8, useGPU: false)
    let engine = GPURangeProofProtocolEngine(config: config)
    expect(!engine.isGPUEnabled, "GPU disabled when config.useGPU = false")
}

// MARK: - CPU-Only Path

private func testCPUOnlyProveVerify() {
    let config = RangeProofProtocolConfig(bitSize: 8, useGPU: false, gpuThreshold: 16)
    let engine = GPURangeProofProtocolEngine(config: config)
    let generators = ProtocolGenerators.generate(n: 8)
    let blinding = frFromInt(55)

    let proof = engine.prove(value: 123, blinding: blinding, generators: generators)
    let valid = engine.verify(proof: proof, commitment: proof.V, generators: generators)
    expect(valid, "CPU-only prove/verify works for value 123")
}

// MARK: - Transcript Determinism

private func testTranscriptDeterminism() {
    let engine = makeEngine()
    let generators = ProtocolGenerators.generate(n: 8)
    let blinding = frFromInt(42)

    let proof1 = engine.prove(value: 100, blinding: blinding, generators: generators)
    let proof2 = engine.prove(value: 100, blinding: blinding, generators: generators)

    // Same inputs -> same proof (deterministic transcript)
    expect(frEq(proof1.tHat, proof2.tHat), "Deterministic: same tHat")
    expect(frEq(proof1.taux, proof2.taux), "Deterministic: same taux")
    expect(frEq(proof1.mu, proof2.mu), "Deterministic: same mu")
    expect(pointEqual(proof1.T1, proof2.T1), "Deterministic: same T1")
    expect(pointEqual(proof1.T2, proof2.T2), "Deterministic: same T2")
}

// MARK: - IPA Proof Structure

private func testIPAProofStructure() {
    let engine = makeEngine()
    let generators = ProtocolGenerators.generate(n: 8)
    let blinding = frFromInt(42)

    let proof = engine.prove(value: 200, blinding: blinding, generators: generators)
    let ipa = proof.ipProof

    // For n=8, IPA should have log2(8) = 3 rounds
    expectEqual(ipa.Ls.count, 3, "IPA has 3 L commitments for n=8")
    expectEqual(ipa.Rs.count, 3, "IPA has 3 R commitments for n=8")

    // Final scalars should be non-zero (with overwhelming probability)
    expect(!ipa.a.isZero, "IPA final scalar a is non-zero")
    expect(!ipa.b.isZero, "IPA final scalar b is non-zero")
}

private func testBatchIPAProofStructure() {
    let engine = makeEngine()
    let bitSize = 8
    let m = 2
    let generators = ProtocolGenerators.generateBatch(n: bitSize, count: m)
    let values: [UInt64] = [10, 200]
    let blindings = [frFromInt(1), frFromInt(2)]

    let batchProof = engine.batchProve(values: values, blindings: blindings,
                                        generators: generators, bitSize: bitSize)
    let ipa = batchProof.ipProof

    // totalN = 16, log2(16) = 4 rounds
    expectEqual(ipa.Ls.count, 4, "Batch IPA has 4 L commitments for totalN=16")
    expectEqual(ipa.Rs.count, 4, "Batch IPA has 4 R commitments for totalN=16")
}

// MARK: - Proof Fields

private func testProtocolRangeProofFields() {
    let engine = makeEngine()
    let generators = ProtocolGenerators.generate(n: 8)
    let blinding = frFromInt(42)

    let proof = engine.prove(value: 100, blinding: blinding, generators: generators)

    expectEqual(proof.bitSize, 8, "Proof bitSize = 8")

    // V should not be identity
    expect(!pointIsIdentity(proof.V), "V is not identity")
    expect(!pointIsIdentity(proof.A), "A is not identity")
    expect(!pointIsIdentity(proof.S), "S is not identity")
    expect(!pointIsIdentity(proof.T1), "T1 is not identity")
    expect(!pointIsIdentity(proof.T2), "T2 is not identity")

    // Field elements should be non-zero (with overwhelming probability)
    expect(!proof.tHat.isZero, "tHat is non-zero")
    expect(!proof.taux.isZero, "taux is non-zero")
    expect(!proof.mu.isZero, "mu is non-zero")
}

private func testBatchProtocolRangeProofFields() {
    let engine = makeEngine()
    let bitSize = 8
    let m = 2
    let generators = ProtocolGenerators.generateBatch(n: bitSize, count: m)
    let values: [UInt64] = [50, 150]
    let blindings = [frFromInt(5), frFromInt(6)]

    let proof = engine.batchProve(values: values, blindings: blindings,
                                   generators: generators, bitSize: bitSize)

    expectEqual(proof.count, 2, "Batch proof count = 2")
    expectEqual(proof.bitSize, 8, "Batch proof bitSize = 8")
    expectEqual(proof.commitments.count, 2, "2 commitments in batch proof")
    expectEqual(proof.As.count, 1, "1 aggregated A commitment")
    expectEqual(proof.Ss.count, 1, "1 aggregated S commitment")

    expect(!proof.tHat.isZero, "Batch tHat is non-zero")
    expect(!proof.taux.isZero, "Batch taux is non-zero")
}

// MARK: - Multiple Bit Sizes

private func testMultipleBitSizes() {
    // Verify that the protocol works consistently across different bit sizes
    let bitSizes = [8, 16]
    for n in bitSizes {
        let engine = makeEngineWithBitSize(n)
        let generators = ProtocolGenerators.generate(n: n)
        let blinding = frFromInt(42)
        let value: UInt64 = UInt64(min(100, (1 << n) - 1))

        let proof = engine.prove(value: value, blinding: blinding, generators: generators)
        let valid = engine.verify(proof: proof, commitment: proof.V, generators: generators)
        expect(valid, "\(n)-bit range proof verifies for value \(value)")
        expectEqual(proof.bitSize, n, "Proof records bitSize = \(n)")
    }
}

// MARK: - Version

private func testVersionReported() {
    let version = GPURangeProofProtocolEngine.version
    expect(!version.version.isEmpty, "Version string is non-empty")
    expect(!version.updated.isEmpty, "Updated date is non-empty")
    expect(version.description.contains("1.0.0"), "Version is 1.0.0")
}
