// Tests for GPULassoEngine -- GPU-accelerated Lasso sparse lookup argument

import Foundation
import zkMetal

public func runGPULassoEngineTests() {
    suite("GPULassoEngine")

    testLassoSimpleRangeLookup()
    testLassoSubtableDecomposition()
    testLassoMemoryCheckConsistency()
    testLassoInvalidLookupDetection()
    testLassoStructuredTable()
}

// MARK: - Simple Range Lookup

func testLassoSimpleRangeLookup() {
    do {
        let engine = try GPULassoEngine()

        // 16-bit range check decomposed into 2 byte-sized subtables
        let table = GPULassoTable.build(.range(bits: 16, chunks: 2))

        // Lookup values: small numbers that fit in [0, 2^16)
        let lookups: [Fr] = [
            frFromInt(0),
            frFromInt(1),
            frFromInt(255),
            frFromInt(256),
            frFromInt(1000),
            frFromInt(65535),
            frFromInt(300),
            frFromInt(42)
        ]

        let beta = frFromInt(99999)
        let proof = try engine.prove(lookups: lookups, table: table, beta: beta)

        expect(proof.memoryCheckPassed, "Range lookup memory check passed")
        expect(proof.numChunks == 2, "Range table has 2 chunks")
        expect(proof.subtableProofs.count == 2, "2 subtable proofs")

        // Verify
        let valid = engine.verify(proof: proof, lookups: lookups, table: table)
        expect(valid, "Range lookup verification passed")

        // Check read sums equal table sums for each chunk
        for k in 0..<proof.numChunks {
            let sp = proof.subtableProofs[k]
            expect(frEqual(sp.readSum, sp.tableSum),
                   "Chunk \(k) memory check: readSum == tableSum")
        }
    } catch {
        expect(false, "Simple range lookup threw: \(error)")
    }
}

// MARK: - Subtable Decomposition

func testLassoSubtableDecomposition() {
    do {
        let engine = try GPULassoEngine()

        // 8-bit range, 2 chunks of 4 bits each (subtable size 16)
        let table = GPULassoTable.build(.range(bits: 8, chunks: 2))

        expect(table.numChunks == 2, "8-bit/2-chunk has 2 subtables")
        expect(table.subtables[0].count == 16, "Each subtable has 16 entries")

        // Test decomposition: value 0xAB = 171 = 0b10101011
        // chunk0 (low 4 bits) = 0xB = 11, chunk1 (high 4 bits) = 0xA = 10
        let val = frFromInt(0xAB)
        let decomposed = table.decompose(val)
        expect(decomposed.count == 2, "Decomposition produces 2 indices")
        expect(decomposed[0] == 0x0B, "Low nibble = 0xB")
        expect(decomposed[1] == 0x0A, "High nibble = 0xA")

        // Test recomposition
        let components = decomposed.map { table.subtables[0][$0] }
        let recomposed = table.compose(components)
        expect(frEqual(recomposed, val), "Recomposition recovers original value")

        // Prove and verify
        let lookups: [Fr] = [frFromInt(0), frFromInt(15), frFromInt(0xAB), frFromInt(0xFF)]
        let proof = try engine.prove(lookups: lookups, table: table)
        expect(proof.memoryCheckPassed, "Decomposed lookup passed")

        let valid = engine.verify(proof: proof, lookups: lookups, table: table)
        expect(valid, "Decomposed lookup verification passed")
    } catch {
        expect(false, "Subtable decomposition threw: \(error)")
    }
}

// MARK: - Memory Check Consistency

func testLassoMemoryCheckConsistency() {
    do {
        let engine = try GPULassoEngine()

        // Use timestamp-based memory checking
        let subtableSize = 8
        let subtable: [Fr] = (0..<subtableSize).map { frFromInt(UInt64($0)) }

        // Access pattern: indices into the subtable
        let indices = [0, 1, 2, 0, 1, 0, 3, 7]

        // Compute timestamps
        let (readTs, writeTs, finalTs) = engine.computeTimestamps(
            indices: indices, subtableSize: subtableSize)

        // Verify timestamps are consistent
        // Address 0 accessed 3 times: final_ts[0] should be 3
        expect(frEqual(finalTs[0], frFromInt(3)), "final_ts[0] = 3 (accessed 3 times)")
        // Address 1 accessed 2 times: final_ts[1] should be 2
        expect(frEqual(finalTs[1], frFromInt(2)), "final_ts[1] = 2 (accessed 2 times)")
        // Address 4 never accessed: final_ts[4] should be 0
        expect(frEqual(finalTs[4], Fr.zero), "final_ts[4] = 0 (never accessed)")

        // Verify read timestamps: first read of address 0 should have ts=0
        expect(frEqual(readTs[0], Fr.zero), "First read of addr 0: ts=0")
        // Third read of address 0 (index 5): ts should be 2
        expect(frEqual(readTs[5], frFromInt(2)), "Third read of addr 0: ts=2")

        // Write timestamps should be read + 1
        for i in 0..<indices.count {
            let expectedWrite = frAdd(readTs[i], Fr.one)
            expect(frEqual(writeTs[i], expectedWrite),
                   "write_ts[\(i)] = read_ts[\(i)] + 1")
        }

        // Verify multiset equality with random challenges
        let gamma = frFromInt(77777)
        let alpha = frFromInt(33333)
        let passed = try engine.verifyTimestampMemoryCheck(
            indices: indices, subtable: subtable,
            readTs: readTs, writeTs: writeTs, finalTs: finalTs,
            gamma: gamma, alpha: alpha)
        expect(passed, "Timestamp memory check multiset equality holds")
    } catch {
        expect(false, "Memory check consistency threw: \(error)")
    }
}

// MARK: - Invalid Lookup Detection

func testLassoInvalidLookupDetection() {
    do {
        let engine = try GPULassoEngine()

        // 8-bit range, 2 chunks of 4 bits
        let table = GPULassoTable.build(.range(bits: 8, chunks: 2))

        // Valid lookups first
        let validLookups: [Fr] = [frFromInt(0), frFromInt(100), frFromInt(255), frFromInt(42)]
        let validProof = try engine.prove(lookups: validLookups, table: table)
        let validResult = engine.verify(proof: validProof, lookups: validLookups, table: table)
        expect(validResult, "Valid lookups pass verification")

        // Tamper with proof: modify a read inverse
        let sp0 = validProof.subtableProofs[0]
        var badReadInvs = sp0.readInverses
        badReadInvs[0] = frAdd(badReadInvs[0], Fr.one)
        let tamperedSP = GPUSubtableMemoryProof(
            chunkIndex: sp0.chunkIndex,
            readCounts: sp0.readCounts,
            beta: sp0.beta,
            readInverses: badReadInvs,
            tableTerms: sp0.tableTerms,
            readSum: sp0.readSum,
            tableSum: sp0.tableSum
        )
        var tamperedProofs = validProof.subtableProofs
        tamperedProofs[0] = tamperedSP
        let tamperedProof = GPULassoProof(
            numChunks: validProof.numChunks,
            subtableProofs: tamperedProofs,
            indices: validProof.indices,
            memoryCheckPassed: validProof.memoryCheckPassed
        )
        let tamperedResult = engine.verify(proof: tamperedProof, lookups: validLookups, table: table)
        expect(!tamperedResult, "Tampered read inverse is rejected")

        // Tamper: wrong read counts
        let badCountsSP = GPUSubtableMemoryProof(
            chunkIndex: sp0.chunkIndex,
            readCounts: [Fr](repeating: Fr.one, count: sp0.readCounts.count),
            beta: sp0.beta,
            readInverses: sp0.readInverses,
            tableTerms: sp0.tableTerms,
            readSum: sp0.readSum,
            tableSum: sp0.tableSum
        )
        var tamperedProofs2 = validProof.subtableProofs
        tamperedProofs2[0] = badCountsSP
        let tamperedProof2 = GPULassoProof(
            numChunks: validProof.numChunks,
            subtableProofs: tamperedProofs2,
            indices: validProof.indices,
            memoryCheckPassed: validProof.memoryCheckPassed
        )
        let tamperedResult2 = engine.verify(proof: tamperedProof2, lookups: validLookups, table: table)
        expect(!tamperedResult2, "Wrong read counts rejected")
    } catch {
        expect(false, "Invalid lookup detection threw: \(error)")
    }
}

// MARK: - Structured Table (Bitwise XOR)

func testLassoStructuredTable() {
    do {
        let engine = try GPULassoEngine()

        // XOR table for 8-bit values
        let table = GPULassoTable.build(.bitwiseXor(bits: 8))

        expect(table.numChunks == 1, "8-bit XOR has 1 chunk (8 bits per chunk)")
        expect(table.subtables[0].count == 256 * 256, "XOR subtable has 256*256 entries")

        // Lookups: index into the XOR table as a*256+b for XOR(a,b)
        // For XOR(5,3) = 6: index = 5*256 + 3 = 1283
        // The value at index 1283 in the subtable should be frFromInt(6)
        let xorVal = table.subtables[0][5 * 256 + 3]
        expect(frEqual(xorVal, frFromInt(6)), "XOR(5,3) = 6 in subtable")

        // AND table for 8-bit values
        let andTable = GPULassoTable.build(.bitwiseAnd(bits: 8))
        let andVal = andTable.subtables[0][0xFF * 256 + 0x0F]
        expect(frEqual(andVal, frFromInt(0x0F)), "AND(0xFF, 0x0F) = 0x0F")

        // Prove and verify with range table (simpler to test end-to-end)
        let rangeTable = GPULassoTable.build(.range(bits: 8, chunks: 2))
        let lookups: [Fr] = (0..<16).map { frFromInt(UInt64($0 * 17)) } // 0, 17, 34, ..., 255
        let proof = try engine.prove(lookups: lookups, table: rangeTable)
        expect(proof.memoryCheckPassed, "Structured range table: memory check passed")

        let valid = engine.verify(proof: proof, lookups: lookups, table: rangeTable)
        expect(valid, "Structured range table: verification passed")

        // Test multilinear extension evaluation
        // Simple test: MLE of [1, 2, 3, 4] at point [0, 0] should give evals[0] = 1
        let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let mle00 = engine.evaluateMultilinearExtension(evals: evals, point: [Fr.zero, Fr.zero])
        expect(frEqual(mle00, frFromInt(1)), "MLE([1,2,3,4], [0,0]) = 1")

        // MLE at [1, 0] = evals[1] = 2
        let mle10 = engine.evaluateMultilinearExtension(evals: evals, point: [Fr.one, Fr.zero])
        expect(frEqual(mle10, frFromInt(2)), "MLE([1,2,3,4], [1,0]) = 2")

        // MLE at [0, 1] = evals[2] = 3
        let mle01 = engine.evaluateMultilinearExtension(evals: evals, point: [Fr.zero, Fr.one])
        expect(frEqual(mle01, frFromInt(3)), "MLE([1,2,3,4], [0,1]) = 3")

        // MLE at [1, 1] = evals[3] = 4
        let mle11 = engine.evaluateMultilinearExtension(evals: evals, point: [Fr.one, Fr.one])
        expect(frEqual(mle11, frFromInt(4)), "MLE([1,2,3,4], [1,1]) = 4")
    } catch {
        expect(false, "Structured table test threw: \(error)")
    }
}
