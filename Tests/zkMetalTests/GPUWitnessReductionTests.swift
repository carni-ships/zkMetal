// GPUWitnessReductionTests — Tests for GPU-accelerated witness reduction engine
//
// Tests column compression, sparse detection, commitment, batch processing,
// streaming generation, R1CS validation, and reconstruction.

import Foundation
import zkMetal

// MARK: - Test: Column Compression — Independent Columns

private func testColumnCompressionIndependent() {
    suite("WitnessReduction: Independent Columns")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        // 3 rows x 3 columns, all independent
        // Col 0: [1, 2, 3], Col 1: [4, 5, 6], Col 2: [7, 8, 10]
        let witness = [
            frFromInt(1), frFromInt(4), frFromInt(7),
            frFromInt(2), frFromInt(5), frFromInt(8),
            frFromInt(3), frFromInt(6), frFromInt(10),
        ]

        let result = engine.compressColumns(witness: witness, numRows: 3, numCols: 3)
        expect(result.independentColumns.count == 3,
               "all 3 columns independent (got \(result.independentColumns.count))")
        expect(result.dependentColumns.isEmpty,
               "no dependent columns (got \(result.dependentColumns.count))")
        expect(result.compressionRatio == 1.0,
               "compression ratio = 1.0 (no compression)")
    }
}

// MARK: - Test: Column Compression — Dependent Column

private func testColumnCompressionDependent() {
    suite("WitnessReduction: Dependent Column Detection")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        // 3 rows x 3 columns
        // Col 0: [2, 4, 6]
        // Col 1: [6, 12, 18] = 3 * Col 0
        // Col 2: [1, 3, 5] — independent
        let two = frFromInt(2)
        let four = frFromInt(4)
        let six = frFromInt(6)
        let twelve = frFromInt(12)
        let eighteen = frFromInt(18)
        let one = frFromInt(1)
        let three = frFromInt(3)
        let five = frFromInt(5)

        let witness = [
            two, six, one,
            four, twelve, three,
            six, eighteen, five,
        ]

        let result = engine.compressColumns(witness: witness, numRows: 3, numCols: 3)
        expect(result.independentColumns.count == 2,
               "2 independent columns (got \(result.independentColumns.count))")
        expect(result.dependentColumns.count == 1,
               "1 dependent column (got \(result.dependentColumns.count))")
        expect(result.dependentColumns.contains(1),
               "column 1 is dependent")

        // Check the scalar: col1 = 3 * col0
        if let dep = result.dependencies.first {
            expect(dep.dependentCol == 1, "dependent col is 1")
            expect(dep.sourceCol == 0, "source col is 0")
            let expectedScalar = frFromInt(3)
            expect(dep.scalar == expectedScalar,
                   "scalar = 3")
        } else {
            expect(false, "missing dependency info")
        }
    }
}

// MARK: - Test: Column Compression — Zero Column

private func testColumnCompressionZero() {
    suite("WitnessReduction: Zero Column Elimination")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        // 2 rows x 3 columns, col 1 is all zero
        let witness = [
            frFromInt(5), Fr.zero, frFromInt(9),
            frFromInt(7), Fr.zero, frFromInt(11),
        ]

        let result = engine.compressColumns(witness: witness, numRows: 2, numCols: 3)
        expect(result.dependentColumns.contains(1),
               "zero column 1 detected as dependent")
        expect(result.independentColumns.count == 2,
               "2 independent columns remain")
    }
}

// MARK: - Test: Sparse Detection

private func testSparseDetection() {
    suite("WitnessReduction: Sparse Detection")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        // 4 rows x 3 columns
        // Col 0: [1, 0, 0, 0] — 75% sparse
        // Col 1: [1, 2, 3, 4] — 0% sparse
        // Col 2: [0, 0, 5, 0] — 75% sparse
        let witness = [
            frFromInt(1), frFromInt(1), Fr.zero,
            Fr.zero, frFromInt(2), Fr.zero,
            Fr.zero, frFromInt(3), frFromInt(5),
            Fr.zero, frFromInt(4), Fr.zero,
        ]

        // Threshold 0.5: should detect cols 0 and 2 as sparse
        let sparse = engine.detectSparse(witness: witness, numRows: 4, numCols: 3,
                                         threshold: 0.5)
        expect(sparse.count == 2, "2 sparse columns detected (got \(sparse.count))")

        if sparse.count >= 1 {
            let col0 = sparse[0]
            expect(col0.sparsity == 0.75, "col 0 sparsity = 0.75 (got \(col0.sparsity))")
            expect(col0.indices.count == 1, "col 0 has 1 non-zero entry")
            expect(col0.indices[0] == 0, "col 0 non-zero at row 0")
        }

        if sparse.count >= 2 {
            let col2 = sparse[1]
            expect(col2.sparsity == 0.75, "col 2 sparsity = 0.75")
            expect(col2.indices.count == 1, "col 2 has 1 non-zero entry")
            expect(col2.indices[0] == 2, "col 2 non-zero at row 2")
        }

        // Threshold 0.9: none should qualify
        let sparseHigh = engine.detectSparse(witness: witness, numRows: 4, numCols: 3,
                                             threshold: 0.9)
        expect(sparseHigh.isEmpty,
               "no columns at 90% sparsity threshold (got \(sparseHigh.count))")
    }
}

// MARK: - Test: Sparse Detection — All Dense

private func testSparseDetectionDense() {
    suite("WitnessReduction: Dense Witness (No Sparse)")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        // 2x2, all non-zero
        let witness = [
            frFromInt(1), frFromInt(2),
            frFromInt(3), frFromInt(4),
        ]

        let sparse = engine.detectSparse(witness: witness, numRows: 2, numCols: 2,
                                         threshold: 0.5)
        expect(sparse.isEmpty, "fully dense witness has no sparse columns")
    }
}

// MARK: - Test: Witness Commitment — Poseidon2

private func testCommitmentPoseidon2() {
    suite("WitnessReduction: Poseidon2 Commitment")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        let witness = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let commitment = engine.commitWitness(witness, method: "poseidon2")

        expect(commitment.witnessSize == 3, "committed 3 elements")
        expect(commitment.method == "poseidon2", "method is poseidon2")
        expect(commitment.value != Fr.zero, "commitment is non-zero")

        // Determinism: same input => same commitment
        let commitment2 = engine.commitWitness(witness, method: "poseidon2")
        expect(commitment.value == commitment2.value, "commitment is deterministic")

        // Different input => different commitment
        let witness2 = [frFromInt(1), frFromInt(2), frFromInt(4)]
        let commitment3 = engine.commitWitness(witness2, method: "poseidon2")
        expect(commitment.value != commitment3.value, "different input => different commitment")
    }
}

// MARK: - Test: Witness Commitment — Pedersen

private func testCommitmentPedersen() {
    suite("WitnessReduction: Pedersen Commitment")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        let witness = [frFromInt(10), frFromInt(20), frFromInt(30)]
        let commitment = engine.commitWitness(witness, method: "pedersen")

        expect(commitment.witnessSize == 3, "committed 3 elements")
        expect(commitment.method == "pedersen", "method is pedersen")
        expect(commitment.value != Fr.zero, "pedersen commitment is non-zero")

        // Determinism
        let commitment2 = engine.commitWitness(witness, method: "pedersen")
        expect(commitment.value == commitment2.value, "pedersen commitment is deterministic")
    }
}

// MARK: - Test: Witness Commitment — Empty

private func testCommitmentEmpty() {
    suite("WitnessReduction: Empty Commitment")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        let commitment = engine.commitWitness([], method: "poseidon2")
        expect(commitment.value == Fr.zero, "empty witness commits to zero")
        expect(commitment.witnessSize == 0, "size = 0")
    }
}

// MARK: - Test: R1CS Validation — Satisfying Witness

private func testR1CSValidationPass() {
    suite("WitnessReduction: R1CS Validation (pass)")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        // Simple R1CS: w0 * w1 = w2
        // Witness: [3, 5, 15]
        let witness = [frFromInt(3), frFromInt(5), frFromInt(15)]

        let constraints = [
            WRConstraint(
                a: [(0, Fr.one)],   // A = [1, 0, 0]
                b: [(1, Fr.one)],   // B = [0, 1, 0]
                c: [(2, Fr.one)]    // C = [0, 0, 1]
            )
        ]

        let result = engine.validateR1CS(witness: witness, constraints: constraints)
        expect(result.valid, "witness [3, 5, 15] satisfies w0*w1=w2")
        expect(result.firstFailure == -1, "no failure index")
        expect(result.numConstraints == 1, "1 constraint checked")
    }
}

// MARK: - Test: R1CS Validation — Failing Witness

private func testR1CSValidationFail() {
    suite("WitnessReduction: R1CS Validation (fail)")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        // w0 * w1 = w2, but witness has wrong product
        let witness = [frFromInt(3), frFromInt(5), frFromInt(16)]

        let constraints = [
            WRConstraint(
                a: [(0, Fr.one)],
                b: [(1, Fr.one)],
                c: [(2, Fr.one)]
            )
        ]

        let result = engine.validateR1CS(witness: witness, constraints: constraints)
        expect(!result.valid, "witness [3, 5, 16] fails w0*w1=w2")
        expect(result.firstFailure == 0, "failure at constraint 0")
    }
}

// MARK: - Test: R1CS Validation — Multiple Constraints

private func testR1CSValidationMultiple() {
    suite("WitnessReduction: R1CS Multiple Constraints")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        // w0 * w1 = w2 (3 * 5 = 15)
        // w2 * w0 = w3 (15 * 3 = 45)
        let witness = [frFromInt(3), frFromInt(5), frFromInt(15), frFromInt(45)]

        let constraints = [
            WRConstraint(
                a: [(0, Fr.one)], b: [(1, Fr.one)], c: [(2, Fr.one)]
            ),
            WRConstraint(
                a: [(2, Fr.one)], b: [(0, Fr.one)], c: [(3, Fr.one)]
            ),
        ]

        let result = engine.validateR1CS(witness: witness, constraints: constraints)
        expect(result.valid, "both constraints satisfied")
        expect(result.numConstraints == 2, "2 constraints")
    }
}

// MARK: - Test: R1CS Validation — Sparse Coefficients

private func testR1CSSparseCoefficients() {
    suite("WitnessReduction: R1CS Sparse Coefficients")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        // Constraint: (2*w0 + 3*w1) * (w2) = (6*w0 + 9*w1)
        // witness: w0=1, w1=2, w2=3
        // A.w = 2*1 + 3*2 = 8
        // B.w = 3
        // C.w = 6*1 + 9*2 = 24
        // Check: 8 * 3 = 24 => pass
        let two = frFromInt(2)
        let three = frFromInt(3)
        let sixFr = frFromInt(6)
        let nine = frFromInt(9)

        let witness = [frFromInt(1), frFromInt(2), frFromInt(3)]

        let constraints = [
            WRConstraint(
                a: [(0, two), (1, three)],
                b: [(2, Fr.one)],
                c: [(0, sixFr), (1, nine)]
            )
        ]

        let result = engine.validateR1CS(witness: witness, constraints: constraints)
        expect(result.valid, "sparse coefficient R1CS passes")
    }
}

// MARK: - Test: Batch Processing

private func testBatchProcessing() {
    suite("WitnessReduction: Batch Processing")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        let w1 = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let w2 = [frFromInt(10), frFromInt(20), Fr.zero, Fr.zero]
        let w3 = [frFromInt(7)]

        let result = engine.batchProcess(witnesses: [w1, w2, w3])

        expect(result.witnesses.count == 3, "3 witnesses processed")
        expect(result.commitments.count == 3, "3 commitments computed")
        expect(result.totalElements == 8, "total elements = 3 + 4 + 1 = 8")

        // w2 had trailing zeros, should be trimmed
        expect(result.witnesses[1].count <= 4,
               "trailing zeros trimmed from w2")

        // Each commitment should be non-zero (non-trivial witnesses)
        for (i, c) in result.commitments.enumerated() {
            expect(c.value != Fr.zero, "commitment \(i) is non-zero")
        }
    }
}

// MARK: - Test: Streaming Witness Generation

private func testStreamingWitness() {
    suite("WitnessReduction: Streaming Generation")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        // Stream witness in 3 chunks
        var state = engine.streamingBegin()
        expect(state.chunksProcessed == 0, "initial: 0 chunks")
        expect(state.totalElements == 0, "initial: 0 elements")

        let chunk1 = [frFromInt(1), frFromInt(2)]
        let chunk2 = [frFromInt(3), frFromInt(4)]
        let chunk3 = [frFromInt(5)]

        engine.streamingIngest(state: &state, chunk: chunk1)
        expect(state.chunksProcessed == 1, "1 chunk after first ingest")
        expect(state.totalElements == 2, "2 elements after first ingest")

        engine.streamingIngest(state: &state, chunk: chunk2)
        expect(state.chunksProcessed == 2, "2 chunks")
        expect(state.totalElements == 4, "4 elements")

        engine.streamingIngest(state: &state, chunk: chunk3)
        expect(state.chunksProcessed == 3, "3 chunks")
        expect(state.totalElements == 5, "5 elements")

        let (witness, commitment) = engine.streamingFinalize(state: state)
        expect(witness.count == 5, "final witness has 5 elements")
        expect(commitment.witnessSize == 5, "commitment covers 5 elements")
        expect(commitment.value != Fr.zero, "streaming commitment is non-zero")

        // Verify streaming commitment matches one-shot commitment
        let oneshotCommitment = engine.commitWitness(witness)
        expect(commitment.value == oneshotCommitment.value,
               "streaming commitment matches one-shot")
    }
}

// MARK: - Test: Reconstruct from Compressed

private func testReconstruction() {
    suite("WitnessReduction: Reconstruction from Compressed")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        // Original: 2 rows x 3 cols
        // Col 0: [2, 4], Col 1: [6, 12] = 3*Col0, Col 2: [1, 3]
        let original = [
            frFromInt(2), frFromInt(6), frFromInt(1),
            frFromInt(4), frFromInt(12), frFromInt(3),
        ]

        let compression = engine.compressColumns(witness: original, numRows: 2, numCols: 3)
        expect(compression.independentColumns.count == 2, "2 independent cols")
        expect(compression.dependentColumns.count == 1, "1 dependent col")

        // Build compressed witness (only independent columns)
        var compressed = [Fr]()
        for row in 0..<2 {
            for col in compression.independentColumns {
                compressed.append(original[row * 3 + col])
            }
        }

        // Reconstruct
        let reconstructed = engine.reconstruct(
            compressed: compressed, compression: compression,
            numRows: 2, numCols: 3)

        // Verify matches original
        var allMatch = true
        for i in 0..<original.count {
            if original[i] != reconstructed[i] {
                allMatch = false
                print("  [DETAIL] Mismatch at index \(i)")
            }
        }
        expect(allMatch, "reconstructed witness matches original")
    }
}

// MARK: - Test: GPU Batch Multiply

private func testGPUBatchMultiply() {
    suite("WitnessReduction: Batch Multiply")

    do {
        let engine = try GPUWitnessReductionEngine(forceGPU: false)

        let a = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(0)]
        let b = [frFromInt(5), frFromInt(13), frFromInt(2), frFromInt(99)]

        let result = try engine.gpuBatchMultiply(a: a, b: b)

        expect(result.count == 4, "4 results")
        expect(result[0] == frFromInt(15), "3*5 = 15")
        expect(result[1] == frFromInt(91), "7*13 = 91")
        expect(result[2] == frFromInt(22), "11*2 = 22")
        expect(result[3] == Fr.zero, "0*99 = 0")
    } catch {
        expect(false, "batch multiply threw: \(error)")
    }
}

// MARK: - Test: GPU Batch Add

private func testGPUBatchAdd() {
    suite("WitnessReduction: Batch Add")

    do {
        let engine = try GPUWitnessReductionEngine(forceGPU: false)

        let a = [frFromInt(10), frFromInt(20), Fr.zero]
        let b = [frFromInt(5), frFromInt(30), frFromInt(42)]

        let result = try engine.gpuBatchAdd(a: a, b: b)

        expect(result.count == 3, "3 results")
        expect(result[0] == frFromInt(15), "10+5 = 15")
        expect(result[1] == frFromInt(50), "20+30 = 50")
        expect(result[2] == frFromInt(42), "0+42 = 42")
    } catch {
        expect(false, "batch add threw: \(error)")
    }
}

// MARK: - Test: R1CS Empty Constraints

private func testR1CSEmpty() {
    suite("WitnessReduction: R1CS Empty Constraints")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        let witness = [frFromInt(1)]
        let result = engine.validateR1CS(witness: witness, constraints: [])
        expect(result.valid, "empty constraints always valid")
        expect(result.numConstraints == 0, "0 constraints")
    }
}

// MARK: - Test: Edge Cases

private func testEdgeCases() {
    suite("WitnessReduction: Edge Cases")

    do {
        let engine = GPUWitnessReductionEngine(cpuOnly: true)

        // Empty witness compression
        let emptyResult = engine.compressColumns(witness: [], numRows: 0, numCols: 0)
        expect(emptyResult.independentColumns.isEmpty, "empty witness: no columns")

        // Single element witness
        let single = [frFromInt(42)]
        let singleResult = engine.compressColumns(witness: single, numRows: 1, numCols: 1)
        expect(singleResult.independentColumns.count == 1, "single element is independent")
        expect(singleResult.dependentColumns.isEmpty, "no dependent in single-element")

        // Single element sparse detection
        let sparse = engine.detectSparse(witness: single, numRows: 1, numCols: 1, threshold: 0.5)
        expect(sparse.isEmpty, "single non-zero element is not sparse at 50%")

        // Batch with empty input
        let batchEmpty = engine.batchProcess(witnesses: [])
        expect(batchEmpty.witnesses.isEmpty, "empty batch returns empty")
        expect(batchEmpty.totalElements == 0, "empty batch has 0 elements")
    }
}

// MARK: - Public Entry Point

public func runGPUWitnessReductionTests() {
    testColumnCompressionIndependent()
    testColumnCompressionDependent()
    testColumnCompressionZero()
    testSparseDetection()
    testSparseDetectionDense()
    testCommitmentPoseidon2()
    testCommitmentPedersen()
    testCommitmentEmpty()
    testR1CSValidationPass()
    testR1CSValidationFail()
    testR1CSValidationMultiple()
    testR1CSSparseCoefficients()
    testBatchProcessing()
    testStreamingWitness()
    testReconstruction()
    testGPUBatchMultiply()
    testGPUBatchAdd()
    testR1CSEmpty()
    testEdgeCases()
}
