// GPU CSR Sparse Matvec Engine Tests
//
// Tests for GPUSparseMatvecEngine including:
// - Single matvec GPU vs CPU correctness
// - Fused triple matvec GPU vs CPU correctness
// - Batch matvec GPU vs CPU correctness
// - Edge cases (empty rows, single element, all zeros)

import Foundation
import Metal
import NeonFieldOps
import zkMetal

// MARK: - Test Runner

func runGPUSparseMatvecEngineTests() {
    print("=== GPUSparseMatvecEngine Tests ===")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    var allPassed = true

    do {
        let engine = try GPUSparseMatvecEngine()

        // Test single matvec
        if !testSingleMatvec(engine: engine) { allPassed = false }

        // Test fused triple matvec
        if !testFusedTripleMatvec(engine: engine) { allPassed = false }

        // Test batch matvec
        if !testBatchMatvec(engine: engine) { allPassed = false }

        // Test edge cases
        if !testEdgeCases(engine: engine) { allPassed = false }

        if allPassed {
            print("\n✓ All GPUSparseMatvecEngine tests passed!")
        } else {
            print("\n✗ Some GPUSparseMatvecEngine tests FAILED!")
        }
    } catch {
        print("Failed to create GPUSparseMatvecEngine: \(error)")
    }
}

// MARK: - Helper Functions

/// Generate deterministic Fr from index
func testFr(_ seed: UInt64) -> Fr {
    return frFromInt(seed &+ 1)
}

/// Reference CPU matvec for verification
func sparseMatvecRef(
    rowPtr: [UInt32],
    colIdx: [UInt32],
    values: [Fr],
    z: [Fr]
) -> [Fr] {
    let m = Int(rowPtr.count) - 1
    var result = [Fr](repeating: .zero, count: m)

    for i in 0..<m {
        var acc: Fr = .zero
        for k in Int(rowPtr[i])..<Int(rowPtr[i + 1]) {
            let col = Int(colIdx[k])
            acc = frAdd(acc, frMul(values[k], z[col]))
        }
        result[i] = acc
    }
    return result
}

/// Reference CPU triple matvec for verification
func sparseTripleMatvecRef(
    rowPtr: [UInt32],
    colIdx: [UInt32],
    valuesA: [Fr],
    valuesB: [Fr],
    valuesC: [Fr],
    z: [Fr]
) -> (az: [Fr], bz: [Fr], cz: [Fr]) {
    let m = Int(rowPtr.count) - 1
    var az = [Fr](repeating: .zero, count: m)
    var bz = [Fr](repeating: .zero, count: m)
    var cz = [Fr](repeating: .zero, count: m)

    for i in 0..<m {
        var accA: Fr = .zero
        var accB: Fr = .zero
        var accC: Fr = .zero
        for k in Int(rowPtr[i])..<Int(rowPtr[i + 1]) {
            let col = Int(colIdx[k])
            let zval = z[col]
            accA = frAdd(accA, frMul(valuesA[k], zval))
            accB = frAdd(accB, frMul(valuesB[k], zval))
            accC = frAdd(accC, frMul(valuesC[k], zval))
        }
        az[i] = accA
        bz[i] = accB
        cz[i] = accC
    }
    return (az, bz, cz)
}

/// Generate a deterministic CSR matrix for testing
func generateTestCSR(
    rows: Int,
    cols: Int,
    sparsity: Double  // 0.0 to 1.0, fraction of non-zeros
) -> (rowPtr: [UInt32], colIdx: [UInt32], values: [Fr]) {
    var rowPtr = [UInt32]()
    var colIdx = [UInt32]()
    var values = [Fr]()

    rowPtr.append(0)

    for i in 0..<rows {
        for j in 0..<cols {
            // Use hash-like pattern for determinism
            let hash = UInt64(i) * 1000003 ^ UInt64(j) * 10007
            let isNonZero = (Double(hash & 0xFFFF) / 65536.0) < sparsity
            if isNonZero {
                colIdx.append(UInt32(j))
                values.append(testFr(hash))
            }
        }
        rowPtr.append(UInt32(colIdx.count))
    }

    return (rowPtr, colIdx, values)
}

/// Generate deterministic vector
func generateTestVector(length: Int) -> [Fr] {
    var vec = [Fr](repeating: .zero, count: length)
    for i in 0..<length {
        vec[i] = testFr(UInt64(i) * 1000003)
    }
    return vec
}

/// Compare two Fr arrays
func sparseCompareFrArrays(_ a: [Fr], _ b: [Fr]) -> Bool {
    if a.count != b.count { return false }
    for i in 0..<a.count {
        if !frEq(a[i], b[i]) {
            return false
        }
    }
    return true
}

// MARK: - Test Cases

@discardableResult
func testSingleMatvec(engine: GPUSparseMatvecEngine) -> Bool {
    print("\n--- testSingleMatvec ---")

    var allPassed = true

    // Test sizes: small (CPU path), medium (GPU path), large (GPU path)
    let testCases = [
        (rows: 32, cols: 64, sparsity: 0.1),   // Small - CPU path
        (rows: 128, cols: 256, sparsity: 0.05), // Medium - GPU path
        (rows: 512, cols: 512, sparsity: 0.02), // Large - GPU path
        (rows: 1024, cols: 1024, sparsity: 0.01), // Very large - GPU path
    ]

    for (rows, cols, sparsity) in testCases {
        print("  Testing \(rows)x\(cols) matrix with sparsity \(String(format: "%.2f", sparsity))...")

        let (rowPtr, colIdx, values) = generateTestCSR(rows: rows, cols: cols, sparsity: sparsity)
        let z = generateTestVector(length: cols)
        let m = rows

        // Reference result
        let expected = sparseMatvecRef(rowPtr: rowPtr, colIdx: colIdx, values: values, z: z)

        // GPU result
        let actual = engine.matvec(rowPtr: rowPtr, colIdx: colIdx, values: values, z: z, m: m)

        // Compare
        if sparseCompareFrArrays(expected, actual) {
            print("    ✓ PASS")
        } else {
            print("    ✗ FAIL")
            print("      Expected: \(expected.prefix(3))")
            print("      Actual:   \(actual.prefix(3))")
            allPassed = false
        }
    }

    return allPassed
}

@discardableResult
func testFusedTripleMatvec(engine: GPUSparseMatvecEngine) -> Bool {
    print("\n--- testFusedTripleMatvec ---")

    var allPassed = true

    let testCases = [
        (rows: 64, cols: 128, sparsity: 0.05),
        (rows: 256, cols: 256, sparsity: 0.02),
        (rows: 512, cols: 512, sparsity: 0.01),
    ]

    for (rows, cols, sparsity) in testCases {
        print("  Testing \(rows)x\(cols) triple matvec...")

        let (rowPtr, colIdx, valuesA) = generateTestCSR(rows: rows, cols: cols, sparsity: sparsity)
        let (_, _, valuesB) = generateTestCSR(rows: rows, cols: cols, sparsity: sparsity)
        let (_, _, valuesC) = generateTestCSR(rows: rows, cols: cols, sparsity: sparsity)
        let z = generateTestVector(length: cols)
        let m = rows

        // Reference result
        let expected = sparseTripleMatvecRef(
            rowPtr: rowPtr, colIdx: colIdx,
            valuesA: valuesA, valuesB: valuesB, valuesC: valuesC,
            z: z
        )

        // GPU result
        let actual = engine.matvecTriple(
            rowPtr: rowPtr, colIdx: colIdx,
            valuesA: valuesA, valuesB: valuesB, valuesC: valuesC,
            z: z, m: m
        )

        // Compare
        let azMatch = sparseCompareFrArrays(expected.az, actual.az)
        let bzMatch = sparseCompareFrArrays(expected.bz, actual.bz)
        let czMatch = sparseCompareFrArrays(expected.cz, actual.cz)

        if azMatch && bzMatch && czMatch {
            print("    ✓ PASS")
        } else {
            print("    ✗ FAIL")
            if !azMatch {
                print("      A*z mismatch")
            }
            if !bzMatch {
                print("      B*z mismatch")
            }
            if !czMatch {
                print("      C*z mismatch")
            }
            allPassed = false
        }
    }

    return allPassed
}

@discardableResult
func testBatchMatvec(engine: GPUSparseMatvecEngine) -> Bool {
    print("\n--- testBatchMatvec ---")

    var allPassed = true

    let rows = 256
    let cols = 256
    let k = 4  // Number of vectors in batch
    let sparsity = 0.02

    print("  Testing batch matvec with k=\(k) vectors...")

    let (rowPtr, colIdx, values) = generateTestCSR(rows: rows, cols: cols, sparsity: sparsity)

    // Generate k random vectors
    var zVectors = [Fr]()
    for i in 0..<k {
        let vec = generateTestVector(length: cols)
        zVectors.append(contentsOf: vec)
    }

    // Reference results
    var expected = [[Fr]]()
    for i in 0..<k {
        let z = Array(zVectors[i * cols..<(i + 1) * cols])
        expected.append(sparseMatvecRef(rowPtr: rowPtr, colIdx: colIdx, values: values, z: z))
    }

    // GPU result
    let actual = engine.matvecBatch(
        rowPtr: rowPtr, colIdx: colIdx, values: values,
        zVectors: zVectors, m: rows, n: cols, k: k
    )

    // Compare
    for i in 0..<k {
        if sparseCompareFrArrays(expected[i], actual[i]) {
            print("    ✓ Batch \(i) PASS")
        } else {
            print("    ✗ Batch \(i) FAIL")
            allPassed = false
        }
    }

    return allPassed
}

@discardableResult
func testEdgeCases(engine: GPUSparseMatvecEngine) -> Bool {
    print("\n--- testEdgeCases ---")

    var allPassed = true

    // Test 1: Matrix with some empty rows
    print("  Testing matrix with empty rows...")
    let emptyRowRowPtr: [UInt32] = [0, 2, 2, 5, 5]  // Row 1 and 3 are empty
    let emptyRowColIdx: [UInt32] = [0, 1, 0, 1, 2]
    var emptyRowValues = [Fr]()
    for i in 0..<5 {
        emptyRowValues.append(testFr(UInt64(i) + 1))
    }
    let emptyRowZ = generateTestVector(length: 3)

    let expected = sparseMatvecRef(rowPtr: emptyRowRowPtr, colIdx: emptyRowColIdx, values: emptyRowValues, z: emptyRowZ)
    let actual = engine.matvec(rowPtr: emptyRowRowPtr, colIdx: emptyRowColIdx, values: emptyRowValues, z: emptyRowZ, m: 4)

    if sparseCompareFrArrays(expected, actual) {
        print("    ✓ PASS (empty rows)")
    } else {
        print("    ✗ FAIL (empty rows)")
        allPassed = false
    }

    // Test 2: Single row, single non-zero
    print("  Testing 1x1 matrix...")
    let singleRowPtr: [UInt32] = [0, 1]
    let singleColIdx: [UInt32] = [0]
    let singleValue = [testFr(7)]
    let singleZ = generateTestVector(length: 1)

    let singleExpected = sparseMatvecRef(rowPtr: singleRowPtr, colIdx: singleColIdx, values: singleValue, z: singleZ)
    let singleActual = engine.matvec(rowPtr: singleRowPtr, colIdx: singleColIdx, values: singleValue, z: singleZ, m: 1)

    if sparseCompareFrArrays(singleExpected, singleActual) {
        print("    ✓ PASS (1x1)")
    } else {
        print("    ✗ FAIL (1x1)")
        allPassed = false
    }

    // Test 3: Dense matrix (high sparsity)
    print("  Testing dense matrix (sparsity 0.5)...")
    let (denseRowPtr, denseColIdx, denseValues) = generateTestCSR(rows: 64, cols: 64, sparsity: 0.5)
    let denseZ = generateTestVector(length: 64)

    let denseExpected = sparseMatvecRef(rowPtr: denseRowPtr, colIdx: denseColIdx, values: denseValues, z: denseZ)
    let denseActual = engine.matvec(rowPtr: denseRowPtr, colIdx: denseColIdx, values: denseValues, z: denseZ, m: 64)

    if sparseCompareFrArrays(denseExpected, denseActual) {
        print("    ✓ PASS (dense matrix)")
    } else {
        print("    ✗ FAIL (dense matrix)")
        allPassed = false
    }

    return allPassed
}

// MARK: - Benchmark

func runGPUSparseMatvecBenchmarks() {
    print("\n=== GPUSparseMatvecEngine Benchmarks ===")

    guard let engine = try? GPUSparseMatvecEngine() else {
        print("Failed to create GPUSparseMatvecEngine")
        return
    }

    let sizes = [
        (64, 64, 0.05),
        (128, 128, 0.02),
        (256, 256, 0.01),
        (512, 512, 0.005),
    ]

    for (rows, cols, sparsity) in sizes {
        let (rowPtr, colIdx, values) = generateTestCSR(rows: rows, cols: cols, sparsity: sparsity)
        let z = generateTestVector(length: cols)
        let m = rows

        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 100
        for _ in 0..<iterations {
            _ = engine.matvec(rowPtr: rowPtr, colIdx: colIdx, values: values, z: z, m: m)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let avgMs = (elapsed / Double(iterations)) * 1000
        let nnz = values.count
        print("  \(rows)x\(cols), \(nnz) nnz: \(String(format: "%.3f", avgMs)) ms/op")
    }

    // Benchmark fused triple
    print("\n--- Triple Matvec Benchmark ---")
    for (rows, cols, sparsity) in sizes {
        let (rowPtr, colIdx, valuesA) = generateTestCSR(rows: rows, cols: cols, sparsity: sparsity)
        let (_, _, valuesB) = generateTestCSR(rows: rows, cols: cols, sparsity: sparsity)
        let (_, _, valuesC) = generateTestCSR(rows: rows, cols: cols, sparsity: sparsity)
        let z = generateTestVector(length: cols)
        let m = rows

        let start = CFAbsoluteTimeGetCurrent()
        let iterations = 100
        for _ in 0..<iterations {
            _ = engine.matvecTriple(rowPtr: rowPtr, colIdx: colIdx,
                                   valuesA: valuesA, valuesB: valuesB, valuesC: valuesC,
                                   z: z, m: m)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let avgMs = (elapsed / Double(iterations)) * 1000
        print("  \(rows)x\(cols) triple: \(String(format: "%.3f", avgMs)) ms/op")
    }
}
