// Tensor Proof Compression Benchmark
// Measures compression from N evaluations to sqrt(N) proof elements
// and compares with direct opening and Basefold.

import zkMetal
import Foundation

public func runTensorBench() {
    print("=== Tensor Proof Compression Benchmark ===")

    var rng: UInt64 = 0xDEAD_BEEF_C0DE_0055
    func nextRng() -> UInt64 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        return rng >> 32
    }

    // --- Correctness Tests ---
    print("\n--- Correctness verification ---")

    // Test 1: Tensor product
    do {
        let point: [Fr] = [frFromInt(3), frFromInt(5)]
        let tp = TensorCompressor.tensorProduct(point)
        var tpSum = Fr.zero
        for t in tp { tpSum = frAdd(tpSum, t) }
        let sumCorrect = frToInt(tpSum) == frToInt(Fr.one)
        print("  Tensor product sum=1: \(sumCorrect ? "PASS" : "FAIL")")
    }

    // Test 2: eq polynomial
    do {
        let point: [Fr] = [frFromInt(7), frFromInt(11)]
        let eq = TensorCompressor.eqPolynomial(point)
        var eqSum = Fr.zero
        for e in eq { eqSum = frAdd(eqSum, e) }
        let eqSumCorrect = frToInt(eqSum) == frToInt(Fr.one)
        print("  eq polynomial sum=1: \(eqSumCorrect ? "PASS" : "FAIL")")
    }

    // Test 3: Compress + verify round-trip
    do {
        let numVars = 6
        let n = 1 << numVars
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            evals[i] = frFromInt(nextRng())
        }
        var point = [Fr]()
        for _ in 0..<numVars {
            point.append(frFromInt(nextRng()))
        }

        let expectedValue = TensorCompressor.multilinearEval(evals: evals, point: point)

        let halfVars = numVars / 2
        let sqrtN = 1 << halfVars
        let tL = TensorCompressor.tensorProduct(Array(point[0..<halfVars]))
        let tR = TensorCompressor.tensorProduct(Array(point[halfVars..<numVars]))
        let v = TensorCompressor.matVecMul(evaluations: evals, vec: tR, rows: sqrtN, cols: sqrtN)
        let tensorValue = TensorCompressor.dotProduct(tL, v)
        let tensorCorrect = frToInt(tensorValue) == frToInt(expectedValue)
        print("  Tensor decomposition f(r) = tL^T M tR: \(tensorCorrect ? "PASS" : "FAIL")")

        let proof = TensorCompressor.compress(evaluations: evals, point: point, value: expectedValue)
        let verified = TensorCompressor.verify(point: point, value: expectedValue, proof: proof)
        print("  Compress + verify (2^\(numVars)): \(verified ? "PASS" : "FAIL")")
        print("  Proof size: \(proof.sizeInElements) field elements (vs \(n) direct)")
    }

    // Test 4: Verify rejects wrong value
    do {
        let numVars = 4
        let n = 1 << numVars
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = frFromInt(nextRng()) }
        var point = [Fr]()
        for _ in 0..<numVars { point.append(frFromInt(nextRng())) }

        let correctValue = TensorCompressor.multilinearEval(evals: evals, point: point)
        let proof = TensorCompressor.compress(evaluations: evals, point: point, value: correctValue)

        let wrongValue = frFromInt(999999)
        let rejected = !TensorCompressor.verify(point: point, value: wrongValue, proof: proof)
        print("  Reject wrong value: \(rejected ? "PASS" : "FAIL")")
    }

    // Test 5: Larger correctness test
    do {
        let numVars = 10
        let n = 1 << numVars
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = frFromInt(nextRng()) }
        var point = [Fr]()
        for _ in 0..<numVars { point.append(frFromInt(nextRng())) }

        let expectedValue = TensorCompressor.multilinearEval(evals: evals, point: point)
        let proof = TensorCompressor.compress(evaluations: evals, point: point, value: expectedValue)
        let verified = TensorCompressor.verify(point: point, value: expectedValue, proof: proof)
        print("  Compress + verify (2^\(numVars)): \(verified ? "PASS" : "FAIL")")
        let sqrtN = 1 << (numVars / 2)
        print("  Proof size: \(proof.sizeInElements) elements (vs \(n) direct, \(sqrtN) sqrt(N))")
    }

    // --- Proof Size Comparison ---
    print("\n--- Proof size comparison (field elements) ---")
    print("  numVars |      N |  Direct | Tensor | Basefold")
    for numVars in stride(from: 10, through: 22, by: 2) {
        let (direct, tensor, basefold) = TensorCompressor.proofSizes(numVars: numVars)
        let n = 1 << numVars
        print(String(format: "  %7d | %6d | %7d | %6d | %8d", numVars, n, direct, tensor, basefold))
    }

    // --- Compression Performance ---
    print("\n--- Compression performance ---")
    let benchSizes = [10, 14, 18]
    for numVars in benchSizes {
        let n = 1 << numVars
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = frFromInt(nextRng()) }
        var point = [Fr]()
        for _ in 0..<numVars { point.append(frFromInt(nextRng())) }
        let value = TensorCompressor.multilinearEval(evals: evals, point: point)

        let _ = TensorCompressor.compress(evaluations: evals, point: point, value: value)

        let runs = 5
        var compressTimes = [Double]()
        var verifyTimes = [Double]()
        for _ in 0..<runs {
            let t0 = CFAbsoluteTimeGetCurrent()
            let proof = TensorCompressor.compress(evaluations: evals, point: point, value: value)
            compressTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)

            let t1 = CFAbsoluteTimeGetCurrent()
            let _ = TensorCompressor.verify(point: point, value: value, proof: proof)
            verifyTimes.append((CFAbsoluteTimeGetCurrent() - t1) * 1000)
        }
        compressTimes.sort()
        verifyTimes.sort()
        let sqrtN = 1 << (numVars / 2)
        let proofSize = TensorCompressor.proofSizes(numVars: numVars).tensor
        print(String(format: "  2^%-2d (N=%7d, sqrt=%4d) | compress: %8.2fms | verify: %8.3fms | proof: %d elems",
                    numVars, n, sqrtN, compressTimes[runs / 2], verifyTimes[runs / 2], proofSize))
    }

    // --- Component Breakdown ---
    print("\n--- Component breakdown (2^14) ---")
    do {
        let numVars = 14
        let n = 1 << numVars
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = frFromInt(nextRng()) }
        var point = [Fr]()
        for _ in 0..<numVars { point.append(frFromInt(nextRng())) }

        let halfVars = numVars / 2
        let sqrtN = 1 << halfVars

        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            let _ = TensorCompressor.tensorProduct(Array(point[0..<halfVars]))
        }
        let tpTime = (CFAbsoluteTimeGetCurrent() - t0) * 10

        let tR = TensorCompressor.tensorProduct(Array(point[halfVars..<numVars]))
        let t1 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            let _ = TensorCompressor.matVecMul(evaluations: evals, vec: tR, rows: sqrtN, cols: sqrtN)
        }
        let mvTime = (CFAbsoluteTimeGetCurrent() - t1) * 10

        let tL = TensorCompressor.tensorProduct(Array(point[0..<halfVars]))
        let v = TensorCompressor.matVecMul(evaluations: evals, vec: tR, rows: sqrtN, cols: sqrtN)
        let challenges = (0..<halfVars).map { _ in frFromInt(nextRng()) }
        let t2 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            let _ = TensorCompressor.cpuSumcheck(evalsA: tL, evalsB: v, challenges: challenges)
        }
        let scTime = (CFAbsoluteTimeGetCurrent() - t2) * 10

        let t3 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            let _ = TensorCompressor.evaluateMatrixRow(evaluations: evals, rowPoint: challenges,
                                                        rows: sqrtN, cols: sqrtN)
        }
        let eqTime = (CFAbsoluteTimeGetCurrent() - t3) * 10

        print(String(format: "  Tensor product (%d elem):   %8.3fms", sqrtN, tpTime))
        print(String(format: "  Matrix-vector (%dx%d):     %8.3fms", sqrtN, sqrtN, mvTime))
        print(String(format: "  Sumcheck (%d rounds):       %8.3fms", halfVars, scTime))
        print(String(format: "  Eq+row eval (%dx%d):       %8.3fms", sqrtN, sqrtN, eqTime))
    }

    // --- Compression Ratio ---
    print("\n--- Compression ratio ---")
    for numVars in [10, 14, 18, 22] {
        let n = 1 << numVars
        let (direct, tensor, _) = TensorCompressor.proofSizes(numVars: numVars)
        let ratio = Double(direct) / Double(tensor)
        print(String(format: "  2^%-2d: %7d -> %5d elements (%.1f x compression)",
                    numVars, direct, tensor, ratio))
    }

    print("\nDone.")
}
