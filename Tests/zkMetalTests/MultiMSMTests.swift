import Foundation
import zkMetal

func runMultiMSMTests() {
    let gx = fpFromInt(1), gy = fpFromInt(2)
    let gProj = pointFromAffine(PointAffine(x: gx, y: gy))

    // Generate test points and scalars
    let n = 64
    var projPts = [PointProjective](); var acc = gProj
    for _ in 0..<n { projPts.append(acc); acc = pointAdd(acc, gProj) }
    let pts = batchToAffine(projPts)

    var rng: UInt64 = 0xDEAD_CAFE_1234_5678
    func nextScalar() -> [UInt32] {
        var limbs = [UInt32](repeating: 0, count: 8)
        for j in 0..<8 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
        }
        return limbs
    }

    // MARK: - Multi-MSM tests

    suite("Multi-MSM")
    do {
        let engine = try MetalMSM()

        // Generate 3 scalar sets
        let k = 3
        var scalarSets = [[[UInt32]]]()
        for _ in 0..<k {
            var scalars = [[UInt32]]()
            for _ in 0..<n { scalars.append(nextScalar()) }
            scalarSets.append(scalars)
        }

        // Multi-MSM
        let multiResults = try multiMSM(engine: engine, points: pts, scalarSets: scalarSets)
        expect(multiResults.count == k, "multiMSM returns k results")

        // Verify each result matches individual MSM
        for i in 0..<k {
            let singleResult = try engine.msm(points: pts, scalars: scalarSets[i])
            expect(pointEqual(multiResults[i], singleResult), "multiMSM[\(i)] matches single MSM")
        }

        // Empty scalar sets
        let emptyResult = try multiMSM(engine: engine, points: pts, scalarSets: [])
        expect(emptyResult.isEmpty, "multiMSM empty scalarSets returns empty")

        // Single scalar set
        let singleSetResult = try multiMSM(engine: engine, points: pts, scalarSets: [scalarSets[0]])
        expect(singleSetResult.count == 1, "multiMSM single set returns 1 result")
        expect(pointEqual(singleSetResult[0], multiResults[0]), "multiMSM single set matches")
    } catch {
        expect(false, "Multi-MSM error: \(error)")
    }

    // MARK: - Inner Product MSM tests

    suite("Inner Product MSM")
    do {
        let engine = try MetalMSM()

        var scalars = [[UInt32]]()
        for _ in 0..<n { scalars.append(nextScalar()) }

        // Split into halves
        let mid = n / 2
        let (left, right) = try innerProductMSMHalves(engine: engine, points: pts, scalars: scalars)

        // Verify left half
        let leftPts = Array(pts[0..<mid])
        let leftScalars = Array(scalars[0..<mid])
        let leftExpected = cPippengerMSM(points: leftPts, scalars: leftScalars)
        expect(pointEqual(left, leftExpected), "innerProductMSM left half correct")

        // Verify right half
        let rightPts = Array(pts[mid..<n])
        let rightScalars = Array(scalars[mid..<n])
        let rightExpected = cPippengerMSM(points: rightPts, scalars: rightScalars)
        expect(pointEqual(right, rightExpected), "innerProductMSM right half correct")

        // Verify L + R = full MSM
        let fullResult = cPippengerMSM(points: pts, scalars: scalars)
        let sumLR = pointAdd(left, right)
        expect(pointEqual(sumLR, fullResult), "L + R = full MSM")

        // Test 4-way partition
        let q1 = n / 4, q2 = n / 2, q3 = 3 * n / 4
        let parts = try innerProductMSM(
            engine: engine, points: pts, scalars: scalars,
            partitions: [0, q1, q2, q3, n])
        expect(parts.count == 4, "4-way partition returns 4 results")
        var partSum = pointIdentity()
        for p in parts { partSum = pointAdd(partSum, p) }
        expect(pointEqual(partSum, fullResult), "4-way partition sum = full MSM")
    } catch {
        expect(false, "Inner Product MSM error: \(error)")
    }

    // MARK: - Batch MSM Async tests

    suite("Batch MSM Async")
    do {
        let engine = try MetalMSM()

        // Create 4 independent MSM tasks of varying sizes
        var tasks = [MSMTask]()
        for taskSize in [16, 32, 48, 64] {
            let taskPts = Array(pts.prefix(taskSize))
            var taskScalars = [[UInt32]]()
            for _ in 0..<taskSize { taskScalars.append(nextScalar()) }
            tasks.append(MSMTask(points: taskPts, scalars: taskScalars))
        }

        let batchResults = try batchMSMAsync(engine: engine, tasks: tasks)
        expect(batchResults.count == 4, "batchMSMAsync returns 4 results")

        // Verify each result
        for (i, task) in tasks.enumerated() {
            let expected = cPippengerMSM(points: task.points, scalars: task.scalars)
            expect(pointEqual(batchResults[i], expected), "batchMSMAsync[\(i)] correct")
        }

        // Single task
        let singleBatch = try batchMSMAsync(engine: engine, tasks: [tasks[0]])
        expect(singleBatch.count == 1, "batchMSMAsync single task returns 1 result")

        // Empty tasks
        let emptyBatch = try batchMSMAsync(engine: engine, tasks: [])
        expect(emptyBatch.isEmpty, "batchMSMAsync empty returns empty")
    } catch {
        expect(false, "Batch MSM Async error: \(error)")
    }

    // MARK: - MSM Precomputation tests

    suite("MSM Precomputation (BGMW)")
    do {
        let smallN = 32
        let smallPts = Array(pts.prefix(smallN))

        // Precompute table
        let table = precomputeWindowTable(points: smallPts, windowBits: 7)
        expect(table.pointCount == smallN, "table pointCount")
        expect(table.windowBits == 7, "table windowBits")
        expect(table.numWindows == 37, "table numWindows (ceil(256/7))")
        expect(table.tableSize == 127, "table tableSize (2^7 - 1)")
        expect(table.totalEntries == smallN * 37 * 127, "table totalEntries")

        // BGMW MSM vs Pippenger
        var scalars = [[UInt32]]()
        for _ in 0..<smallN { scalars.append(nextScalar()) }

        let bgmwResult = bgmwMSM(table: table, scalars: scalars)
        let pippengerResult = cPippengerMSM(points: smallPts, scalars: scalars)
        expect(pointEqual(bgmwResult, pippengerResult), "BGMW MSM matches Pippenger")

        // Test with multiple scalar sets
        var scalarSets = [[[UInt32]]]()
        for _ in 0..<3 {
            var ss = [[UInt32]]()
            for _ in 0..<smallN { ss.append(nextScalar()) }
            scalarSets.append(ss)
        }
        let bgmwMulti = bgmwMultiMSM(table: table, scalarSets: scalarSets)
        expect(bgmwMulti.count == 3, "bgmwMultiMSM returns 3 results")
        for i in 0..<3 {
            let expected = cPippengerMSM(points: smallPts, scalars: scalarSets[i])
            expect(pointEqual(bgmwMulti[i], expected), "bgmwMultiMSM[\(i)] correct")
        }

        // Serialization round-trip
        let tmpDir = FileManager.default.temporaryDirectory
        let tmpFile = tmpDir.appendingPathComponent("test_bgmw_\(ProcessInfo.processInfo.processIdentifier).bgmw")
        defer { try? FileManager.default.removeItem(at: tmpFile) }

        try serializePrecomputedTable(table, to: tmpFile)
        let loaded = try deserializePrecomputedTable(from: tmpFile)
        expect(loaded.pointCount == table.pointCount, "deserialized pointCount")
        expect(loaded.windowBits == table.windowBits, "deserialized windowBits")
        expect(loaded.numWindows == table.numWindows, "deserialized numWindows")
        expect(loaded.tableSize == table.tableSize, "deserialized tableSize")

        // Verify loaded table produces same MSM result
        let loadedResult = bgmwMSM(table: loaded, scalars: scalars)
        expect(pointEqual(loadedResult, pippengerResult), "deserialized table MSM matches")

        // Cached precomputation
        let cachedTable = getOrPrecomputeTable(points: smallPts, windowBits: 7, cacheKey: "test_n32_w7")
        let cachedResult = bgmwMSM(table: cachedTable, scalars: scalars)
        expect(pointEqual(cachedResult, pippengerResult), "cached table MSM matches")

        // Second call should load from cache
        let cachedTable2 = getOrPrecomputeTable(points: smallPts, windowBits: 7, cacheKey: "test_n32_w7")
        let cachedResult2 = bgmwMSM(table: cachedTable2, scalars: scalars)
        expect(pointEqual(cachedResult2, pippengerResult), "re-cached table MSM matches")
    } catch {
        expect(false, "MSM Precomputation error: \(error)")
    }

    // Edge case: identity / zero scalar
    suite("MSM Precomputation Edge Cases")
    do {
        let smallN = 4
        let smallPts = Array(pts.prefix(smallN))
        let table = precomputeWindowTable(points: smallPts, windowBits: 7)

        // All-zero scalars -> identity
        let zeroScalars = [[UInt32]](repeating: [UInt32](repeating: 0, count: 8), count: smallN)
        let zeroResult = bgmwMSM(table: table, scalars: zeroScalars)
        expect(pointIsIdentity(zeroResult), "BGMW all-zero scalars -> identity")

        // Scalar = 1 for first point, 0 for rest
        var oneScalars = [[UInt32]](repeating: [UInt32](repeating: 0, count: 8), count: smallN)
        oneScalars[0] = [1, 0, 0, 0, 0, 0, 0, 0]
        let oneResult = bgmwMSM(table: table, scalars: oneScalars)
        let expected = pointFromAffine(smallPts[0])
        expect(pointEqual(oneResult, expected), "BGMW scalar=1 for first point")
    }
}
