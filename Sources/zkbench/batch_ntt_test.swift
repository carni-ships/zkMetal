// BatchNTTTest.swift
// Correctness test and benchmark for BatchNTTEngine
import Foundation
import Metal
import zkMetal

/// Test batch NTT vs sequential NTT
func runBatchNTTTest() throws {
    print("=== Batch NTT Correctness Test ===\n")

    let engine = try NTTEngine()
    let batchEngine = try BatchNTTEngine()
    let logN = 10  // 2^10 = 1024 elements
    let n = 1 << logN
    let numTransforms = 1

    // Create simple test: input [1, 2, 3, ..., n]
    var input = [Fr](repeating: Fr.zero, count: n)
    for i in 0..<n {
        input[i] = frFromInt(UInt64(i + 1))
    }

    // Create batch buffer with single transform
    guard let batchBuffer = engine.device.makeBuffer(
        length: n * MemoryLayout<Fr>.stride,
        options: .storageModeShared
    ) else {
        throw MSMError.gpuError("Failed to allocate batch buffer")
    }

    // Copy data to batch buffer
    let batchPtr = batchBuffer.contents().bindMemory(to: Fr.self, capacity: n)
    for i in 0..<n {
        batchPtr[i] = input[i]
    }

    // Create sequential buffer
    guard let seqBuffer = engine.device.makeBuffer(
        length: n * MemoryLayout<Fr>.stride,
        options: .storageModeShared
    ) else {
        throw MSMError.gpuError("Failed to allocate buffer")
    }
    let seqPtr = seqBuffer.contents().bindMemory(to: Fr.self, capacity: n)
    for i in 0..<n {
        seqPtr[i] = input[i]
    }

    // Sequential forward NTT
    guard let seqCmdBuf = engine.commandQueue.makeCommandBuffer() else {
        throw MSMError.noCommandBuffer
    }
    try engine.encodeNTT(data: seqBuffer, logN: logN, cmdBuf: seqCmdBuf)
    seqCmdBuf.commit()
    seqCmdBuf.waitUntilCompleted()

    // Batch forward NTT
    guard let batchCmdBuf = engine.commandQueue.makeCommandBuffer() else {
        throw MSMError.noCommandBuffer
    }
    batchEngine.encodeNTTBatch(buffer: batchBuffer, numTransforms: numTransforms, logN: logN, cmdBuf: batchCmdBuf)
    batchCmdBuf.commit()
    batchCmdBuf.waitUntilCompleted()

    // For input [1..1024], first element of NTT output should be sum = 524800
    let batchFirst = frToInt(batchPtr[0])
    let seqFirst = frToInt(seqPtr[0])
    print("First element (should be sum 1..1024 = 524800):")
    print("  Batch:     \(batchFirst)")
    print("  Sequential: \(seqFirst)")
    print("  Match:     \(batchFirst == seqFirst ? "YES" : "NO")")

    // Check all elements match
    var mismatches = 0
    for i in 0..<n {
        let expected = frToInt(seqPtr[i])
        let got = frToInt(batchPtr[i])
        if expected != got {
            if mismatches < 5 {
                print("  MISMATCH at index \(i): expected \(expected), got \(got)")
            }
            mismatches += 1
        }
    }
    print("Forward NTT: \(mismatches == 0 ? "PASS" : "FAIL") (\(mismatches) mismatches)")

    // Test round-trip
    print("\n=== Round-trip Test ===\n")

    // Reload input
    for i in 0..<n {
        batchPtr[i] = input[i]
    }

    // Batch NTT + iNTT
    guard let roundTripCmdBuf = engine.commandQueue.makeCommandBuffer() else {
        throw MSMError.noCommandBuffer
    }
    batchEngine.encodeNTTBatch(buffer: batchBuffer, numTransforms: numTransforms, logN: logN, cmdBuf: roundTripCmdBuf)
    batchEngine.encodeINTTBatch(buffer: batchBuffer, numTransforms: numTransforms, logN: logN, cmdBuf: roundTripCmdBuf)
    roundTripCmdBuf.commit()
    roundTripCmdBuf.waitUntilCompleted()

    // Check recovery
    var errors = 0
    for i in 0..<n {
        let expected = frToInt(input[i])
        let got = frToInt(batchPtr[i])
        if expected != got {
            if errors < 5 {
                print("  RECOVERY ERROR at index \(i): expected \(expected), got \(got)")
            }
            errors += 1
        }
    }
    print("Round-trip recovery: \(errors == 0 ? "PASS" : "FAIL") (\(errors) errors)")
}

/// Benchmark batch NTT performance
func runBatchNTTBench() throws {
    print("=== Batch NTT Performance Benchmark ===\n")

    let engine = try NTTEngine()
    let batchEngine = try BatchNTTEngine()

    let logN = 18  // 2^18 = 262144 elements
    let n = 1 << logN
    let iterations = 10
    let warmup = 2

    // Test with 1 transform (single transform case)
    let numTransforms = 1

    // Create random input
    var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
    var input = [Fr](repeating: Fr.zero, count: n)
    for i in 0..<n {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        input[i] = frFromInt(rng)
    }

    // Create buffers
    guard let batchBuffer = engine.device.makeBuffer(
        length: n * MemoryLayout<Fr>.stride * numTransforms,
        options: .storageModeShared
    ) else {
        throw MSMError.gpuError("Failed to allocate buffer")
    }
    let batchPtr = batchBuffer.contents().bindMemory(to: Fr.self, capacity: n * numTransforms)

    guard let seqBuffer = engine.device.makeBuffer(
        length: n * MemoryLayout<Fr>.stride,
        options: .storageModeShared
    ) else {
        throw MSMError.gpuError("Failed to allocate buffer")
    }
    let seqPtr = seqBuffer.contents().bindMemory(to: Fr.self, capacity: n)

    // Warmup
    for _ in 0..<warmup {
        for i in 0..<n {
            seqPtr[i] = input[i]
            batchPtr[i] = input[i]
        }
        _ = engine.commandQueue.makeCommandBuffer()
        try engine.ntt(data: seqBuffer, logN: logN)
        let cmdBuf2 = engine.commandQueue.makeCommandBuffer()!
        batchEngine.encodeNTTBatch(buffer: batchBuffer, numTransforms: numTransforms, logN: logN, cmdBuf: cmdBuf2)
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()
    }

    // Benchmark sequential NTT
    print("Sequential NTT (1 transform, 2^\(logN) = \(n) elements):")
    var seqTimes: [Double] = []
    for _ in 0..<iterations {
        for i in 0..<n {
            seqPtr[i] = input[i]
        }
        let start = CFAbsoluteTimeGetCurrent()
        try engine.ntt(data: seqBuffer, logN: logN)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        seqTimes.append(elapsed)
    }
    seqTimes.sort()
    let seqMedian = seqTimes[iterations / 2]
    let seqThroughput = Double(n) / seqMedian / 1e6  // million elements per second
    print("  Median: \(String(format: "%.3f", seqMedian * 1000))ms (\(String(format: "%.1f", seqThroughput))M elem/s)")

    // Benchmark batch NTT
    print("\nBatch NTT (1 transform, 2^\(logN) = \(n) elements):")
    var batchTimes: [Double] = []
    for _ in 0..<iterations {
        for i in 0..<n {
            batchPtr[i] = input[i]
        }
        let start = CFAbsoluteTimeGetCurrent()
        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else { continue }
        batchEngine.encodeNTTBatch(buffer: batchBuffer, numTransforms: numTransforms, logN: logN, cmdBuf: cmdBuf)
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        batchTimes.append(elapsed)
    }
    batchTimes.sort()
    let batchMedian = batchTimes[iterations / 2]
    let batchThroughput = Double(n) / batchMedian / 1e6
    print("  Median: \(String(format: "%.3f", batchMedian * 1000))ms (\(String(format: "%.1f", batchThroughput))M elem/s)")

    // Speedup
    let speedup = seqMedian / batchMedian
    print("\nSpeedup (batch vs sequential): \(String(format: "%.2fx", speedup))")

    // Multi-transform benchmark
    print("\n=== Multi-Transform Batch NTT ===")
    print("Testing with K transforms in single buffer...")

    let numTransformsArr = [4, 8, 16, 32]
    for numK in numTransformsArr {
        let totalSize = n * numK
        guard let multiBuffer = engine.device.makeBuffer(
            length: totalSize * MemoryLayout<Fr>.stride,
            options: .storageModeShared
        ) else {
            continue
        }
        let multiPtr = multiBuffer.contents().bindMemory(to: Fr.self, capacity: totalSize)

        // Fill with input data repeated for each transform
        for k in 0..<numK {
            for i in 0..<n {
                multiPtr[k * n + i] = input[i]
            }
        }

        // Warmup
        for _ in 0..<warmup {
            guard let cb = engine.commandQueue.makeCommandBuffer() else { continue }
            batchEngine.encodeNTTBatch(buffer: multiBuffer, numTransforms: numK, logN: logN, cmdBuf: cb)
            cb.commit()
            cb.waitUntilCompleted()
        }

        // Benchmark batch with multiple transforms
        var multiTimes: [Double] = []
        for _ in 0..<iterations {
            let start = CFAbsoluteTimeGetCurrent()
            guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else { continue }
            batchEngine.encodeNTTBatch(buffer: multiBuffer, numTransforms: numK, logN: logN, cmdBuf: cmdBuf)
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            multiTimes.append(elapsed)
        }
        multiTimes.sort()
        let multiMedian = multiTimes[iterations / 2]

        // Calculate per-transform throughput
        let throughput = Double(numK * n) / multiMedian / 1e6  // million elements per second
        let perTransformTime = multiMedian / Double(numK)
        print("\n  \(numK) transforms (\(numK * n) total elements):")
        print("    Total time: \(String(format: "%.3f", multiMedian * 1000))ms")
        print("    Per-transform: \(String(format: "%.3f", perTransformTime * 1000))ms")
        print("    Throughput: \(String(format: "%.1f", throughput))M elem/s")
    }

    print("\n=== Optimization Opportunities ===")
    print("1. Fused bitrev+butterfly stages (reduce kernel launch overhead)")
    print("2. Radix-4 butterflies (fewer stages, better SM utilization)")
    print("3. Shared memory caching for small transforms")
    print("4. Async command buffer execution (overlap with compute)")
}