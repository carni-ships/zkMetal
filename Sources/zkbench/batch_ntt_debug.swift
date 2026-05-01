// BatchNTTDebug.swift
// Debug test for batch NTT - verify grid dimensions and basic kernel behavior
import Foundation
import Metal
import zkMetal

/// Test batch NTT by calling a minimal sequence and checking intermediate results
func runBatchNTTMinimalDebug() throws {
    print("=== Batch NTT Minimal Debug ===\n")

    let logN = 4  // 16 elements
    let n = 1 << logN
    let numTransforms = 1

    let engine = try NTTEngine()
    let batchEngine = try BatchNTTEngine()

    // Input: simple pattern [1, 2, 3, ...]
    var input = [Fr](repeating: Fr.zero, count: n)
    for i in 0..<n {
        input[i] = frFromInt(UInt64(i + 1))
    }

    // Test 1: Just check that buffer layout is correct after copy
    print("Test 1: Buffer copy verification")
    guard let batchBuffer = engine.device.makeBuffer(
        length: n * MemoryLayout<Fr>.stride * numTransforms,
        options: .storageModeShared
    ) else {
        throw MSMError.gpuError("Failed to allocate buffer")
    }
    let batchPtr = batchBuffer.contents().bindMemory(to: Fr.self, capacity: n * numTransforms)
    for i in 0..<n {
        batchPtr[i] = input[i]
    }

    print("Input matches buffer: ", terminator: "")
    var match = true
    for i in 0..<n {
        if frToUInt64(batchPtr[i]) != UInt64(i + 1) {
            match = false
            break
        }
    }
    print(match ? "YES" : "NO")

    // Test 2: Run only the bitrev kernel alone and check output
    // We can't call it directly since it's private, but we can run encodeNTTBatch
    // and look at intermediate results... actually we can't get those.

    // Instead, let's try running the same input through the sequential path
    // multiple times to verify consistency

    print("\nTest 2: Sequential consistency")
    guard let seqBuffer = engine.device.makeBuffer(
        length: n * MemoryLayout<Fr>.stride,
        options: .storageModeShared
    ) else {
        throw MSMError.gpuError("Failed to allocate buffer")
    }
    let seqPtr = seqBuffer.contents().bindMemory(to: Fr.self, capacity: n)

    // Run 3 times and check they all match
    var results: [[UInt64]] = []
    for run in 0..<3 {
        for i in 0..<n {
            seqPtr[i] = input[i]
        }
        try engine.ntt(data: seqBuffer, logN: logN)
        let result = (0..<n).map { frToUInt64(seqPtr[$0]) }
        results.append(result)
    }

    print("All 3 runs match: \(results[0] == results[1] && results[1] == results[2] ? "YES" : "NO")")

    // Test 3: Now run batch 3 times and check consistency
    print("\nTest 3: Batch consistency")
    var batchResults: [[UInt64]] = []
    for _ in 0..<3 {
        for i in 0..<n {
            batchPtr[i] = input[i]
        }
        guard let cmdBuf = engine.commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        batchEngine.encodeNTTBatch(buffer: batchBuffer, numTransforms: numTransforms, logN: logN, cmdBuf: cmdBuf)
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        let result = (0..<n).map { frToUInt64(batchPtr[$0]) }
        batchResults.append(result)
    }

    print("All 3 batch runs match: \(batchResults[0] == batchResults[1] && batchResults[1] == batchResults[2] ? "YES" : "NO")")
    print("First batch result: \(batchResults[0])")

    // Test 4: Compare sequential vs batch (first run)
    print("\nTest 4: Sequential vs Batch")
    print("Sequential: \(results[0])")
    print("Batch:      \(batchResults[0])")
    print("Match:      \(results[0] == batchResults[0] ? "YES" : "NO")")
}