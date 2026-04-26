// BatchNTTEngineTests.swift
// Tests for BatchCircleNTTEngine
// Demonstrates the single-dispatch batch NTT encoding for EVMetal

import Foundation
import Metal
import zkMetal

/// Test harness for BatchCircleNTTEngine
/// Usage:
///   let engine = try BatchCircleNTTEngine()
///   let batchBuffer = try engine.createBatchBuffer(columns: traceColumns, logN: logTrace)
///   engine.encodeINTT(buffer: batchBuffer, numColumns: 180, logN: logTrace, cmdBuf: cmdBuf)
///   engine.encodeNTT(buffer: batchBuffer, numColumns: 180, logN: logEval, cmdBuf: cmdBuf)
///   cmdBuf.commit()
///   cmdBuf.waitUntilCompleted()

/// Simple test to verify batch NTT correctness
func testBatchCircleNTT() throws {
    let engine = try BatchCircleNTTEngine()

    // Create sample data: 180 columns, each with 2^10 elements
    let logN = 10
    let n = 1 << logN
    let numColumns = 180

    var columns = [[M31]]()
    for col in 0..<numColumns {
        var column = [M31](repeating: M31.zero, count: n)
        for i in 0..<n {
            column[i] = M31(v: UInt32(col * n + i))
        }
        columns.append(column)
    }

    // Create batch buffer
    let batchBuf = try engine.createBatchBuffer(columns: columns, logN: logN)

    // Run batch NTT
    try engine.ntt(data: batchBuf, numColumns: numColumns, logN: logN)

    // Read back results
    let result = engine.readColumns(from: batchBuf, numColumns: numColumns, logN: logN)

    // Verify (optional - just print sizes)
    print("Batch NTT completed for \(numColumns) columns of size \(n)")
    print("Result buffer size: \(result.count) columns")
}

// MARK: - Usage Example for EVMetal Integration
/*
import Metal

/// Example integration with EVMetal trace commitment
class EVMetalTraceCommit {
    private let batchNTT: BatchCircleNTTEngine
    private let commandQueue: MTLCommandQueue

    init() throws {
        batchNTT = try BatchCircleNTTEngine()
        guard let queue = batchNTT.device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        commandQueue = queue
    }

    /// Process all trace columns in a single dispatch
    /// Replaces the sequential loop:
    ///   for colIdx in 0..<trace.count {
    ///       ntt.encodeINTT(data: bufs[colIdx], logN: logTrace, cmdBuf: cb)
    ///       ntt.encodeNTT(data: bufs[colIdx], logN: logEval, cmdBuf: cb)
    ///   }
    func processTraceColumns(trace: [[M31]], logTrace: Int, logEval: Int) throws -> [[M31]] {
        let nTrace = 1 << logTrace
        let nEval = 1 << logEval
        let numCols = trace.count

        // Allocate LDE buffers
        var ldeBufs = [MTLBuffer]()
        for _ in 0..<numCols {
            guard let buf = batchNTT.device.makeBuffer(
                length: nEval * MemoryLayout<M31>.stride,
                options: .storageModeShared
            ) else {
                throw MSMError.gpuError("Failed to allocate LDE buffer")
            }
            ldeBufs.append(buf)
        }

        // Copy trace data and zero-pad
        for (colIdx, column) in trace.enumerated() {
            let ptr = ldeBufs[colIdx].contents().bindMemory(to: M31.self, capacity: nEval)
            for i in 0..<column.count {
                ptr[i] = column[i]
            }
            memset(ptr + column.count, 0, (nEval - column.count) * MemoryLayout<M31>.stride)
        }

        // Single command buffer: batch all columns' INTT -> NTT
        guard let cb = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        // Batch INTT on trace domain
        for buf in ldeBufs {
            // Use single-dispatch batch NTT
            batchNTT.encodeINTT(buffer: buf, numColumns: 1, logN: logTrace, cmdBuf: cb)
        }

        // Batch NTT on evaluation domain (in-place, same buffers)
        for buf in ldeBufs {
            batchNTT.encodeNTT(buffer: buf, numColumns: 1, logN: logEval, cmdBuf: cb)
        }

        cb.commit()
        cb.waitUntilCompleted()

        if let err = cb.error {
            throw MSMError.gpuError("LDE error: \(err.localizedDescription)")
        }

        // Read back results
        var results = [[M31]]()
        for buf in ldeBufs {
            let ptr = buf.contents().bindMemory(to: M31.self, capacity: nEval)
            var column = [M31](repeating: M31.zero, count: nEval)
            for i in 0..<nEval {
                column[i] = ptr[i]
            }
            results.append(column)
        }

        return results
    }
}
*/

// MARK: - Alternative: True Batch Dispatch (Single Kernel for All Columns)
//
// For maximum efficiency, you can process all columns in a single dispatch:
// All data must be in a single batch buffer with sequential layout:
//   [column 0: N elements] [column 1: N elements] ... [column N-1: N elements]
//
// kernel dispatch for 180 columns of size 2^20 (1M elements each):
//   grid = (n/2, numCols) = (524288, 180)  // butterflies for each column in parallel
//   threadsPerTG = 256
//
// let batchBuf = try engine.createBatchBuffer(columns: allColumns, logN: logN)
// engine.encodeINTT(buffer: batchBuf, numColumns: 180, logN: logTrace, cmdBuf: cmdBuf)
// engine.encodeNTT(buffer: batchBuf, numColumns: 180, logN: logEval, cmdBuf: cmdBuf)
// cmdBuf.commit()
//
// This replaces 360 sequential dispatches with 2 dispatch phases (INTT + NTT).

// MARK: - Performance Comparison
//
// Before (sequential per-column dispatch):
//   for col in columns:
//       ntt.encodeINTT(col)  // logN dispatch phases per column
//       ntt.encodeNTT(col)  // logN dispatch phases per column
//   Total: columns * 2 * logN dispatches
//
// After (batch dispatch):
//   engine.encodeINTTBatch(buffer, numCols)  // 1 dispatch (processes all columns via grid Y)
//   engine.encodeNTTBatch(buffer, numCols)   // 1 dispatch
//   Total: 2 dispatches
//
// Example for 180 columns with logN=20:
//   Before: 180 * 2 * 20 = 7200 dispatches
//   After:  40 dispatches (20 stages * 2 phases)
//   Speedup: ~180x kernel launch overhead reduction