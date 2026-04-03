// Parallel CPU MSM — Pippenger's algorithm with multithreaded bucket accumulation
// Does NOT replace vanilla sequential MSM (pointScalarMul loop in benchmarks).
// BN254 G1 only.

import Foundation

/// Parallel CPU Pippenger MSM for BN254.
/// Uses window-based decomposition with threaded bucket accumulation and window-parallel reduction.
public func parallelMSM(points: [PointAffine], scalars: [[UInt32]]) -> PointProjective {
    let n = points.count
    precondition(n == scalars.count)
    if n == 0 { return pointIdentity() }

    // Choose window size based on n
    let windowBits: Int
    if n <= 256 { windowBits = 8 }
    else if n <= 4096 { windowBits = 10 }
    else if n <= 65536 { windowBits = 12 }
    else { windowBits = 14 }

    let numWindows = (256 + windowBits - 1) / windowBits
    let numBuckets = (1 << windowBits) - 1  // skip bucket 0 (identity)

    // Convert scalars from 8×UInt32 limbs to integer form for bit extraction
    // scalars are already in integer form (not Montgomery)

    // Extract window digit for scalar i, window w
    func getWindowDigit(_ scalarIdx: Int, _ window: Int) -> Int {
        let bitOffset = window * windowBits
        let limbs = scalars[scalarIdx]
        var digit: UInt32 = 0
        for bit in 0..<windowBits {
            let globalBit = bitOffset + bit
            if globalBit >= 256 { break }
            let limbIdx = globalBit / 32
            let bitInLimb = globalBit % 32
            if limbs[limbIdx] & (1 << bitInLimb) != 0 {
                digit |= 1 << bit
            }
        }
        return Int(digit)
    }

    // Process each window in parallel
    var windowResults = [PointProjective](repeating: pointIdentity(), count: numWindows)
    let nThreads = ProcessInfo.processInfo.activeProcessorCount

    DispatchQueue.concurrentPerform(iterations: numWindows) { w in
        // Bucket accumulation for this window
        var buckets = [PointProjective](repeating: pointIdentity(), count: numBuckets + 1)

        for i in 0..<n {
            let digit = getWindowDigit(i, w)
            if digit != 0 {
                let pt = pointFromAffine(points[i])
                buckets[digit] = pointAdd(buckets[digit], pt)
            }
        }

        // Running-sum reduction: result = sum_{j=1}^{numBuckets} j * buckets[j]
        var runningSum = pointIdentity()
        var windowSum = pointIdentity()
        for j in stride(from: numBuckets, through: 1, by: -1) {
            runningSum = pointAdd(runningSum, buckets[j])
            windowSum = pointAdd(windowSum, runningSum)
        }

        windowResults[w] = windowSum
    }

    // Combine windows: result = sum_w windowResults[w] * 2^(w*windowBits)
    // Process from most significant window down using Horner's method
    var result = windowResults[numWindows - 1]
    for w in stride(from: numWindows - 2, through: 0, by: -1) {
        // Shift by windowBits
        for _ in 0..<windowBits {
            result = pointDouble(result)
        }
        result = pointAdd(result, windowResults[w])
    }

    return result
}
