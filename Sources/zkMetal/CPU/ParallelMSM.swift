// Parallel CPU MSM — Pippenger's algorithm with multithreaded bucket accumulation
// Does NOT replace vanilla sequential MSM (pointScalarMul loop in benchmarks).
// BN254 G1 only.

import Foundation
import NeonFieldOps

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

// MARK: - C point scalar multiplication

/// Fast point scalar multiplication using C CIOS field arithmetic.
/// ~300× faster than Swift pointScalarMul for BN254 Fp operations.
public func cPointScalarMul(_ p: PointProjective, _ scalar: Fr) -> PointProjective {
    let limbs = frToLimbs(scalar)
    var result = PointProjective(x: .one, y: .one, z: .zero)
    withUnsafeBytes(of: p) { pBuf in
        limbs.withUnsafeBufferPointer { scBuf in
            withUnsafeMutableBytes(of: &result) { resBuf in
                bn254_point_scalar_mul(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    scBuf.baseAddress!,
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}

/// C-accelerated Fr inner product: sum(a[i] * b[i])
public func cFrInnerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
    precondition(a.count == b.count)
    var result = Fr.zero
    a.withUnsafeBytes { aBuf in
        b.withUnsafeBytes { bBuf in
            withUnsafeMutableBytes(of: &result) { resBuf in
                bn254_fr_inner_product(
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(a.count),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}

/// C-accelerated Fr vector fold: out[i] = a[i]*x + b[i]*xInv
public func cFrVectorFold(_ a: [Fr], _ b: [Fr], x: Fr, xInv: Fr) -> [Fr] {
    precondition(a.count == b.count)
    let n = a.count
    var out = [Fr](repeating: .zero, count: n)
    a.withUnsafeBytes { aBuf in
        b.withUnsafeBytes { bBuf in
            withUnsafeBytes(of: x) { xBuf in
                withUnsafeBytes(of: xInv) { xiBuf in
                    out.withUnsafeMutableBytes { outBuf in
                        bn254_fr_vector_fold(
                            aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            xBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            xiBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                            Int32(n),
                            outBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                        )
                    }
                }
            }
        }
    }
    return out
}

// MARK: - C Pippenger MSM (CIOS Montgomery field ops, multi-threaded)

/// Optimized CPU Pippenger MSM using C CIOS Montgomery arithmetic.
/// ~20-30× faster than Swift Pippenger due to __uint128_t field ops.
public func cPippengerMSM(points: [PointAffine], scalars: [[UInt32]]) -> PointProjective {
    let n = points.count
    precondition(n == scalars.count)
    if n == 0 { return pointIdentity() }

    // Flatten scalars (Swift [[UInt32]] → contiguous [UInt32])
    var flatScalars = [UInt32]()
    flatScalars.reserveCapacity(n * 8)
    for s in scalars { flatScalars.append(contentsOf: s) }

    var result = PointProjective(x: .one, y: .one, z: .zero)

    points.withUnsafeBytes { ptsBuf in
        flatScalars.withUnsafeBufferPointer { scBuf in
            withUnsafeMutableBytes(of: &result) { resBuf in
                let ptsPtr = ptsBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                let resPtr = resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                bn254_pippenger_msm(ptsPtr, scBuf.baseAddress!, Int32(n), resPtr)
            }
        }
    }
    return result
}

/// MSM from projective points — direct scalar-mul accumulation (no affine conversion).
/// Faster than Pippenger for small n (avoids batchToAffine + thread overhead).
public func cMSMProjective(points: [PointProjective], scalars: [Fr]) -> PointProjective {
    let n = points.count
    precondition(n == scalars.count)
    if n == 0 { return pointIdentity() }

    // Convert Fr scalars to UInt32 limbs
    var flatScalars = [UInt32]()
    flatScalars.reserveCapacity(n * 8)
    for s in scalars { flatScalars.append(contentsOf: frToLimbs(s)) }

    var result = PointProjective(x: .one, y: .one, z: .zero)

    points.withUnsafeBytes { ptsBuf in
        flatScalars.withUnsafeBufferPointer { scBuf in
            withUnsafeMutableBytes(of: &result) { resBuf in
                bn254_msm_projective(
                    ptsBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    scBuf.baseAddress!,
                    Int32(n),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                )
            }
        }
    }
    return result
}
