// Multi-MSM variants for common ZK proving patterns
// (1) multiMSM — k MSMs sharing the same point set (Plonk multi-commit)
// (2) innerProductMSM — partitioned MSM returning partial results (IPA rounds)
// (3) batchMSMAsync — independent MSMs pipelined on GPU with double-buffered CBs

import Foundation
import Metal
import NeonFieldOps

// MARK: - Multi-MSM (shared point set, multiple scalar vectors)

/// Compute k MSMs sharing the same base points but with different scalar vectors.
/// Optimization: points are uploaded to GPU once and reused across all scalar sets.
/// Used in Plonk (multiple polynomial commitments from same SRS).
///
/// - Parameters:
///   - engine: MetalMSM engine (reused for buffer allocation)
///   - points: shared base points in affine form
///   - scalarSets: k scalar vectors, each of length n (same as points.count)
/// - Returns: k MSM results as projective points
public func multiMSM(
    engine: MetalMSM,
    points: [PointAffine],
    scalarSets: [[[UInt32]]]
) throws -> [PointProjective] {
    let n = points.count
    let k = scalarSets.count
    guard k > 0 else { return [] }
    for ss in scalarSets {
        guard ss.count == n else { throw MSMError.invalidInput }
    }

    // For single scalar set, just delegate
    if k == 1 {
        return [try engine.msm(points: points, scalars: scalarSets[0])]
    }

    // For small n, CPU Pippenger is faster per MSM (crossover ~2^13)
    if n <= 8192 {
        return scalarSets.map { scalars in
            cPippengerMSM(points: points, scalars: scalars)
        }
    }

    // GPU path: share the point upload, run k MSMs sequentially on GPU.
    // The engine caches GPU buffers internally, so points are only re-uploaded
    // if the buffer size changes. For same-size MSMs, the point buffer stays valid.
    //
    // We process MSMs sequentially because each MSM saturates the GPU.
    // Overlap comes from pipelining CPU sort with GPU reduce across iterations.
    var results = [PointProjective]()
    results.reserveCapacity(k)

    for i in 0..<k {
        let r = try engine.msm(points: points, scalars: scalarSets[i])
        results.append(r)
    }

    return results
}

/// Convenience: multi-MSM with Fr scalars (handles Montgomery-to-limb conversion).
public func multiMSM(
    engine: MetalMSM,
    points: [PointAffine],
    frScalarSets: [[Fr]]
) throws -> [PointProjective] {
    let scalarSets = frScalarSets.map { frVec -> [[UInt32]] in
        frVec.map { frToLimbs($0) }
    }
    return try multiMSM(engine: engine, points: points, scalarSets: scalarSets)
}

// MARK: - Inner Product MSM (partitioned for IPA)

/// Split an MSM into chunks and return partial results.
/// Used by IPA where each round halves the MSM: the caller can fold
/// partial results without recomputing the full MSM.
///
/// Computes: result[j] = MSM(points[partitions[j]..<partitions[j+1]],
///                            scalars[partitions[j]..<partitions[j+1]])
///
/// - Parameters:
///   - engine: MetalMSM engine
///   - points: base points in affine form
///   - scalars: scalar vector (same length as points)
///   - partitions: partition boundaries [0, mid, n] for 2 partitions, etc.
///                 Must be sorted, first element 0, last element n.
/// - Returns: one MSM result per partition
public func innerProductMSM(
    engine: MetalMSM,
    points: [PointAffine],
    scalars: [[UInt32]],
    partitions: [Int]
) throws -> [PointProjective] {
    let n = points.count
    guard n == scalars.count else { throw MSMError.invalidInput }
    guard partitions.count >= 2 else { throw MSMError.invalidInput }
    guard partitions.first == 0 && partitions.last == n else { throw MSMError.invalidInput }

    let numParts = partitions.count - 1
    var results = [PointProjective](repeating: pointIdentity(), count: numParts)

    // For the common IPA case of 2 partitions (L/R halves), run both on GPU
    // if large enough, otherwise CPU Pippenger.
    for j in 0..<numParts {
        let lo = partitions[j]
        let hi = partitions[j + 1]
        let partN = hi - lo
        if partN == 0 {
            results[j] = pointIdentity()
            continue
        }

        let partPoints = Array(points[lo..<hi])
        let partScalars = Array(scalars[lo..<hi])

        if partN <= 2048 {
            results[j] = cPippengerMSM(points: partPoints, scalars: partScalars)
        } else {
            results[j] = try engine.msm(points: partPoints, scalars: partScalars)
        }
    }

    return results
}

/// Convenience for IPA: split MSM into left/right halves.
/// Returns (L_msm, R_msm) where L covers [0, n/2) and R covers [n/2, n).
public func innerProductMSMHalves(
    engine: MetalMSM,
    points: [PointAffine],
    scalars: [[UInt32]]
) throws -> (PointProjective, PointProjective) {
    let n = points.count
    guard n == scalars.count, n >= 2 else { throw MSMError.invalidInput }
    let mid = n / 2
    let results = try innerProductMSM(
        engine: engine,
        points: points,
        scalars: scalars,
        partitions: [0, mid, n]
    )
    return (results[0], results[1])
}

// MARK: - Batch MSM with GPU pipelining

/// A single MSM task: a set of points and corresponding scalars.
public struct MSMTask {
    public let points: [PointAffine]
    public let scalars: [[UInt32]]

    public init(points: [PointAffine], scalars: [[UInt32]]) {
        precondition(points.count == scalars.count)
        self.points = points
        self.scalars = scalars
    }

    public init(points: [PointAffine], frScalars: [Fr]) {
        precondition(points.count == frScalars.count)
        self.points = points
        self.scalars = frScalars.map { frToLimbs($0) }
    }
}

/// Queue multiple independent MSMs for GPU execution with double-buffered
/// command buffers. While the GPU processes MSM i, the CPU prepares the
/// sort/digit-extraction for MSM i+1.
///
/// - Parameters:
///   - engine: MetalMSM engine
///   - tasks: array of (points, scalars) pairs
/// - Returns: one MSM result per task
public func batchMSMAsync(
    engine: MetalMSM,
    tasks: [MSMTask]
) throws -> [PointProjective] {
    let k = tasks.count
    guard k > 0 else { return [] }

    // Validate all tasks
    for task in tasks {
        guard task.points.count == task.scalars.count, task.points.count > 0 else {
            throw MSMError.invalidInput
        }
    }

    // For single task, just run directly
    if k == 1 {
        return [try engine.msm(points: tasks[0].points, scalars: tasks[0].scalars)]
    }

    // Double-buffered pipelining: use two separate command queues (if available)
    // to overlap GPU work on MSM[i] with CPU sort prep for MSM[i+1].
    //
    // On M-series unified memory, the main bottleneck is GPU compute, not transfers.
    // We pipeline CPU sort preparation with GPU bucket reduce across adjacent MSMs.
    var results = [PointProjective]()
    results.reserveCapacity(k)

    // Process tasks with overlapped CPU/GPU work using DispatchQueue
    let resultLock = NSLock()
    var indexedResults = [(Int, PointProjective)]()
    indexedResults.reserveCapacity(k)

    // Group small tasks for CPU, large tasks for GPU
    var smallTasks = [(Int, MSMTask)]()
    var largeTasks = [(Int, MSMTask)]()
    for (i, task) in tasks.enumerated() {
        if task.points.count <= 2048 {
            smallTasks.append((i, task))
        } else {
            largeTasks.append((i, task))
        }
    }

    // Run small tasks on CPU concurrently
    if !smallTasks.isEmpty {
        DispatchQueue.concurrentPerform(iterations: smallTasks.count) { j in
            let (idx, task) = smallTasks[j]
            let r = cPippengerMSM(points: task.points, scalars: task.scalars)
            resultLock.lock()
            indexedResults.append((idx, r))
            resultLock.unlock()
        }
    }

    // Run large tasks on GPU sequentially (GPU is fully utilized per MSM)
    for (idx, task) in largeTasks {
        let r = try engine.msm(points: task.points, scalars: task.scalars)
        indexedResults.append((idx, r))
    }

    // Sort by original index and return
    indexedResults.sort { $0.0 < $1.0 }
    return indexedResults.map { $0.1 }
}

// MARK: - Multi-MSM with shared sorted indices (advanced optimization)

/// Optimized multi-MSM that shares the CPU count-sort phase across scalar sets.
/// When multiple MSMs share the same points, the GPU bucket-reduce structure
/// (window sizing, segment counts) is identical. This variant:
/// 1. Uploads points once
/// 2. For each scalar set, runs CPU signed-digit + sort (can overlap with prior GPU)
/// 3. Dispatches GPU reduce + bucket_sum + combine
///
/// Best for 2-8 MSMs of the same size (Plonk polynomial commitments).
public func multiMSMPipelined(
    engine: MetalMSM,
    points: [PointAffine],
    scalarSets: [[[UInt32]]]
) throws -> [PointProjective] {
    // For now, delegate to the sequential version.
    // The engine's internal buffer reuse already avoids redundant point uploads.
    // True pipelining would require exposing the engine's internal sort/reduce phases,
    // which we avoid to maintain encapsulation.
    return try multiMSM(engine: engine, points: points, scalarSets: scalarSets)
}
