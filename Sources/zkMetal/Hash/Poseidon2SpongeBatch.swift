// Poseidon2 Sponge Batch — parallel sponge hashing for multiple independent messages
//
// Provides batch interfaces for hashing N independent messages via the Poseidon2
// sponge construction. Uses CPU parallelism (GCD) for throughput.
//
// Note: GPU batch via poseidon2_permute kernel is not used here because the GPU
// and CPU permutation implementations produce subtly different intermediate values
// for certain inputs (both mathematically correct, but raw-byte differences compound
// across sponge rounds). The CPU CIOS permutation is fast enough for typical batches.

import Foundation
import NeonFieldOps

// MARK: - Batch Sponge Hash (BN254 Fr)

/// Batch sponge hashing for N independent messages.
///
/// Each message is hashed via Poseidon2 sponge (t=3, rate=2, capacity=1) to produce
/// a single Fr output. Messages can have different lengths.
///
/// Uses CPU parallelism via GCD for large batches (>= 64 messages).
///
/// - Parameters:
///   - inputs: Array of messages, each message is an array of Fr elements
///   - domainTag: Domain separation tag (default 0)
///   - engine: Optional Poseidon2Engine (unused, kept for API compatibility)
/// - Returns: Array of N hashes, one per message
public func batchSpongeHash(
    inputs: [[Fr]],
    domainTag: UInt64 = 0,
    engine: Poseidon2Engine? = nil
) throws -> [Fr] {
    let n = inputs.count
    guard n > 0 else { return [] }
    return batchSpongeHashCPU(inputs: inputs, domainTag: domainTag)
}

/// CPU implementation of batch sponge hash.
///
/// Processes each message independently using the Poseidon2Sponge.
/// For large batches (>= 64), uses GCD concurrent dispatch for parallelism.
public func batchSpongeHashCPU(
    inputs: [[Fr]],
    domainTag: UInt64 = 0
) -> [Fr] {
    let n = inputs.count
    guard n > 0 else { return [] }

    if n < 64 {
        // Small batch: sequential
        return inputs.map { msg in
            Poseidon2Sponge.hash(msg, domainTag: domainTag)
        }
    }

    // Large batch: parallel via GCD
    var results = [Fr](repeating: Fr.zero, count: n)
    let chunkSize = max(1, n / ProcessInfo.processInfo.activeProcessorCount)

    DispatchQueue.concurrentPerform(iterations: (n + chunkSize - 1) / chunkSize) { chunk in
        let start = chunk * chunkSize
        let end = min(start + chunkSize, n)
        for i in start..<end {
            results[i] = Poseidon2Sponge.hash(inputs[i], domainTag: domainTag)
        }
    }
    return results
}

// MARK: - Batch Sponge Hash (BabyBear)

/// CPU batch sponge hash for BabyBear messages.
///
/// Each message is hashed via Poseidon2BbSponge to produce 8 Bb elements.
///
/// - Parameters:
///   - inputs: Array of messages, each message is an array of Bb elements
///   - domainTag: Domain separation tag (default 0)
/// - Returns: Flattened array of N*8 Bb elements (8 per message)
public func batchSpongeHashBb(
    inputs: [[Bb]],
    domainTag: UInt32 = 0
) -> [Bb] {
    let n = inputs.count
    guard n > 0 else { return [] }

    let outputPerMsg = 8
    if n < 64 {
        var results = [Bb]()
        results.reserveCapacity(n * outputPerMsg)
        for msg in inputs {
            let h = Poseidon2BbSponge.hash(msg, domainTag: domainTag)
            results.append(contentsOf: h)
        }
        return results
    }

    // Large batch: parallel via GCD
    var results = [Bb](repeating: Bb(v: 0), count: n * outputPerMsg)
    let chunkSize = max(1, n / ProcessInfo.processInfo.activeProcessorCount)
    DispatchQueue.concurrentPerform(iterations: (n + chunkSize - 1) / chunkSize) { chunk in
        let start = chunk * chunkSize
        let end = min(start + chunkSize, n)
        for i in start..<end {
            let h = Poseidon2BbSponge.hash(inputs[i], domainTag: domainTag)
            for j in 0..<outputPerMsg {
                results[i * outputPerMsg + j] = h[j]
            }
        }
    }
    return results
}

// MARK: - Uniform-Length Batch

/// Batch hash for uniform-length messages.
///
/// When all messages have the same length, this avoids per-message array slicing
/// by working directly on the flat input array.
///
/// - Parameters:
///   - flatInput: All messages concatenated: N * messageLen Fr elements
///   - messageLen: Number of Fr elements per message (must be uniform)
///   - domainTag: Domain separation tag
///   - engine: Optional Poseidon2Engine (unused, kept for API compatibility)
/// - Returns: Array of N hashes
public func batchSpongeHashUniform(
    flatInput: [Fr],
    messageLen: Int,
    domainTag: UInt64 = 0,
    engine: Poseidon2Engine? = nil
) throws -> [Fr] {
    precondition(flatInput.count % messageLen == 0, "Input length must be multiple of messageLen")
    let n = flatInput.count / messageLen
    guard n > 0 else { return [] }

    var results = [Fr](repeating: Fr.zero, count: n)

    if n < 64 {
        for i in 0..<n {
            let msg = Array(flatInput[i * messageLen ..< (i + 1) * messageLen])
            results[i] = Poseidon2Sponge.hash(msg, domainTag: domainTag)
        }
    } else {
        let chunkSize = max(1, n / ProcessInfo.processInfo.activeProcessorCount)
        DispatchQueue.concurrentPerform(iterations: (n + chunkSize - 1) / chunkSize) { chunk in
            let start = chunk * chunkSize
            let end = min(start + chunkSize, n)
            for i in start..<end {
                let msg = Array(flatInput[i * messageLen ..< (i + 1) * messageLen])
                results[i] = Poseidon2Sponge.hash(msg, domainTag: domainTag)
            }
        }
    }

    return results
}
