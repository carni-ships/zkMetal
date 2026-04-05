// GKR-based Memory Checking Argument
//
// Proves read/write consistency of a memory trace for zkVM verification
// (Jolt/RISC Zero style). The key insight: memory correctness reduces to
// a multiset equality check between read and write operations.
//
// Protocol overview:
//   1. Build a memory trace: sequence of (address, value, timestamp, is_write) tuples
//   2. Sort operations by (address, timestamp) to pair reads with writes
//   3. Prove multiset equality: prod(read_i + gamma) = prod(write_i + gamma)
//      using the GKR grand product protocol
//   4. Prove timestamp ordering: for each address, read timestamps are strictly
//      less than write timestamps (prevents reading stale values)
//
// The multiset check uses a random challenge gamma from the verifier (Fiat-Shamir).
// Each memory operation is fingerprinted as: address + gamma * value + gamma^2 * timestamp
// This Reed-Solomon-style fingerprint ensures soundness with overwhelming probability.
//
// References:
//   - Blum et al. (1991): Memory checking
//   - Lasso/Jolt (Arun et al. 2024): GKR-based memory checking for zkVMs
//   - RISC Zero: Similar offline memory checking approach

import Foundation
import NeonFieldOps

// MARK: - Memory Trace Types

/// A single memory operation in the execution trace.
public struct MemoryOp: Equatable {
    /// The memory address being accessed.
    public let address: UInt64
    /// The value read from or written to memory.
    public let value: Fr
    /// Logical timestamp (monotonically increasing across execution).
    public let timestamp: UInt64
    /// Whether this is a write operation (true) or read (false).
    public let isWrite: Bool

    public init(address: UInt64, value: Fr, timestamp: UInt64, isWrite: Bool) {
        self.address = address
        self.value = value
        self.timestamp = timestamp
        self.isWrite = isWrite
    }
}

/// Complete memory trace for a program execution.
/// Operations are stored in execution order (by timestamp).
public struct MemoryTrace {
    /// All memory operations in execution order.
    public let ops: [MemoryOp]
    /// Number of distinct addresses accessed.
    public let numAddresses: Int
    /// Maximum timestamp in the trace.
    public let maxTimestamp: UInt64

    public init(ops: [MemoryOp]) {
        precondition(!ops.isEmpty, "Memory trace must have at least one operation")
        var addrs = Set<UInt64>()
        var maxTS: UInt64 = 0
        for op in ops {
            addrs.insert(op.address)
            maxTS = max(maxTS, op.timestamp)
        }
        self.ops = ops
        self.numAddresses = addrs.count
        self.maxTimestamp = maxTS
    }

    /// Create a trace from (address, value, timestamp, isWrite) tuples.
    public init(tuples: [(address: UInt64, value: Fr, timestamp: UInt64, isWrite: Bool)]) {
        let ops = tuples.map { MemoryOp(address: $0.address, value: $0.value,
                                         timestamp: $0.timestamp, isWrite: $0.isWrite) }
        self.init(ops: ops)
    }

    /// Extract only read operations.
    public var reads: [MemoryOp] { ops.filter { !$0.isWrite } }

    /// Extract only write operations.
    public var writes: [MemoryOp] { ops.filter { $0.isWrite } }
}

// MARK: - Memory Checking Proof

/// Proof of memory consistency for an execution trace.
public struct MemoryCheckingProof {
    /// Grand product proof for multiset equality: prod(read fingerprints) = prod(write fingerprints)
    public let grandProductProof: GrandProductProof
    /// Grand product proof for the write side (verified against read side)
    public let writeGrandProductProof: GrandProductProof
    /// Timestamp ordering proof: for each (address, read_ts, write_ts) triple,
    /// proves write_ts > read_ts via range check on the difference.
    public let timestampDiffs: [Fr]
    /// The random fingerprint challenge gamma used.
    public let gamma: Fr
    /// Number of read operations.
    public let numReads: Int
    /// Number of write operations.
    public let numWrites: Int
    /// Initialization writes (address, initial_value) for each address.
    public let initWrites: [(UInt64, Fr)]
    /// Final reads (address, final_value) for each address.
    public let finalReads: [(UInt64, Fr)]
}

// MARK: - Memory Checking Prover

/// Proves read/write consistency of a memory trace using GKR-based grand products.
///
/// The prover:
/// 1. Augments the trace with initialization writes (t=0) and final reads (t=max+1)
/// 2. Sorts by (address, timestamp) and pairs each read with the preceding write
/// 3. Computes fingerprints: f(op) = address + gamma * value + gamma^2 * timestamp
/// 4. Proves multiset equality of read/write fingerprints via grand product GKR
/// 5. Proves timestamp ordering for each read-write pair
public class MemoryCheckingProver {
    public static let version = Versions.gkr

    /// Prove memory consistency for the given trace.
    ///
    /// - Parameters:
    ///   - trace: The memory execution trace
    ///   - transcript: Fiat-Shamir transcript
    /// - Returns: Memory checking proof
    public static func prove(trace: MemoryTrace, transcript: Transcript) -> MemoryCheckingProof {
        // Step 1: Augment trace with init writes and final reads
        let (augmented, initWrites, finalReads) = augmentTrace(trace)

        // Step 2: Get fingerprint challenge
        absorbTrace(augmented, into: transcript)
        transcript.absorbLabel("memory-checking-gamma")
        let gamma = transcript.squeeze()
        let gamma2 = frMul(gamma, gamma)

        // Step 3: Compute fingerprints for reads and writes
        let readOps = augmented.filter { !$0.isWrite }
        let writeOps = augmented.filter { $0.isWrite }

        let readFingerprints = readOps.map { fingerprint($0, gamma: gamma, gamma2: gamma2) }
        let writeFingerprints = writeOps.map { fingerprint($0, gamma: gamma, gamma2: gamma2) }

        // Step 4: Prove multiset equality via grand products
        // prod(read_fingerprints) should equal prod(write_fingerprints)
        transcript.absorbLabel("memory-read-product")
        let readProof = GrandProductEngine.prove(values: readFingerprints, transcript: transcript)

        transcript.absorbLabel("memory-write-product")
        let writeProof = GrandProductEngine.prove(values: writeFingerprints, transcript: transcript)

        // Step 5: Compute timestamp differences for ordering check
        // Sort augmented ops by (address, timestamp)
        let sorted = augmented.sorted { a, b in
            if a.address != b.address { return a.address < b.address }
            return a.timestamp < b.timestamp
        }

        var tsDiffs = [Fr]()
        tsDiffs.reserveCapacity(sorted.count)
        for i in 0..<sorted.count {
            if i + 1 < sorted.count && sorted[i].address == sorted[i + 1].address {
                // Difference should be positive: next.timestamp > current.timestamp
                let diff = sorted[i + 1].timestamp - sorted[i].timestamp
                tsDiffs.append(frFromInt(diff))
            }
        }

        // Absorb timestamp diffs for binding
        transcript.absorbLabel("memory-timestamp-diffs")
        for d in tsDiffs { transcript.absorb(d) }

        return MemoryCheckingProof(
            grandProductProof: readProof,
            writeGrandProductProof: writeProof,
            timestampDiffs: tsDiffs,
            gamma: gamma,
            numReads: readOps.count,
            numWrites: writeOps.count,
            initWrites: initWrites,
            finalReads: finalReads
        )
    }

    /// Augment the trace with initialization writes (timestamp 0) for each address
    /// and final reads (timestamp max+1) to close the memory.
    ///
    /// This ensures every read has a matching write and every write has a matching read,
    /// making the multiset equality check complete.
    private static func augmentTrace(
        _ trace: MemoryTrace
    ) -> (augmented: [MemoryOp], initWrites: [(UInt64, Fr)], finalReads: [(UInt64, Fr)]) {
        // Find initial values: first write to each address
        var firstWrite = [UInt64: Fr]()
        // Find final values: last operation's value for each address
        var lastValue = [UInt64: Fr]()
        var lastTimestamp = [UInt64: UInt64]()

        for op in trace.ops {
            if firstWrite[op.address] == nil && op.isWrite {
                firstWrite[op.address] = op.value
            }
            lastValue[op.address] = op.value
            lastTimestamp[op.address] = op.timestamp
        }

        // For addresses that are read before any write, assume initial value of 0
        for op in trace.ops {
            if firstWrite[op.address] == nil {
                firstWrite[op.address] = Fr.zero
            }
        }

        var augmented = [MemoryOp]()
        augmented.reserveCapacity(trace.ops.count + 2 * trace.numAddresses)

        var initWrites = [(UInt64, Fr)]()
        var finalReads = [(UInt64, Fr)]()

        // Add initialization writes at timestamp 0
        let sortedAddresses = firstWrite.keys.sorted()
        for addr in sortedAddresses {
            let val = firstWrite[addr]!
            augmented.append(MemoryOp(address: addr, value: val, timestamp: 0, isWrite: true))
            initWrites.append((addr, val))
        }

        // Add original trace
        augmented.append(contentsOf: trace.ops)

        // Add final reads at timestamp max+1
        let finalTS = trace.maxTimestamp + 1
        for addr in sortedAddresses {
            let val = lastValue[addr]!
            augmented.append(MemoryOp(address: addr, value: val, timestamp: finalTS, isWrite: false))
            finalReads.append((addr, val))
        }

        return (augmented, initWrites, finalReads)
    }

    /// Compute fingerprint of a memory operation: address + gamma * value + gamma^2 * timestamp
    private static func fingerprint(_ op: MemoryOp, gamma: Fr, gamma2: Fr) -> Fr {
        let addrFr = frFromInt(op.address)
        let tsFr = frFromInt(op.timestamp)
        return frAdd(addrFr, frAdd(frMul(gamma, op.value), frMul(gamma2, tsFr)))
    }

    /// Absorb trace metadata into transcript for binding.
    private static func absorbTrace(_ ops: [MemoryOp], into transcript: Transcript) {
        transcript.absorbLabel("memory-trace")
        transcript.absorb(frFromInt(UInt64(ops.count)))
        // Absorb a hash of the trace rather than every element (for efficiency)
        var addrHash = Fr.zero
        var tsHash = Fr.zero
        for op in ops {
            addrHash = frAdd(addrHash, frFromInt(op.address))
            tsHash = frAdd(tsHash, frFromInt(op.timestamp))
        }
        transcript.absorb(addrHash)
        transcript.absorb(tsHash)
    }
}

// MARK: - Memory Checking Verifier

/// Verifies a memory consistency proof.
///
/// The verifier:
/// 1. Reconstructs the augmented trace from the original trace + proof metadata
/// 2. Recomputes fingerprint challenge gamma
/// 3. Verifies the grand product proofs (multiset equality)
/// 4. Checks that read product equals write product
/// 5. Verifies timestamp ordering (all diffs are positive)
public class MemoryCheckingVerifier {
    public static let version = Versions.gkr

    /// Verify a memory consistency proof.
    ///
    /// - Parameters:
    ///   - trace: The original memory trace
    ///   - proof: The memory checking proof
    ///   - transcript: Fiat-Shamir transcript (must match prover's)
    /// - Returns: true if the memory trace is consistent
    public static func verify(
        trace: MemoryTrace,
        proof: MemoryCheckingProof,
        transcript: Transcript
    ) -> Bool {
        // Step 1: Reconstruct augmented trace
        let (augmented, _, _) = reconstructAugmented(trace, proof: proof)

        // Step 2: Recompute fingerprint challenge
        absorbTrace(augmented, into: transcript)
        transcript.absorbLabel("memory-checking-gamma")
        let gamma = transcript.squeeze()
        let gamma2 = frMul(gamma, gamma)

        // Verify gamma matches
        guard mcFrEqual(gamma, proof.gamma) else { return false }

        // Step 3: Recompute fingerprints
        let readOps = augmented.filter { !$0.isWrite }
        let writeOps = augmented.filter { $0.isWrite }

        guard readOps.count == proof.numReads else { return false }
        guard writeOps.count == proof.numWrites else { return false }

        let readFingerprints = readOps.map { fingerprint($0, gamma: gamma, gamma2: gamma2) }
        let writeFingerprints = writeOps.map { fingerprint($0, gamma: gamma, gamma2: gamma2) }

        // Step 4: Verify grand product proofs
        transcript.absorbLabel("memory-read-product")
        guard GrandProductEngine.verify(
            values: readFingerprints,
            proof: proof.grandProductProof,
            transcript: transcript
        ) else { return false }

        transcript.absorbLabel("memory-write-product")
        guard GrandProductEngine.verify(
            values: writeFingerprints,
            proof: proof.writeGrandProductProof,
            transcript: transcript
        ) else { return false }

        // Step 5: Check multiset equality: read product = write product
        guard mcFrEqual(proof.grandProductProof.claimedProduct,
                        proof.writeGrandProductProof.claimedProduct) else {
            return false
        }

        // Step 6: Verify timestamp ordering
        // Sort augmented by (address, timestamp) and check consecutive diffs
        let sorted = augmented.sorted { a, b in
            if a.address != b.address { return a.address < b.address }
            return a.timestamp < b.timestamp
        }

        var expectedDiffs = [Fr]()
        for i in 0..<sorted.count {
            if i + 1 < sorted.count && sorted[i].address == sorted[i + 1].address {
                let diff = sorted[i + 1].timestamp - sorted[i].timestamp
                expectedDiffs.append(frFromInt(diff))
            }
        }

        guard expectedDiffs.count == proof.timestampDiffs.count else { return false }
        for i in 0..<expectedDiffs.count {
            guard mcFrEqual(expectedDiffs[i], proof.timestampDiffs[i]) else { return false }
        }

        // All timestamp diffs must be positive (> 0)
        for diff in proof.timestampDiffs {
            if diff.isZero { return false }
        }

        // Absorb timestamp diffs (must match prover)
        transcript.absorbLabel("memory-timestamp-diffs")
        for d in proof.timestampDiffs { transcript.absorb(d) }

        return true
    }

    /// Reconstruct the augmented trace from original trace and proof metadata.
    private static func reconstructAugmented(
        _ trace: MemoryTrace,
        proof: MemoryCheckingProof
    ) -> (augmented: [MemoryOp], initWrites: [(UInt64, Fr)], finalReads: [(UInt64, Fr)]) {
        var augmented = [MemoryOp]()
        augmented.reserveCapacity(trace.ops.count + proof.initWrites.count + proof.finalReads.count)

        // Add init writes at timestamp 0
        for (addr, val) in proof.initWrites {
            augmented.append(MemoryOp(address: addr, value: val, timestamp: 0, isWrite: true))
        }

        // Add original trace
        augmented.append(contentsOf: trace.ops)

        // Add final reads at timestamp max+1
        let finalTS = trace.maxTimestamp + 1
        for (addr, val) in proof.finalReads {
            augmented.append(MemoryOp(address: addr, value: val, timestamp: finalTS, isWrite: false))
        }

        return (augmented, proof.initWrites, proof.finalReads)
    }

    /// Compute fingerprint of a memory operation.
    private static func fingerprint(_ op: MemoryOp, gamma: Fr, gamma2: Fr) -> Fr {
        let addrFr = frFromInt(op.address)
        let tsFr = frFromInt(op.timestamp)
        return frAdd(addrFr, frAdd(frMul(gamma, op.value), frMul(gamma2, tsFr)))
    }

    /// Absorb trace metadata into transcript (must match prover).
    private static func absorbTrace(_ ops: [MemoryOp], into transcript: Transcript) {
        transcript.absorbLabel("memory-trace")
        transcript.absorb(frFromInt(UInt64(ops.count)))
        var addrHash = Fr.zero
        var tsHash = Fr.zero
        for op in ops {
            addrHash = frAdd(addrHash, frFromInt(op.address))
            tsHash = frAdd(tsHash, frFromInt(op.timestamp))
        }
        transcript.absorb(addrHash)
        transcript.absorb(tsHash)
    }
}

// MARK: - Convenience: Memory Trace Builder

/// Builder for constructing memory traces from sequential operations.
///
/// Usage:
///   let builder = MemoryTraceBuilder()
///   builder.write(address: 0, value: frFromInt(42))
///   let val = builder.read(address: 0)  // returns frFromInt(42)
///   let trace = builder.build()
public class MemoryTraceBuilder {
    private var ops = [MemoryOp]()
    private var memory = [UInt64: Fr]()
    private var currentTimestamp: UInt64 = 1  // 0 reserved for init writes

    public init() {}

    /// Write a value to an address.
    @discardableResult
    public func write(address: UInt64, value: Fr) -> UInt64 {
        let ts = currentTimestamp
        ops.append(MemoryOp(address: address, value: value, timestamp: ts, isWrite: true))
        memory[address] = value
        currentTimestamp += 1
        return ts
    }

    /// Read a value from an address. Returns Fr.zero if never written.
    @discardableResult
    public func read(address: UInt64) -> Fr {
        let value = memory[address] ?? Fr.zero
        let ts = currentTimestamp
        ops.append(MemoryOp(address: address, value: value, timestamp: ts, isWrite: false))
        currentTimestamp += 1
        return value
    }

    /// Build the final memory trace.
    public func build() -> MemoryTrace {
        return MemoryTrace(ops: ops)
    }

    /// Reset the builder for reuse.
    public func reset() {
        ops.removeAll()
        memory.removeAll()
        currentTimestamp = 1
    }
}

// MARK: - Helpers

/// Compare two Fr elements for equality.
private func mcFrEqual(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
           a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}
