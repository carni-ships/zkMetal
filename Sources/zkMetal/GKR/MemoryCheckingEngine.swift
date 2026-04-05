// Memory Checking Engine
//
// Implements a permutation-based memory checking argument used in zkVMs
// (Jolt, RISC Zero) to verify correct memory read/write operations.
//
// Core idea: for every memory access (addr, value, timestamp), the prover shows
// that the multiset of all reads equals the multiset of all writes (after accounting
// for initialization). This is done via a grand product fingerprint:
//   product(gamma - (addr + beta*value + beta^2*timestamp)) for reads
//   = product(gamma - (addr + beta*value + beta^2*timestamp)) for writes
//
// The engine performs offline memory checking:
//   1. Sort operations by (address, timestamp)
//   2. Verify read-after-write consistency (every read returns the last written value)
//   3. Construct grand product fingerprints
//   4. Verify fingerprint equality (multiset match)
//
// References:
//   - Blum et al. (1991): Checking computations with memory
//   - Lasso/Jolt (Arun et al. 2024): GKR-based memory checking for zkVMs
//   - RISC Zero: Offline memory checking approach

import Foundation
import NeonFieldOps

// MARK: - Memory Operation Enum

/// A memory operation with associated address, value, and timestamp.
public enum MemCheckOp: Equatable {
    /// A read operation: (address, value, timestamp)
    case read(addr: UInt64, value: Fr, timestamp: UInt64)
    /// A write operation: (address, value, timestamp)
    case write(addr: UInt64, value: Fr, timestamp: UInt64)

    /// The memory address being accessed.
    public var addr: UInt64 {
        switch self {
        case .read(let a, _, _): return a
        case .write(let a, _, _): return a
        }
    }

    /// The value read or written.
    public var value: Fr {
        switch self {
        case .read(_, let v, _): return v
        case .write(_, let v, _): return v
        }
    }

    /// The logical timestamp of this operation.
    public var timestamp: UInt64 {
        switch self {
        case .read(_, _, let t): return t
        case .write(_, _, let t): return t
        }
    }

    /// Whether this is a write operation.
    public var isWrite: Bool {
        switch self {
        case .write: return true
        case .read: return false
        }
    }

    public static func == (lhs: MemCheckOp, rhs: MemCheckOp) -> Bool {
        switch (lhs, rhs) {
        case let (.read(a1, v1, t1), .read(a2, v2, t2)):
            return a1 == a2 && t1 == t2 && mceFrEqual(v1, v2)
        case let (.write(a1, v1, t1), .write(a2, v2, t2)):
            return a1 == a2 && t1 == t2 && mceFrEqual(v1, v2)
        default:
            return false
        }
    }
}

// MARK: - Memory Checking Configuration

/// Configuration for the memory checking engine.
public struct MemoryCheckingConfig {
    /// Maximum number of distinct memory addresses (memory size).
    public let memorySize: UInt64
    /// Address width in bits (addresses must be < 2^addressWidth).
    public let addressWidth: Int

    public init(memorySize: UInt64 = 1 << 20, addressWidth: Int = 20) {
        precondition(addressWidth > 0 && addressWidth <= 64, "Address width must be 1..64")
        precondition(memorySize > 0, "Memory size must be positive")
        self.memorySize = memorySize
        self.addressWidth = addressWidth
    }

    /// Default configuration: 1M addresses, 20-bit width.
    public static let defaultConfig = MemoryCheckingConfig()
}

// MARK: - Memory Checking Result

/// Result of a memory checking verification.
public struct MemoryCheckingResult {
    /// Whether the memory trace is valid.
    public let isValid: Bool
    /// The grand product fingerprint for the read multiset.
    public let readFingerprint: Fr
    /// The grand product fingerprint for the write multiset.
    public let writeFingerprint: Fr
    /// Number of read operations (including init/final adjustments).
    public let numReads: Int
    /// Number of write operations (including init/final adjustments).
    public let numWrites: Int
    /// If invalid, a description of the failure.
    public let failureReason: String?

    public init(isValid: Bool, readFingerprint: Fr, writeFingerprint: Fr,
                numReads: Int, numWrites: Int, failureReason: String? = nil) {
        self.isValid = isValid
        self.readFingerprint = readFingerprint
        self.writeFingerprint = writeFingerprint
        self.numReads = numReads
        self.numWrites = numWrites
        self.failureReason = failureReason
    }
}

// MARK: - Memory Checking Engine

/// Engine that verifies memory read/write consistency using offline memory checking
/// with a grand product fingerprint argument.
///
/// Protocol:
///   1. Sort operations by (address, timestamp) for offline checking
///   2. Verify read-after-write consistency: every read returns the last written value
///   3. Construct fingerprints: product(gamma - (addr + beta*value + beta^2*timestamp))
///   4. Verify multiset equality: read_fingerprint == write_fingerprint
///
/// The random challenges (gamma, beta) are derived from a Fiat-Shamir transcript
/// that absorbs the entire operation trace.
public class MemoryCheckingEngine {
    public static let version = Versions.memoryChecking

    private let config: MemoryCheckingConfig

    public init(config: MemoryCheckingConfig = .defaultConfig) {
        self.config = config
    }

    /// Verify a memory trace for read/write consistency.
    ///
    /// - Parameters:
    ///   - ops: The sequence of memory operations in execution order
    ///   - transcript: Fiat-Shamir transcript for deriving random challenges
    /// - Returns: A result indicating validity and fingerprint values
    public func verify(ops: [MemCheckOp], transcript: Transcript) -> MemoryCheckingResult {
        guard !ops.isEmpty else {
            return MemoryCheckingResult(
                isValid: false, readFingerprint: Fr.zero, writeFingerprint: Fr.zero,
                numReads: 0, numWrites: 0, failureReason: "Empty operation trace")
        }

        // Validate addresses
        let maxAddr = UInt64(1) << config.addressWidth
        for op in ops {
            if op.addr >= maxAddr {
                return MemoryCheckingResult(
                    isValid: false, readFingerprint: Fr.zero, writeFingerprint: Fr.zero,
                    numReads: 0, numWrites: 0,
                    failureReason: "Address \(op.addr) exceeds address width \(config.addressWidth)")
            }
        }

        // Step 1: Sort operations by (address, timestamp) for offline checking
        let sorted = ops.sorted { a, b in
            if a.addr != b.addr { return a.addr < b.addr }
            return a.timestamp < b.timestamp
        }

        // Step 2: Verify read-after-write consistency
        // For each address, track the last written value. Every read must match it.
        var lastWritten = [UInt64: Fr]()  // addr -> last written value

        for op in sorted {
            switch op {
            case .write(let addr, let value, _):
                lastWritten[addr] = value
            case .read(let addr, let value, _):
                let expected = lastWritten[addr] ?? Fr.zero
                if !mceFrEqual(value, expected) {
                    return MemoryCheckingResult(
                        isValid: false, readFingerprint: Fr.zero, writeFingerprint: Fr.zero,
                        numReads: 0, numWrites: 0,
                        failureReason: "Read at address \(addr) returned wrong value")
                }
            }
        }

        // Step 3: Derive random challenges from transcript
        absorbOps(ops, into: transcript)
        transcript.absorbLabel("memory-checking-engine-challenges")
        let gamma = transcript.squeeze()
        let beta = transcript.squeeze()
        let beta2 = frMul(beta, beta)

        // Step 4: Separate reads and writes
        let reads = ops.filter { !$0.isWrite }
        let writes = ops.filter { $0.isWrite }

        // Step 5: Compute grand product fingerprints
        // fingerprint = product over all ops of (gamma - (addr + beta*value + beta^2*timestamp))
        let readFingerprint = computeFingerprint(ops: reads, gamma: gamma, beta: beta, beta2: beta2)
        let writeFingerprint = computeFingerprint(ops: writes, gamma: gamma, beta: beta, beta2: beta2)

        // Step 6: Check multiset equality
        let match = mceFrEqual(readFingerprint, writeFingerprint)

        return MemoryCheckingResult(
            isValid: match,
            readFingerprint: readFingerprint,
            writeFingerprint: writeFingerprint,
            numReads: reads.count,
            numWrites: writes.count,
            failureReason: match ? nil : "Fingerprint mismatch: read multiset != write multiset")
    }

    /// Verify with automatic initialization and finalization.
    ///
    /// This is a convenience method that:
    /// 1. Inserts init writes (value=0, timestamp=0) for all accessed addresses
    /// 2. Inserts final reads (last value, timestamp=max+1) for all addresses
    /// 3. Runs the full memory checking verification
    ///
    /// - Parameters:
    ///   - ops: The raw program execution trace
    ///   - transcript: Fiat-Shamir transcript
    /// - Returns: Result with validity and fingerprints
    public func verifyWithInitFinalize(ops: [MemCheckOp], transcript: Transcript) -> MemoryCheckingResult {
        let augmented = augmentOps(ops)
        return verify(ops: augmented, transcript: transcript)
    }

    // MARK: - Internal

    /// Compute the grand product fingerprint for a set of operations.
    /// fingerprint = product_i (gamma - (addr_i + beta * value_i + beta^2 * timestamp_i))
    private func computeFingerprint(ops: [MemCheckOp], gamma: Fr, beta: Fr, beta2: Fr) -> Fr {
        var product = Fr.one
        for op in ops {
            let addrFr = frFromInt(op.addr)
            let tsFr = frFromInt(op.timestamp)
            // inner = addr + beta * value + beta^2 * timestamp
            let inner = frAdd(addrFr, frAdd(frMul(beta, op.value), frMul(beta2, tsFr)))
            // factor = gamma - inner
            let factor = frSub(gamma, inner)
            product = frMul(product, factor)
        }
        return product
    }

    /// Augment the operation trace with init writes and final reads.
    internal func augmentOps(_ ops: [MemCheckOp]) -> [MemCheckOp] {
        // Find all accessed addresses and track last value + max timestamp
        var lastValue = [UInt64: Fr]()
        var maxTimestamp: UInt64 = 0

        for op in ops {
            lastValue[op.addr] = op.value
            maxTimestamp = max(maxTimestamp, op.timestamp)
        }

        let finalTS = maxTimestamp + 1
        let sortedAddrs = lastValue.keys.sorted()

        var augmented = [MemCheckOp]()
        augmented.reserveCapacity(ops.count + 2 * sortedAddrs.count)

        // Init writes: value=0, timestamp=0 for each accessed address
        for addr in sortedAddrs {
            augmented.append(.write(addr: addr, value: Fr.zero, timestamp: 0))
        }

        // Original operations
        augmented.append(contentsOf: ops)

        // Final reads: last value, timestamp=max+1
        for addr in sortedAddrs {
            let val = lastValue[addr]!
            augmented.append(.read(addr: addr, value: val, timestamp: finalTS))
        }

        return augmented
    }

    /// Absorb operation metadata into transcript for Fiat-Shamir binding.
    private func absorbOps(_ ops: [MemCheckOp], into transcript: Transcript) {
        transcript.absorbLabel("memory-checking-engine-trace")
        transcript.absorb(frFromInt(UInt64(ops.count)))
        var addrHash = Fr.zero
        var tsHash = Fr.zero
        for op in ops {
            addrHash = frAdd(addrHash, frFromInt(op.addr))
            tsHash = frAdd(tsHash, frFromInt(op.timestamp))
        }
        transcript.absorb(addrHash)
        transcript.absorb(tsHash)
    }
}

// MARK: - Helpers

/// Compare two Fr elements for equality.
private func mceFrEqual(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
           a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}
