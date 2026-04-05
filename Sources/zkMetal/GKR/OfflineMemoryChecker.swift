// Offline Memory Checker
//
// Higher-level wrapper around MemoryCheckingEngine that handles the full
// lifecycle of memory checking for a program execution trace:
//   1. Takes raw read/write operations from program execution
//   2. Automatically inserts initialization writes (addr=i, value=0, timestamp=0)
//   3. Automatically inserts final reads for consistency closure
//   4. Runs the memory checking argument via MemoryCheckingEngine
//   5. Returns success/failure + fingerprint values for proof composition
//
// This mirrors the offline memory checking approach used in Jolt and RISC Zero:
// the checker sees the entire trace after execution and verifies it offline,
// rather than checking each operation online during execution.
//
// References:
//   - Lasso/Jolt (Arun et al. 2024): offline memory checking for zkVMs
//   - RISC Zero: offline memory checking with permutation argument

import Foundation
import NeonFieldOps

// MARK: - Offline Checker Result

/// Result from the offline memory checker, including fingerprints for proof composition.
public struct OfflineCheckResult {
    /// Whether the execution trace has valid memory behavior.
    public let success: Bool
    /// Read multiset fingerprint (for proof composition with outer IOP).
    public let readFingerprint: Fr
    /// Write multiset fingerprint (for proof composition with outer IOP).
    public let writeFingerprint: Fr
    /// Number of original operations in the trace (before augmentation).
    public let numOriginalOps: Int
    /// Number of augmented operations (after init writes + final reads).
    public let numAugmentedOps: Int
    /// Number of distinct addresses accessed.
    public let numAddresses: Int
    /// If failed, the reason.
    public let failureReason: String?
}

// MARK: - Offline Memory Checker

/// Offline memory checker that takes a program execution trace and verifies
/// that all memory reads return the correct values.
///
/// Usage:
///   let checker = OfflineMemoryChecker()
///   checker.write(addr: 0, value: frFromInt(42), timestamp: 1)
///   checker.read(addr: 0, value: frFromInt(42), timestamp: 2)
///   let result = checker.check()
///   assert(result.success)
///
/// The checker automatically handles initialization (all memory starts at 0)
/// and finalization (inserts final reads to close the permutation argument).
public class OfflineMemoryChecker {
    public static let version = Versions.memoryChecking

    private var ops: [MemCheckOp] = []
    private var currentTimestamp: UInt64 = 1
    private var memory: [UInt64: Fr] = [:]
    private let config: MemoryCheckingConfig

    public init(config: MemoryCheckingConfig = .defaultConfig) {
        self.config = config
    }

    /// Record a write operation.
    ///
    /// - Parameters:
    ///   - addr: Memory address to write
    ///   - value: Value to write
    ///   - timestamp: Optional explicit timestamp (auto-increments if nil)
    @discardableResult
    public func write(addr: UInt64, value: Fr, timestamp: UInt64? = nil) -> UInt64 {
        let ts = timestamp ?? currentTimestamp
        ops.append(.write(addr: addr, value: value, timestamp: ts))
        memory[addr] = value
        if timestamp == nil { currentTimestamp += 1 }
        return ts
    }

    /// Record a read operation.
    ///
    /// - Parameters:
    ///   - addr: Memory address to read
    ///   - value: The value that was read (will be checked for consistency)
    ///   - timestamp: Optional explicit timestamp (auto-increments if nil)
    @discardableResult
    public func read(addr: UInt64, value: Fr, timestamp: UInt64? = nil) -> UInt64 {
        let ts = timestamp ?? currentTimestamp
        ops.append(.read(addr: addr, value: value, timestamp: ts))
        if timestamp == nil { currentTimestamp += 1 }
        return ts
    }

    /// Record a read that automatically uses the correct value from the last write.
    /// Returns the value read.
    @discardableResult
    public func readCorrect(addr: UInt64, timestamp: UInt64? = nil) -> Fr {
        let val = memory[addr] ?? Fr.zero
        let ts = timestamp ?? currentTimestamp
        ops.append(.read(addr: addr, value: val, timestamp: ts))
        if timestamp == nil { currentTimestamp += 1 }
        return val
    }

    /// Add a raw MemCheckOp to the trace.
    public func addOp(_ op: MemCheckOp) {
        ops.append(op)
        if op.isWrite {
            memory[op.addr] = op.value
        }
    }

    /// Run the offline memory checking argument on the recorded trace.
    ///
    /// This method:
    /// 1. Inserts init writes (addr, value=0, timestamp=0) for all accessed addresses
    /// 2. Inserts final reads (addr, last_value, timestamp=max+1) for all addresses
    /// 3. Runs the memory checking engine verification
    /// 4. Returns the result with fingerprints for proof composition
    public func check() -> OfflineCheckResult {
        return checkOps(ops)
    }

    /// Run offline memory checking on a provided list of operations.
    /// Useful when you already have a complete trace from another source.
    public func checkOps(_ operations: [MemCheckOp]) -> OfflineCheckResult {
        guard !operations.isEmpty else {
            return OfflineCheckResult(
                success: false, readFingerprint: Fr.zero, writeFingerprint: Fr.zero,
                numOriginalOps: 0, numAugmentedOps: 0, numAddresses: 0,
                failureReason: "Empty operation trace")
        }

        // Collect address info
        var addresses = Set<UInt64>()
        for op in operations {
            addresses.insert(op.addr)
        }

        let engine = MemoryCheckingEngine(config: config)
        let augmented = engine.augmentOps(operations)
        let transcript = Transcript(label: "offline-memory-checker")
        let result = engine.verify(ops: augmented, transcript: transcript)

        return OfflineCheckResult(
            success: result.isValid,
            readFingerprint: result.readFingerprint,
            writeFingerprint: result.writeFingerprint,
            numOriginalOps: operations.count,
            numAugmentedOps: augmented.count,
            numAddresses: addresses.count,
            failureReason: result.failureReason)
    }

    /// Reset the checker for reuse.
    public func reset() {
        ops.removeAll()
        memory.removeAll()
        currentTimestamp = 1
    }

    /// Get the current operation count.
    public var operationCount: Int { ops.count }

    /// Get all recorded operations.
    public var operations: [MemCheckOp] { ops }
}

// MARK: - Batch Offline Memory Checker

/// Checks multiple independent memory spaces in a single pass.
/// Useful for zkVMs that have separate register file and main memory.
public class BatchOfflineMemoryChecker {
    public static let version = Versions.memoryChecking

    private var checkers: [String: OfflineMemoryChecker] = [:]
    private let config: MemoryCheckingConfig

    public init(config: MemoryCheckingConfig = .defaultConfig) {
        self.config = config
    }

    /// Get or create a checker for a named memory space.
    public func space(_ name: String) -> OfflineMemoryChecker {
        if let existing = checkers[name] { return existing }
        let checker = OfflineMemoryChecker(config: config)
        checkers[name] = checker
        return checker
    }

    /// Check all memory spaces. Returns (allValid, per-space results).
    public func checkAll() -> (success: Bool, results: [(String, OfflineCheckResult)]) {
        var results = [(String, OfflineCheckResult)]()
        var allValid = true
        for (name, checker) in checkers.sorted(by: { $0.key < $1.key }) {
            let result = checker.check()
            results.append((name, result))
            if !result.success { allValid = false }
        }
        return (allValid, results)
    }

    /// Reset all memory spaces.
    public func reset() {
        checkers.removeAll()
    }
}
