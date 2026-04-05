// EVM Precompile Batch Runner
//
// Processes batches of precompile calls with gas metering matching
// EIP-196 (ecAdd/ecMul), EIP-197 (ecPairing), and EIP-2537 (BLS12-381).

import Foundation

// MARK: - Gas Cost Constants

/// EIP-196/197 gas costs
public enum BN254Gas {
    /// ecAdd (precompile 0x06): fixed 150 gas
    public static let ecAdd: UInt64 = 150
    /// ecMul (precompile 0x07): fixed 6000 gas
    public static let ecMul: UInt64 = 6000
    /// ecPairing (precompile 0x08): base cost
    public static let ecPairingBase: UInt64 = 45000
    /// ecPairing per-pair cost
    public static let ecPairingPerPair: UInt64 = 34000
}

/// EIP-2537 gas costs (post-EIP-2537 pricing)
public enum BLS12381Gas {
    /// G1 add (precompile 0x0A): fixed 500 gas
    public static let g1Add: UInt64 = 500
    /// G1 mul (precompile 0x0B): fixed 12000 gas
    public static let g1Mul: UInt64 = 12000
    /// Pairing base cost
    public static let pairingBase: UInt64 = 115000
    /// Pairing per-pair cost
    public static let pairingPerPair: UInt64 = 23000
}

// MARK: - Precompile Call

/// Identifies which EVM precompile to invoke.
public enum EVMPrecompileID: UInt8, CaseIterable {
    case bn254Add     = 0x06
    case bn254Mul     = 0x07
    case bn254Pairing = 0x08
    case bls12381G1Add     = 0x0A
    case bls12381G1Mul     = 0x0B
    case bls12381Pairing   = 0x10

    public var name: String {
        switch self {
        case .bn254Add: return "ecAdd"
        case .bn254Mul: return "ecMul"
        case .bn254Pairing: return "ecPairing"
        case .bls12381G1Add: return "BLS12-381 G1 Add"
        case .bls12381G1Mul: return "BLS12-381 G1 Mul"
        case .bls12381Pairing: return "BLS12-381 Pairing"
        }
    }
}

/// A single precompile call request.
public struct EVMPrecompileCall {
    public let id: EVMPrecompileID
    public let input: [UInt8]

    public init(id: EVMPrecompileID, input: [UInt8]) {
        self.id = id
        self.input = input
    }
}

/// Result of a single precompile execution.
public struct EVMPrecompileResult {
    public let id: EVMPrecompileID
    public let output: [UInt8]?
    public let gasUsed: UInt64
    public let success: Bool
    public let durationNs: UInt64

    public var durationMs: Double {
        return Double(durationNs) / 1_000_000.0
    }
}

// MARK: - Batch Runner

/// Processes a batch of EVM precompile calls with gas metering and performance reporting.
public struct EVMPrecompileRunner {

    public init() {}

    /// Compute the gas cost for a precompile call.
    public func gasCost(for call: EVMPrecompileCall) -> UInt64 {
        switch call.id {
        case .bn254Add:
            return BN254Gas.ecAdd
        case .bn254Mul:
            return BN254Gas.ecMul
        case .bn254Pairing:
            let n = call.input.count / 192
            return BN254Gas.ecPairingBase + UInt64(n) * BN254Gas.ecPairingPerPair
        case .bls12381G1Add:
            return BLS12381Gas.g1Add
        case .bls12381G1Mul:
            return BLS12381Gas.g1Mul
        case .bls12381Pairing:
            let n = call.input.count / 384
            return BLS12381Gas.pairingBase + UInt64(n) * BLS12381Gas.pairingPerPair
        }
    }

    /// Execute a single precompile call.
    public func execute(_ call: EVMPrecompileCall) -> EVMPrecompileResult {
        let gas = gasCost(for: call)
        let start = DispatchTime.now()

        let output: [UInt8]?
        switch call.id {
        case .bn254Add:
            output = EVMPrecompile06_ecAdd(input: call.input)
        case .bn254Mul:
            output = EVMPrecompile07_ecMul(input: call.input)
        case .bn254Pairing:
            output = EVMPrecompile08_ecPairing(input: call.input)
        case .bls12381G1Add:
            output = EVMPrecompile0A_bls12381G1Add(input: call.input)
        case .bls12381G1Mul:
            output = EVMPrecompile0B_bls12381G1Mul(input: call.input)
        case .bls12381Pairing:
            output = EVMPrecompile10_bls12381Pairing(input: call.input)
        }

        let end = DispatchTime.now()
        let elapsed = end.uptimeNanoseconds - start.uptimeNanoseconds

        return EVMPrecompileResult(
            id: call.id,
            output: output,
            gasUsed: gas,
            success: output != nil,
            durationNs: elapsed
        )
    }

    /// Execute a batch of precompile calls sequentially.
    public func executeBatch(_ calls: [EVMPrecompileCall]) -> EVMBatchResult {
        let batchStart = DispatchTime.now()
        var results = [EVMPrecompileResult]()
        results.reserveCapacity(calls.count)

        for call in calls {
            results.append(execute(call))
        }

        let batchEnd = DispatchTime.now()
        let totalNs = batchEnd.uptimeNanoseconds - batchStart.uptimeNanoseconds

        return EVMBatchResult(results: results, totalDurationNs: totalNs)
    }
}

// MARK: - Batch Result

/// Aggregated result of a batch of precompile executions.
public struct EVMBatchResult {
    public let results: [EVMPrecompileResult]
    public let totalDurationNs: UInt64

    public var totalDurationMs: Double {
        return Double(totalDurationNs) / 1_000_000.0
    }

    public var totalGasUsed: UInt64 {
        return results.reduce(0) { $0 + $1.gasUsed }
    }

    public var successCount: Int {
        return results.filter { $0.success }.count
    }

    public var failureCount: Int {
        return results.filter { !$0.success }.count
    }

    /// Gas throughput in gas/second.
    public var gasThroughput: Double {
        let seconds = Double(totalDurationNs) / 1_000_000_000.0
        return seconds > 0 ? Double(totalGasUsed) / seconds : 0
    }

    /// Print a performance summary.
    public func printReport() {
        print("\n=== EVM Precompile Batch Report ===")
        print(String(format: "Calls: %d (%d succeeded, %d failed)",
                      results.count, successCount, failureCount))
        print(String(format: "Total gas: %llu", totalGasUsed))
        print(String(format: "Total time: %.3f ms", totalDurationMs))
        print(String(format: "Throughput: %.0f gas/sec", gasThroughput))

        // Per-type breakdown
        var byType = [EVMPrecompileID: (count: Int, gas: UInt64, ns: UInt64)]()
        for r in results {
            var entry = byType[r.id] ?? (0, 0, 0)
            entry.count += 1
            entry.gas += r.gasUsed
            entry.ns += r.durationNs
            byType[r.id] = entry
        }

        if byType.count > 1 {
            print("\nBreakdown by type:")
            for (id, stats) in byType.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
                let ms = Double(stats.ns) / 1_000_000.0
                let avgUs = Double(stats.ns) / Double(stats.count) / 1_000.0
                print(String(format: "  %s: %d calls, %llu gas, %.3f ms total (%.1f us/call)",
                             id.name, stats.count, stats.gas, ms, avgUs))
            }
        }
    }
}
