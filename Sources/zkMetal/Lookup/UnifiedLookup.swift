// UnifiedLookup -- Single interface for LogUp, Lasso, and cq lookup arguments
//
// Dispatches to the optimal backend based on table characteristics:
//   - LogUp (LookupEngine): general-purpose, O(n + N) prover
//   - Lasso (LassoEngine): tensor-decomposable tables, O(n * c) prover with small subtables
//   - cq (CQEngine): large unstructured tables, O(n log n) prover after O(N^2) preprocessing
//
// Auto-selection heuristic:
//   - Structured tables with Lasso decomposition -> Lasso
//   - Tables with existing cq preprocessing -> cq
//   - Tables > 2^20 unstructured -> cq (amortized over many proofs)
//   - Everything else -> LogUp
//
// References:
//   LogUp: Haboeck 2022
//   Lasso: Setty et al. 2023
//   cq: Eagen-Fiore-Gabizon 2022

import Foundation
import Metal

// MARK: - Strategy Enum

/// The lookup strategy to use for proving.
public enum LookupStrategy: String, CustomStringConvertible {
    /// LogUp (logarithmic derivative) -- general purpose, O(n + N)
    case logUp = "LogUp"
    /// Lasso (structured decomposition) -- O(n * c) with small subtables
    case lasso = "Lasso"
    /// Cached Quotients -- O(n log n) prover, O(N^2) one-time setup
    case cq = "cq"

    public var description: String { rawValue }
}

// MARK: - Unified Proof

/// A proof produced by the unified lookup engine.
/// Wraps the underlying strategy-specific proof with common metadata.
public struct UnifiedLookupProof {
    /// Which strategy was used
    public let strategy: LookupStrategy
    /// Number of lookups proven
    public let numLookups: Int
    /// Table size
    public let tableSize: Int
    /// Underlying LogUp proof (non-nil when strategy == .logUp)
    public let logUpProof: LookupProof?
    /// Underlying Lasso proof (non-nil when strategy == .lasso)
    public let lassoProof: LassoProof?
    /// Underlying cq proof (non-nil when strategy == .cq)
    public let cqProof: CQProof?

    public init(strategy: LookupStrategy, numLookups: Int, tableSize: Int,
                logUpProof: LookupProof? = nil,
                lassoProof: LassoProof? = nil,
                cqProof: CQProof? = nil) {
        self.strategy = strategy
        self.numLookups = numLookups
        self.tableSize = tableSize
        self.logUpProof = logUpProof
        self.lassoProof = lassoProof
        self.cqProof = cqProof
    }
}

// MARK: - Unified Lookup Engine

/// Unified lookup engine that dispatches to the optimal backend.
///
/// Usage:
///   let engine = try UnifiedLookupEngine()
///   let table = RangeTable(bits: 16)
///   let proof = try engine.prove(lookups: myValues, table: table)
///   let valid = try engine.verify(proof: proof, lookups: myValues, table: table)
///
/// Or with explicit strategy:
///   let proof = try engine.prove(lookups: myValues, table: table, strategy: .lasso)
///
/// For cq with preprocessing:
///   let preprocessed = try engine.preprocessCQ(table: table, srs: mySRS)
///   let proof = try engine.prove(lookups: myValues, table: table,
///                                 cqPreprocessed: preprocessed)
public class UnifiedLookupEngine {
    public static let version = Versions.lookup

    // Lazily initialized backends (avoid Metal init cost if not needed)
    private var _logUpEngine: LookupEngine?
    private var _lassoEngine: LassoEngine?
    private var _cqEngines: [ObjectIdentifier: CQEngine] = [:]

    /// Whether to emit profiling info to stderr
    public var profile = false

    public init() {}

    // MARK: - Lazy Backend Access

    private func logUpEngine() throws -> LookupEngine {
        if let e = _logUpEngine { return e }
        let e = try LookupEngine()
        e.profileLogUp = profile
        _logUpEngine = e
        return e
    }

    private func lassoEngine() throws -> LassoEngine {
        if let e = _lassoEngine { return e }
        let e = try LassoEngine()
        e.profileLasso = profile
        _lassoEngine = e
        return e
    }

    // MARK: - Strategy Selection

    /// Auto-select the best strategy for the given table and lookup count.
    ///
    /// Heuristic:
    ///   1. If the table has a LassoTable (tensor-decomposable) -> Lasso
    ///   2. If cq preprocessing is provided -> cq
    ///   3. If table is very large (> 2^20) and unstructured -> cq
    ///   4. Otherwise -> LogUp
    ///
    /// Override with an explicit strategy parameter if needed.
    public func selectStrategy(table: PrebuiltLookupTable,
                                numLookups: Int,
                                cqPreprocessed: CQTableCommitment? = nil) -> LookupStrategy {
        // If cq is already preprocessed, use it
        if cqPreprocessed != nil {
            return .cq
        }

        // Structured tables with Lasso decomposition
        if table.isStructured, table.lassoTable != nil {
            // Lasso is best when subtables are small relative to lookup count.
            // For very small tables (e.g., 256 entries, 256 lookups), LogUp is simpler.
            let subtableTotal = table.lassoTable!.subtables.reduce(0) { $0 + $1.count }
            if subtableTotal < table.count / 2 {
                return .lasso
            }
        }

        // Large unstructured tables benefit from cq (if amortizing setup)
        if table.count > (1 << 20) && !table.isStructured {
            return .cq
        }

        // Default: LogUp handles everything reasonably
        return .logUp
    }

    // MARK: - Prove

    /// Prove that every element in `lookups` exists in the table.
    ///
    /// Auto-selects strategy unless overridden. For cq, pass a preprocessed table
    /// commitment to avoid recomputing the expensive setup.
    ///
    /// - Parameters:
    ///   - lookups: Values to prove membership for (must all exist in table)
    ///   - table: The lookup table
    ///   - strategy: Override strategy (nil for auto-select)
    ///   - cqPreprocessed: Preprocessed cq table commitment (required for cq)
    ///   - cqSRS: SRS for cq (required for cq if no preprocessed table)
    /// - Returns: Unified proof
    public func prove(lookups: [Fr],
                      table: PrebuiltLookupTable,
                      strategy: LookupStrategy? = nil,
                      cqPreprocessed: CQTableCommitment? = nil,
                      cqSRS: [PointAffine]? = nil) throws -> UnifiedLookupProof {
        let chosen = strategy ?? selectStrategy(table: table,
                                                 numLookups: lookups.count,
                                                 cqPreprocessed: cqPreprocessed)

        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        let proof: UnifiedLookupProof

        switch chosen {
        case .logUp:
            proof = try proveLogUp(lookups: lookups, table: table)

        case .lasso:
            guard let lassoTable = table.lassoTable else {
                // Fallback to LogUp if no Lasso decomposition
                return try proveLogUp(lookups: lookups, table: table)
            }
            proof = try proveLasso(lookups: lookups, table: table, lassoTable: lassoTable)

        case .cq:
            guard let srs = cqSRS else {
                preconditionFailure("cq strategy requires SRS (pass cqSRS parameter)")
            }
            proof = try proveCQ(lookups: lookups, table: table,
                                preprocessed: cqPreprocessed, srs: srs)
        }

        if profile {
            let elapsed = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "[unified-lookup] %@ prove: %.2f ms (n=%d, N=%d)\n",
                         chosen.rawValue, elapsed, lookups.count, table.count), stderr)
        }

        return proof
    }

    /// Prove using a batch of lookups against the same table.
    /// More efficient than individual proofs when multiple lookup vectors
    /// share the same table (common in zkVM instruction lookups).
    ///
    /// - Parameters:
    ///   - batches: Array of lookup vectors, all against the same table
    ///   - table: The shared lookup table
    ///   - strategy: Override strategy (nil for auto-select)
    /// - Returns: Array of proofs, one per batch
    public func proveBatch(batches: [[Fr]],
                           table: PrebuiltLookupTable,
                           strategy: LookupStrategy? = nil) throws -> [UnifiedLookupProof] {
        var proofs = [UnifiedLookupProof]()
        proofs.reserveCapacity(batches.count)
        for lookups in batches {
            let proof = try prove(lookups: lookups, table: table, strategy: strategy)
            proofs.append(proof)
        }
        return proofs
    }

    // MARK: - Verify

    /// Verify a unified lookup proof.
    ///
    /// - Parameters:
    ///   - proof: The proof to verify
    ///   - lookups: Original lookup values
    ///   - table: The lookup table
    ///   - cqPreprocessed: cq preprocessed table (required for cq verification)
    ///   - cqSRSSecret: SRS secret for cq verification (test mode)
    /// - Returns: true if the proof is valid
    public func verify(proof: UnifiedLookupProof,
                       lookups: [Fr],
                       table: PrebuiltLookupTable,
                       cqPreprocessed: CQTableCommitment? = nil,
                       cqSRSSecret: Fr? = nil) throws -> Bool {
        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        let valid: Bool

        switch proof.strategy {
        case .logUp:
            guard let logUpProof = proof.logUpProof else { return false }
            let engine = try logUpEngine()
            // Pad table/lookups to match proof dimensions
            let paddedTable = padToPowerOf2(table.values)
            var paddedLookups = lookups
            padArrayToPowerOf2(&paddedLookups, padValue: lookups.first ?? Fr.zero)
            valid = try engine.verify(proof: logUpProof, table: paddedTable, lookups: paddedLookups)

        case .lasso:
            guard let lassoProof = proof.lassoProof,
                  let lassoTable = table.lassoTable else { return false }
            let engine = try lassoEngine()
            var paddedLookups = lookups
            padArrayToPowerOf2(&paddedLookups, padValue: lookups.first ?? Fr.zero)
            valid = try engine.verify(proof: lassoProof, lookups: paddedLookups, table: lassoTable)

        case .cq:
            guard let cqProof = proof.cqProof,
                  let preprocessed = cqPreprocessed,
                  let srsSecret = cqSRSSecret else { return false }
            let srs = preprocessed.cachedQuotientCommitments.isEmpty
                ? [PointAffine]() : [] // CQEngine already has the SRS
            let engine = try CQEngine(srs: srs)
            valid = engine.verify(proof: cqProof, table: preprocessed,
                                  numLookups: lookups.count, srsSecret: srsSecret)
        }

        if profile {
            let elapsed = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "[unified-lookup] %@ verify: %.2f ms -> %@\n",
                         proof.strategy.rawValue, elapsed, valid ? "OK" : "FAIL"), stderr)
        }

        return valid
    }

    // MARK: - cq Preprocessing

    /// Preprocess a table for the cq strategy.
    /// This is a one-time O(N^2) cost that is amortized over many proofs.
    ///
    /// - Parameters:
    ///   - table: The lookup table to preprocess
    ///   - srs: KZG structured reference string
    /// - Returns: Preprocessed table commitment for use with prove/verify
    public func preprocessCQ(table: PrebuiltLookupTable,
                              srs: [PointAffine]) throws -> CQTableCommitment {
        let paddedTable = padToPowerOf2(table.values)
        let engine = try CQEngine(srs: srs)
        return try engine.preprocessTable(table: paddedTable)
    }

    // MARK: - Private: Strategy-Specific Prove

    private func proveLogUp(lookups: [Fr],
                             table: PrebuiltLookupTable) throws -> UnifiedLookupProof {
        let engine = try logUpEngine()

        // Pad to power of 2
        let paddedTable = padToPowerOf2(table.values)
        var paddedLookups = lookups
        // Pad with a valid table value so lookup still passes
        let padValue = paddedTable[0]
        padArrayToPowerOf2(&paddedLookups, padValue: padValue)

        // Derive beta from Fiat-Shamir
        let beta = deriveBeta(lookups: paddedLookups, table: paddedTable)

        let proof = try engine.prove(table: paddedTable, lookups: paddedLookups, beta: beta)
        return UnifiedLookupProof(
            strategy: .logUp,
            numLookups: lookups.count,
            tableSize: table.count,
            logUpProof: proof
        )
    }

    private func proveLasso(lookups: [Fr],
                             table: PrebuiltLookupTable,
                             lassoTable: LassoTable) throws -> UnifiedLookupProof {
        let engine = try lassoEngine()

        var paddedLookups = lookups
        padArrayToPowerOf2(&paddedLookups, padValue: Fr.zero)

        let proof = try engine.prove(lookups: paddedLookups, table: lassoTable)
        return UnifiedLookupProof(
            strategy: .lasso,
            numLookups: lookups.count,
            tableSize: table.count,
            lassoProof: proof
        )
    }

    private func proveCQ(lookups: [Fr],
                          table: PrebuiltLookupTable,
                          preprocessed: CQTableCommitment?,
                          srs: [PointAffine]) throws -> UnifiedLookupProof {
        let engine = try CQEngine(srs: srs)

        let tableCommitment: CQTableCommitment
        if let pre = preprocessed {
            tableCommitment = pre
        } else {
            let paddedTable = padToPowerOf2(table.values)
            tableCommitment = try engine.preprocessTable(table: paddedTable)
        }

        let proof = try engine.prove(lookups: lookups, table: tableCommitment)
        return UnifiedLookupProof(
            strategy: .cq,
            numLookups: lookups.count,
            tableSize: table.count,
            cqProof: proof
        )
    }

    // MARK: - Helpers

    /// Pad an array to the next power of 2, filling with the given value.
    private func padToPowerOf2(_ values: [Fr]) -> [Fr] {
        let n = values.count
        var p = 1
        while p < n { p <<= 1 }
        if p == n { return values }
        var result = values
        let padVal = values.last ?? Fr.zero
        while result.count < p {
            result.append(padVal)
        }
        return result
    }

    /// Pad an array in-place to the next power of 2.
    private func padArrayToPowerOf2(_ values: inout [Fr], padValue: Fr) {
        let n = values.count
        var p = 1
        while p < n { p <<= 1 }
        while values.count < p {
            values.append(padValue)
        }
    }

    /// Derive a Fiat-Shamir beta challenge from the lookup+table data.
    private func deriveBeta(lookups: [Fr], table: [Fr]) -> Fr {
        var transcript = [UInt8]()
        transcript.reserveCapacity(128)
        // Hash first few lookups + table size for a deterministic challenge
        var tSize = UInt64(table.count)
        for _ in 0..<8 {
            transcript.append(UInt8(tSize & 0xFF))
            tSize >>= 8
        }
        var lSize = UInt64(lookups.count)
        for _ in 0..<8 {
            transcript.append(UInt8(lSize & 0xFF))
            lSize >>= 8
        }
        let sampleCount = min(16, lookups.count)
        for i in 0..<sampleCount {
            let limbs = frToInt(lookups[i])
            for limb in limbs {
                var v = limb
                for _ in 0..<8 {
                    transcript.append(UInt8(v & 0xFF))
                    v >>= 8
                }
            }
        }
        let hash = blake3(transcript)
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            for j in 0..<8 {
                limbs[i] |= UInt64(hash[i * 8 + j]) << (j * 8)
            }
        }
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}
