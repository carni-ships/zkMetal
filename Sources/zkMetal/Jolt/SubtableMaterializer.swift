// SubtableMaterializer — Generate full lookup tables from JoltSubtable definitions
//
// Given a JoltSubtable and a chunk size, materializes the complete lookup table
// by evaluating the subtable function on every possible input. These materialized
// tables are used directly by the Lasso prover/verifier.
//
// For unary subtables (Identity, Truncate, SignExtend), the table has 2^C entries.
// For binary subtables (AND, OR, XOR, EQ, LT, LTU), the table has 2^(2C) entries,
// where the input packs two C-bit values as (x << C) | y.
//
// The materializeAll function generates all subtables needed for Jolt instruction
// decomposition, keyed by subtable name for efficient lookup during proving.
//
// References: Jolt (Arun et al. 2024) Section 4.1

import Foundation

// MARK: - SubtableMaterializer

/// Materializes JoltSubtable instances into full lookup table arrays.
public enum SubtableMaterializer {

    /// Materialize a single subtable into a full lookup table.
    ///
    /// Evaluates `subtable.evaluate(input:)` for every input in [0, tableSize).
    /// The resulting array can be used directly as a Lasso subtable.
    ///
    /// - Parameters:
    ///   - subtable: The subtable to materialize
    ///   - chunkBits: Number of bits per chunk (determines table size)
    /// - Returns: Array of evaluated outputs, indexed by input value
    public static func materialize(subtable: JoltSubtable, chunkBits: Int) -> [UInt64] {
        let size = subtable.tableSize
        var table = [UInt64](repeating: 0, count: size)
        for i in 0..<size {
            table[i] = subtable.evaluate(input: UInt64(i))
        }
        return table
    }

    /// Materialize all Jolt subtables needed for RV32IM instruction decomposition.
    ///
    /// Returns a dictionary keyed by subtable name, containing the full lookup table
    /// for each subtable type. This covers all subtables used by JoltInstructionDecomposer.
    ///
    /// - Parameter chunkBits: Number of bits per chunk (default 6 for 64-entry tables)
    /// - Returns: Dictionary mapping subtable name to materialized table
    public static func materializeAll(chunkBits: Int = 6) -> [String: [UInt64]] {
        var tables = [String: [UInt64]]()

        // Unary subtables
        let identity = IdentitySubtable(chunkBits: chunkBits)
        tables[identity.name] = materialize(subtable: identity, chunkBits: chunkBits)

        let trunc8 = TruncateSubtable(chunkBits: chunkBits, truncBits: 8)
        tables[trunc8.name] = materialize(subtable: trunc8, chunkBits: chunkBits)

        let trunc16 = TruncateSubtable(chunkBits: chunkBits, truncBits: 16)
        tables[trunc16.name] = materialize(subtable: trunc16, chunkBits: chunkBits)

        let signExt8 = SignExtendSubtable(chunkBits: chunkBits, fromBits: 8)
        tables[signExt8.name] = materialize(subtable: signExt8, chunkBits: chunkBits)

        let signExt16 = SignExtendSubtable(chunkBits: chunkBits, fromBits: 16)
        tables[signExt16.name] = materialize(subtable: signExt16, chunkBits: chunkBits)

        // Binary subtables
        let andST = AndSubtable(chunkBits: chunkBits)
        tables[andST.name] = materialize(subtable: andST, chunkBits: chunkBits)

        let orST = OrSubtable(chunkBits: chunkBits)
        tables[orST.name] = materialize(subtable: orST, chunkBits: chunkBits)

        let xorST = XorSubtable(chunkBits: chunkBits)
        tables[xorST.name] = materialize(subtable: xorST, chunkBits: chunkBits)

        let eqST = EQSubtable(chunkBits: chunkBits)
        tables[eqST.name] = materialize(subtable: eqST, chunkBits: chunkBits)

        let ltST = LTSubtable(chunkBits: chunkBits)
        tables[ltST.name] = materialize(subtable: ltST, chunkBits: chunkBits)

        let ltuST = LTUSubtable(chunkBits: chunkBits)
        tables[ltuST.name] = materialize(subtable: ltuST, chunkBits: chunkBits)

        let sllST = SllSubtable(chunkBits: chunkBits)
        tables[sllST.name] = materialize(subtable: sllST, chunkBits: chunkBits)

        let srlST = SrlSubtable(chunkBits: chunkBits)
        tables[srlST.name] = materialize(subtable: srlST, chunkBits: chunkBits)

        let sraST = SraSubtable(chunkBits: chunkBits)
        tables[sraST.name] = materialize(subtable: sraST, chunkBits: chunkBits)

        return tables
    }

    /// Materialize a subtable into Fr field elements for direct use with LassoEngine.
    ///
    /// - Parameters:
    ///   - subtable: The subtable to materialize
    ///   - chunkBits: Number of bits per chunk
    /// - Returns: Array of Fr field elements
    public static func materializeAsFr(subtable: JoltSubtable, chunkBits: Int) -> [Fr] {
        let size = subtable.tableSize
        var table = [Fr](repeating: Fr.zero, count: size)
        for i in 0..<size {
            table[i] = frFromInt(subtable.evaluate(input: UInt64(i)))
        }
        return table
    }

    /// Materialize all subtables as Fr field elements.
    ///
    /// - Parameter chunkBits: Number of bits per chunk
    /// - Returns: Dictionary mapping subtable name to Fr table
    public static func materializeAllAsFr(chunkBits: Int = 6) -> [String: [Fr]] {
        var tables = [String: [Fr]]()

        let subtables: [JoltSubtable] = [
            IdentitySubtable(chunkBits: chunkBits),
            TruncateSubtable(chunkBits: chunkBits, truncBits: 8),
            TruncateSubtable(chunkBits: chunkBits, truncBits: 16),
            SignExtendSubtable(chunkBits: chunkBits, fromBits: 8),
            SignExtendSubtable(chunkBits: chunkBits, fromBits: 16),
            AndSubtable(chunkBits: chunkBits),
            OrSubtable(chunkBits: chunkBits),
            XorSubtable(chunkBits: chunkBits),
            EQSubtable(chunkBits: chunkBits),
            LTSubtable(chunkBits: chunkBits),
            LTUSubtable(chunkBits: chunkBits),
            SllSubtable(chunkBits: chunkBits),
            SrlSubtable(chunkBits: chunkBits),
            SraSubtable(chunkBits: chunkBits),
        ]

        for st in subtables {
            tables[st.name] = materializeAsFr(subtable: st, chunkBits: chunkBits)
        }

        return tables
    }

    // MARK: - Table Statistics

    /// Summary of materialized table sizes.
    public struct TableStats {
        public let chunkBits: Int
        public let unaryTableSize: Int     // 2^C entries
        public let binaryTableSize: Int    // 2^(2C) entries
        public let totalSubtables: Int
        public let totalEntries: Int
        public let totalMemoryBytes: Int   // Approximate, assuming 8 bytes per UInt64

        public var description: String {
            return """
            Jolt Subtable Stats (C=\(chunkBits)):
              Unary table size:  \(unaryTableSize) entries
              Binary table size: \(binaryTableSize) entries
              Total subtables:   \(totalSubtables)
              Total entries:     \(totalEntries)
              Memory:            \(totalMemoryBytes / 1024) KiB
            """
        }
    }

    /// Compute statistics for the materialized tables at a given chunk size.
    public static func stats(chunkBits: Int = 6) -> TableStats {
        let unarySize = 1 << chunkBits
        let binarySize = 1 << (2 * chunkBits)
        let numUnary = 5   // identity, truncate_8, truncate_16, sign_extend_8, sign_extend_16
        let numBinary = 9  // and, or, xor, eq, lt, ltu, sll, srl, sra
        let totalEntries = numUnary * unarySize + numBinary * binarySize

        return TableStats(
            chunkBits: chunkBits,
            unaryTableSize: unarySize,
            binaryTableSize: binarySize,
            totalSubtables: numUnary + numBinary,
            totalEntries: totalEntries,
            totalMemoryBytes: totalEntries * 8
        )
    }
}
