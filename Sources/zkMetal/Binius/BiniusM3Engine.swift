// Binius M3 Arithmetization Engine
//
// M3 is a table-and-channel constraint model for Binius binary-field proving.
//
// Key concepts:
//   - Tables: Named collections of columns over binary tower fields
//   - Channels: Multiset communication between tables via push/pull ops
//   - Zero constraints: Polynomial expressions that must vanish on every row
//   - Channel balance: enforced via LogUp (sum of 1/(alpha - pushed) = sum of 1/(alpha - pulled))
//
// The binary tower allows mixing columns of different bit-widths (1, 2, 4, 8, ..., 128)
// within the same table, with smaller fields embedded into larger ones via the tower.
//
// References:
//   - Binius (Irreducible): binary-field SNARK with M3 arithmetization
//   - LogUp (Haboeck 2022): logarithmic derivative lookup argument

import Foundation

// MARK: - M3 Column

/// A column in an M3 table. Each column stores values over a binary tower field
/// at a specific bit-width level.
public struct M3Column {
    /// Column identifier within the table
    public let name: String

    /// Bit-width of elements: 1, 2, 4, 8, 16, 32, 64, or 128.
    /// Determines which binary tower level this column operates in.
    public let bitWidth: Int

    /// Column values stored as BinaryTower128 (the universal embedding target).
    /// Smaller bit-width values are embedded via the canonical tower embedding.
    public var values: [BinaryTower128]

    /// Valid bit-widths for binary tower columns
    public static let validBitWidths: Set<Int> = [1, 2, 4, 8, 16, 32, 64, 128]

    public init(name: String, bitWidth: Int, values: [BinaryTower128] = []) {
        precondition(M3Column.validBitWidths.contains(bitWidth),
                     "Invalid bitWidth \(bitWidth); must be a power of 2 in [1..128]")
        self.name = name
        self.bitWidth = bitWidth
        self.values = values
    }

    /// Create a column from UInt8 values (GF(2^8) elements)
    public static func fromGF8(name: String, values: [UInt8]) -> M3Column {
        let embedded = values.map { BinaryTowerEmbed.gf8Into128($0) }
        return M3Column(name: name, bitWidth: 8, values: embedded)
    }

    /// Create a column from single-bit values (GF(2) elements)
    public static func fromBits(name: String, values: [Bool]) -> M3Column {
        let embedded = values.map { BinaryTower128(lo: $0 ? 1 : 0, hi: 0) }
        return M3Column(name: name, bitWidth: 1, values: embedded)
    }

    /// Create a column from UInt64 values (GF(2^64) elements)
    public static func fromGF64(name: String, values: [UInt64]) -> M3Column {
        let embedded = values.map { BinaryTower128(lo: $0, hi: 0) }
        return M3Column(name: name, bitWidth: 64, values: embedded)
    }

    /// Check that all values fit within the declared bit-width
    public func validateBitWidth() -> Bool {
        let mask: UInt64
        switch bitWidth {
        case 1: mask = 1
        case 2: mask = 0x3
        case 4: mask = 0xF
        case 8: mask = 0xFF
        case 16: mask = 0xFFFF
        case 32: mask = 0xFFFF_FFFF
        case 64: return values.allSatisfy { $0.hi == 0 }
        case 128: return true
        default: return false
        }
        return values.allSatisfy { $0.hi == 0 && $0.lo <= mask }
    }
}

// MARK: - M3 Constraint Expression

/// An expression in the M3 constraint language, evaluated over binary tower fields.
/// Constraints are polynomial expressions that must evaluate to zero on every row.
public indirect enum M3Expr {
    /// Reference to a column value at the current row
    case column(String)

    /// Reference to a column value shifted by `offset` rows (for transition constraints)
    case shifted(column: String, offset: Int)

    /// A constant binary tower field element
    case constant(BinaryTower128)

    /// Addition (XOR in binary fields)
    case add(M3Expr, M3Expr)

    /// Multiplication
    case mul(M3Expr, M3Expr)

    /// Convenience: sum of expressions
    public static func sum(_ exprs: [M3Expr]) -> M3Expr {
        guard let first = exprs.first else { return .constant(.zero) }
        return exprs.dropFirst().reduce(first) { .add($0, $1) }
    }

    /// Convenience: product of expressions
    public static func product(_ exprs: [M3Expr]) -> M3Expr {
        guard let first = exprs.first else { return .constant(.one) }
        return exprs.dropFirst().reduce(first) { .mul($0, $1) }
    }
}

// MARK: - M3 Zero Constraint

/// A named constraint that must evaluate to zero on every row of its table.
public struct M3ZeroConstraint {
    public let name: String
    public let expr: M3Expr

    public init(name: String, expr: M3Expr) {
        self.name = name
        self.expr = expr
    }
}

// MARK: - M3 Channel Operations

/// Channels are the communication mechanism between tables in M3.
/// Tables push tuples of field elements into channels and pull from channels.
/// The multiset of all pushed values must equal the multiset of all pulled values.
public enum M3ChannelOp {
    /// Push a tuple of column values into a named channel (one tuple per row)
    case push(channel: String, columns: [String])

    /// Pull a tuple of column values from a named channel (one tuple per row)
    case pull(channel: String, columns: [String])

    public var channelName: String {
        switch self {
        case .push(let ch, _), .pull(let ch, _): return ch
        }
    }

    public var columnNames: [String] {
        switch self {
        case .push(_, let cols), .pull(_, let cols): return cols
        }
    }

    public var isPush: Bool {
        if case .push = self { return true }
        return false
    }
}

// MARK: - M3 Table

/// A table in the M3 arithmetization. Contains typed columns, zero constraints,
/// and channel operations (push/pull).
public struct M3Table {
    public let name: String
    public var columns: [M3Column]
    public var zeroConstraints: [M3ZeroConstraint]
    public var channelOps: [M3ChannelOp]

    /// Number of rows (must be a power of 2). All columns share this row count.
    public var rowCount: Int {
        columns.first?.values.count ?? 0
    }

    public init(name: String,
                columns: [M3Column] = [],
                zeroConstraints: [M3ZeroConstraint] = [],
                channelOps: [M3ChannelOp] = []) {
        self.name = name
        self.columns = columns
        self.zeroConstraints = zeroConstraints
        self.channelOps = channelOps
    }

    /// Add a column to this table
    public mutating func addColumn(_ col: M3Column) {
        columns.append(col)
    }

    /// Add a zero constraint
    public mutating func addZeroConstraint(_ constraint: M3ZeroConstraint) {
        zeroConstraints.append(constraint)
    }

    /// Add a channel operation
    public mutating func addChannelOp(_ op: M3ChannelOp) {
        channelOps.append(op)
    }

    /// Look up a column by name, returns nil if not found
    public func column(named name: String) -> M3Column? {
        columns.first { $0.name == name }
    }

    /// Validate table structure
    public func validate() -> [String] {
        var errors = [String]()

        if columns.isEmpty {
            errors.append("Table '\(name)' has no columns")
            return errors
        }

        let n = columns[0].values.count
        if n == 0 {
            errors.append("Table '\(name)' has zero rows")
            return errors
        }

        // Row count must be power of 2
        if n & (n - 1) != 0 {
            errors.append("Table '\(name)' row count \(n) is not a power of 2")
        }

        // All columns must have same row count
        for col in columns {
            if col.values.count != n {
                errors.append("Table '\(name)' column '\(col.name)' has \(col.values.count) rows, expected \(n)")
            }
        }

        // Validate bit-widths
        for col in columns {
            if !col.validateBitWidth() {
                errors.append("Table '\(name)' column '\(col.name)' has values exceeding declared bit-width \(col.bitWidth)")
            }
        }

        // Validate channel ops reference valid columns
        let colNames = Set(columns.map { $0.name })
        for op in channelOps {
            for colName in op.columnNames {
                if !colNames.contains(colName) {
                    errors.append("Table '\(name)' channel op references unknown column '\(colName)'")
                }
            }
        }

        // Validate zero constraints reference valid columns
        for constraint in zeroConstraints {
            let refCols = collectColumnRefs(constraint.expr)
            for colName in refCols {
                if !colNames.contains(colName) {
                    errors.append("Table '\(name)' constraint '\(constraint.name)' references unknown column '\(colName)'")
                }
            }
        }

        return errors
    }
}

// MARK: - M3 Witness

/// A complete M3 witness: populated tables with all column values filled in.
public struct M3Witness {
    public let tables: [M3Table]

    /// All channel names used across tables
    public var channelNames: Set<String> {
        var names = Set<String>()
        for table in tables {
            for op in table.channelOps {
                names.insert(op.channelName)
            }
        }
        return names
    }

    public init(tables: [M3Table]) {
        self.tables = tables
    }
}

// MARK: - M3 Compiled Circuit (output of compilation)

/// The result of compiling an M3 witness into polynomial commitments
/// suitable for a Binius STARK proof.
public struct BiniusM3CompiledCircuit {
    /// Flattened column polynomials (multilinear extensions over binary hypercube)
    public let columnPolynomials: [M3ColumnPoly]

    /// Channel balance constraints (LogUp fractional sums)
    public let channelConstraints: [M3ChannelConstraint]

    /// Zero constraint polynomials (should vanish on evaluation domain)
    public let zeroConstraintPolys: [M3ZeroConstraintPoly]

    /// Total number of binary tower field elements across all columns
    public let totalElements: Int

    public init(columnPolynomials: [M3ColumnPoly],
                channelConstraints: [M3ChannelConstraint],
                zeroConstraintPolys: [M3ZeroConstraintPoly],
                totalElements: Int) {
        self.columnPolynomials = columnPolynomials
        self.channelConstraints = channelConstraints
        self.zeroConstraintPolys = zeroConstraintPolys
        self.totalElements = totalElements
    }
}

/// A column polynomial: multilinear extension of column values over the binary hypercube
public struct M3ColumnPoly {
    public let tableName: String
    public let columnName: String
    public let bitWidth: Int
    /// Number of variables in the multilinear polynomial (log2 of row count)
    public let numVars: Int
    /// Evaluations on the hypercube {0,1}^numVars
    public let evaluations: [BinaryTower128]
}

/// Channel balance constraint: LogUp fractional sum identity
public struct M3ChannelConstraint {
    public let channelName: String
    /// (tableName, columnNames, isPush) for each participating operation
    public let participants: [(tableName: String, columns: [String], isPush: Bool)]
}

/// Compiled zero constraint polynomial
public struct M3ZeroConstraintPoly {
    public let tableName: String
    public let constraintName: String
    /// Evaluations of the constraint expression on the table's rows
    public let evaluations: [BinaryTower128]
}

// MARK: - M3 Engine

/// The Binius M3 arithmetization engine.
///
/// Provides:
///   1. Witness generation: fill table columns from user-provided data
///   2. Constraint checking: verify all zero constraints and channel balances
///   3. Compilation: convert M3 tables to polynomial commitments for a Binius STARK
public class BiniusM3Engine {

    /// Enable timing instrumentation
    public var profile = false

    public init() {}

    // MARK: - Witness Generation

    /// Generate a complete M3 witness from table definitions.
    /// Validates structure and returns the witness if valid.
    public func generateWitness(tables: [M3Table]) throws -> M3Witness {
        // Validate all tables
        var allErrors = [String]()
        for table in tables {
            allErrors.append(contentsOf: table.validate())
        }

        if !allErrors.isEmpty {
            throw M3Error.validationFailed(allErrors)
        }

        // Check that table names are unique
        let names = tables.map { $0.name }
        let uniqueNames = Set(names)
        if names.count != uniqueNames.count {
            throw M3Error.duplicateTableName
        }

        return M3Witness(tables: tables)
    }

    // MARK: - Constraint Checking

    /// Verify all zero constraints and channel balances in the witness.
    ///
    /// Returns true if:
    ///   1. Every zero constraint evaluates to zero on every row
    ///   2. For every channel, the multiset of pushed tuples equals the multiset of pulled tuples
    public func checkConstraints(witness: M3Witness) throws -> Bool {
        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        // Check zero constraints for each table
        for table in witness.tables {
            let ok = try checkZeroConstraints(table: table)
            if !ok { return false }
        }

        if profile {
            let _t = CFAbsoluteTimeGetCurrent()
            fputs(String(format: "  [m3] zero constraints: %.2f ms\n", (_t - _t0) * 1000), stderr)
        }

        // Check channel balances
        let channelOk = try checkChannelBalances(witness: witness)
        if !channelOk { return false }

        if profile {
            let total = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [m3] total constraint check: %.2f ms\n", total), stderr)
        }

        return true
    }

    /// Check that all zero constraints in a table evaluate to zero on every row.
    private func checkZeroConstraints(table: M3Table) throws -> Bool {
        let n = table.rowCount
        guard n > 0 else { return true }

        // Build column lookup
        var colMap = [String: M3Column]()
        for col in table.columns {
            colMap[col.name] = col
        }

        for constraint in table.zeroConstraints {
            for row in 0..<n {
                let val = try evaluateExpr(constraint.expr, colMap: colMap, row: row, rowCount: n)
                if !val.isZero {
                    throw M3Error.constraintViolation(
                        table: table.name,
                        constraint: constraint.name,
                        row: row,
                        value: val
                    )
                }
            }
        }
        return true
    }

    /// Evaluate an M3 expression at a specific row of a table.
    private func evaluateExpr(_ expr: M3Expr, colMap: [String: M3Column],
                              row: Int, rowCount: Int) throws -> BinaryTower128 {
        switch expr {
        case .column(let name):
            guard let col = colMap[name] else {
                throw M3Error.unknownColumn(name)
            }
            return col.values[row]

        case .shifted(let name, let offset):
            guard let col = colMap[name] else {
                throw M3Error.unknownColumn(name)
            }
            // Wrap around cyclically
            let targetRow = ((row + offset) % rowCount + rowCount) % rowCount
            return col.values[targetRow]

        case .constant(let val):
            return val

        case .add(let lhs, let rhs):
            let a = try evaluateExpr(lhs, colMap: colMap, row: row, rowCount: rowCount)
            let b = try evaluateExpr(rhs, colMap: colMap, row: row, rowCount: rowCount)
            return a + b  // XOR

        case .mul(let lhs, let rhs):
            let a = try evaluateExpr(lhs, colMap: colMap, row: row, rowCount: rowCount)
            let b = try evaluateExpr(rhs, colMap: colMap, row: row, rowCount: rowCount)
            return a * b
        }
    }

    /// Check that all channels are balanced: pushed multiset == pulled multiset.
    ///
    /// Uses LogUp-style verification: for a random challenge alpha,
    ///   sum_{pushed tuples t} 1/(alpha - hash(t)) == sum_{pulled tuples t} 1/(alpha - hash(t))
    ///
    /// We also do an exact multiset equality check for correctness.
    private func checkChannelBalances(witness: M3Witness) throws -> Bool {
        // Collect all push/pull operations per channel
        var pushes = [String: [(table: M3Table, columns: [String])]]()
        var pulls  = [String: [(table: M3Table, columns: [String])]]()

        for table in witness.tables {
            for op in table.channelOps {
                let entry = (table: table, columns: op.columnNames)
                switch op {
                case .push(let ch, _):
                    pushes[ch, default: []].append(entry)
                case .pull(let ch, _):
                    pulls[ch, default: []].append(entry)
                }
            }
        }

        // Check each channel
        let allChannels = Set(pushes.keys).union(pulls.keys)
        for channel in allChannels {
            let pushOps = pushes[channel] ?? []
            let pullOps = pulls[channel] ?? []

            // Collect all pushed tuples
            var pushedTuples = [[BinaryTower128]]()
            for (table, colNames) in pushOps {
                let n = table.rowCount
                var colMap = [String: M3Column]()
                for col in table.columns { colMap[col.name] = col }
                for row in 0..<n {
                    let tuple = colNames.map { colMap[$0]!.values[row] }
                    pushedTuples.append(tuple)
                }
            }

            // Collect all pulled tuples
            var pulledTuples = [[BinaryTower128]]()
            for (table, colNames) in pullOps {
                let n = table.rowCount
                var colMap = [String: M3Column]()
                for col in table.columns { colMap[col.name] = col }
                for row in 0..<n {
                    let tuple = colNames.map { colMap[$0]!.values[row] }
                    pulledTuples.append(tuple)
                }
            }

            // Multiset equality: sort both and compare
            let sortedPush = pushedTuples.sorted(by: tupleLessThan)
            let sortedPull = pulledTuples.sorted(by: tupleLessThan)

            if sortedPush.count != sortedPull.count {
                throw M3Error.channelImbalance(
                    channel: channel,
                    pushCount: sortedPush.count,
                    pullCount: sortedPull.count
                )
            }

            for i in 0..<sortedPush.count {
                if !tuplesEqual(sortedPush[i], sortedPull[i]) {
                    throw M3Error.channelMismatch(channel: channel, index: i)
                }
            }

            // LogUp verification: random challenge alpha, check fractional sum identity
            let alpha = deriveChannelChallenge(channel: channel, witness: witness)
            let logupOk = try verifyChannelLogUp(
                channel: channel, pushedTuples: pushedTuples,
                pulledTuples: pulledTuples, alpha: alpha)
            if !logupOk { return false }
        }

        return true
    }

    /// Verify channel balance via LogUp fractional sums.
    /// sum of 1/(alpha - hash(pushed_tuple)) == sum of 1/(alpha - hash(pulled_tuple))
    private func verifyChannelLogUp(channel: String,
                                     pushedTuples: [[BinaryTower128]],
                                     pulledTuples: [[BinaryTower128]],
                                     alpha: BinaryTower128) throws -> Bool {
        // Hash tuples to single field elements, then compute fractional sums
        var pushSum = BinaryTower128.zero
        for tuple in pushedTuples {
            let h = hashTuple(tuple)
            let denom = alpha + h  // alpha - h = alpha + h in char 2
            if denom.isZero {
                // Extremely unlikely with random alpha, but handle gracefully
                throw M3Error.logUpZeroDenominator(channel: channel)
            }
            let inv = denom.inverse()
            pushSum = pushSum + inv
        }

        var pullSum = BinaryTower128.zero
        for tuple in pulledTuples {
            let h = hashTuple(tuple)
            let denom = alpha + h
            if denom.isZero {
                throw M3Error.logUpZeroDenominator(channel: channel)
            }
            let inv = denom.inverse()
            pullSum = pullSum + inv
        }

        // In char 2, sum equality check: pushSum + pullSum should be zero
        // (since subtraction = addition in char 2)
        let diff = pushSum + pullSum
        return diff.isZero
    }

    /// Hash a tuple of BinaryTower128 elements to a single element.
    /// Uses a simple algebraic hash: fold with multiplication by a fixed constant + XOR.
    private func hashTuple(_ tuple: [BinaryTower128]) -> BinaryTower128 {
        // Use a non-trivial mixing constant to avoid collisions
        let mixer = BinaryTower128(lo: 0xDEAD_BEEF_CAFE_BABE, hi: 0x1337_C0DE_F00D_FACE)
        var acc = BinaryTower128.zero
        for elem in tuple {
            acc = (acc * mixer) + elem
        }
        return acc
    }

    /// Derive a random challenge for channel balance verification via Fiat-Shamir.
    private func deriveChannelChallenge(channel: String, witness: M3Witness) -> BinaryTower128 {
        // Domain separator: "M3CH" + channel name + first few column values
        var seed: UInt64 = 0x4D33_4348  // "M3CH"
        for byte in channel.utf8 {
            seed = seed &* 31 &+ UInt64(byte)
        }
        // Mix in some witness data for binding
        for table in witness.tables {
            for col in table.columns {
                if let first = col.values.first {
                    seed ^= first.lo
                }
            }
        }
        // Generate a pseudorandom 128-bit challenge
        let lo = seed &* 0x9E37_79B9_7F4A_7C15 &+ 0x6A09_E667_F3BC_C908
        let hi = seed &* 0x517C_C1B7_2722_0A95 &+ 0xBB67_AE85_84CA_A73B
        return BinaryTower128(lo: lo, hi: hi)
    }

    // MARK: - Compilation to Polynomial Commitments

    /// Compile an M3 witness into a circuit suitable for Binius STARK proving.
    ///
    /// This converts each column into a multilinear polynomial (evaluations on
    /// the binary hypercube), prepares channel constraints as LogUp relations,
    /// and evaluates zero constraints into vanishing polynomials.
    public func compile(witness: M3Witness) throws -> BiniusM3CompiledCircuit {
        let _t0 = profile ? CFAbsoluteTimeGetCurrent() : 0

        var columnPolys = [M3ColumnPoly]()
        var zeroConstraintPolys = [M3ZeroConstraintPoly]()
        var totalElements = 0

        for table in witness.tables {
            let n = table.rowCount
            guard n > 0 else { continue }
            let numVars = Int(log2(Double(n)))

            // Column polynomials: multilinear extensions on {0,1}^numVars
            for col in table.columns {
                let poly = M3ColumnPoly(
                    tableName: table.name,
                    columnName: col.name,
                    bitWidth: col.bitWidth,
                    numVars: numVars,
                    evaluations: col.values
                )
                columnPolys.append(poly)
                totalElements += n
            }

            // Zero constraint evaluation polynomials
            var colMap = [String: M3Column]()
            for col in table.columns { colMap[col.name] = col }

            for constraint in table.zeroConstraints {
                var evals = [BinaryTower128](repeating: .zero, count: n)
                for row in 0..<n {
                    evals[row] = try evaluateExpr(
                        constraint.expr, colMap: colMap, row: row, rowCount: n)
                }
                let constraintPoly = M3ZeroConstraintPoly(
                    tableName: table.name,
                    constraintName: constraint.name,
                    evaluations: evals
                )
                zeroConstraintPolys.append(constraintPoly)
            }
        }

        // Channel constraints
        var channelConstraints = [M3ChannelConstraint]()
        var channelParticipants = [String: [(tableName: String, columns: [String], isPush: Bool)]]()

        for table in witness.tables {
            for op in table.channelOps {
                let participant = (tableName: table.name, columns: op.columnNames, isPush: op.isPush)
                channelParticipants[op.channelName, default: []].append(participant)
            }
        }

        for (channelName, participants) in channelParticipants {
            channelConstraints.append(M3ChannelConstraint(
                channelName: channelName,
                participants: participants
            ))
        }

        if profile {
            let elapsed = (CFAbsoluteTimeGetCurrent() - _t0) * 1000
            fputs(String(format: "  [m3] compile: %.2f ms (%d columns, %d elements)\n",
                         elapsed, columnPolys.count, totalElements), stderr)
        }

        return BiniusM3CompiledCircuit(
            columnPolynomials: columnPolys,
            channelConstraints: channelConstraints,
            zeroConstraintPolys: zeroConstraintPolys,
            totalElements: totalElements
        )
    }

    // MARK: - Packing Constraints

    /// Verify packing constraints: a high-bit-width column can be decomposed
    /// into lower-bit-width columns via the binary tower embedding.
    ///
    /// For example, a 16-bit column c can be packed from two 8-bit columns (lo, hi):
    ///   c[row] = lo[row] + hi[row] * X   (where X is the GF(2^16) extension variable)
    public func checkPacking(table: M3Table, packedColumn: String,
                              componentColumns: [String]) throws -> Bool {
        guard let packed = table.column(named: packedColumn) else {
            throw M3Error.unknownColumn(packedColumn)
        }

        let components = try componentColumns.map { name -> M3Column in
            guard let col = table.column(named: name) else {
                throw M3Error.unknownColumn(name)
            }
            return col
        }

        let n = table.rowCount

        // Number of components must be packed.bitWidth / component.bitWidth
        guard let firstComp = components.first else { return true }
        let ratio = packed.bitWidth / firstComp.bitWidth
        guard components.count == ratio else {
            throw M3Error.packingMismatch(
                packed: packedColumn, expected: ratio, got: components.count)
        }

        // Verify: packed value = sum of component[i] * basis_element[i]
        // In the binary tower, the basis elements for GF(2^{2k}) over GF(2^k) are {1, X}
        // For general packing, we use positional embedding
        for row in 0..<n {
            var reconstructed = BinaryTower128.zero
            for (i, comp) in components.enumerated() {
                let shifted = shiftByBits(comp.values[row], bits: i * firstComp.bitWidth)
                reconstructed = reconstructed + shifted
            }
            if reconstructed != packed.values[row] {
                return false
            }
        }

        return true
    }

    /// Shift a binary tower element left by `bits` positions within GF(2^128).
    /// This is equivalent to multiplying by the basis element at that position.
    private func shiftByBits(_ val: BinaryTower128, bits: Int) -> BinaryTower128 {
        if bits == 0 { return val }
        if bits >= 64 {
            return BinaryTower128(lo: 0, hi: val.lo << (bits - 64))
        }
        let lo = val.lo << bits
        let hi = (val.hi << bits) | (val.lo >> (64 - bits))
        return BinaryTower128(lo: lo, hi: hi)
    }

    // MARK: - Helpers

    /// Compare two BinaryTower128 tuples lexicographically
    private func tupleLessThan(_ a: [BinaryTower128], _ b: [BinaryTower128]) -> Bool {
        for i in 0..<min(a.count, b.count) {
            if a[i].hi != b[i].hi { return a[i].hi < b[i].hi }
            if a[i].lo != b[i].lo { return a[i].lo < b[i].lo }
        }
        return a.count < b.count
    }

    /// Check if two tuples of BinaryTower128 are equal
    private func tuplesEqual(_ a: [BinaryTower128], _ b: [BinaryTower128]) -> Bool {
        guard a.count == b.count else { return false }
        for i in 0..<a.count {
            if a[i] != b[i] { return false }
        }
        return true
    }
}

// MARK: - Expression Helpers

/// Collect all column names referenced by an expression
func collectColumnRefs(_ expr: M3Expr) -> Set<String> {
    switch expr {
    case .column(let name):
        return [name]
    case .shifted(let name, _):
        return [name]
    case .constant:
        return []
    case .add(let a, let b), .mul(let a, let b):
        return collectColumnRefs(a).union(collectColumnRefs(b))
    }
}

// MARK: - Errors

public enum M3Error: Error, CustomStringConvertible {
    case validationFailed([String])
    case duplicateTableName
    case unknownColumn(String)
    case constraintViolation(table: String, constraint: String, row: Int, value: BinaryTower128)
    case channelImbalance(channel: String, pushCount: Int, pullCount: Int)
    case channelMismatch(channel: String, index: Int)
    case logUpZeroDenominator(channel: String)
    case packingMismatch(packed: String, expected: Int, got: Int)

    public var description: String {
        switch self {
        case .validationFailed(let errors):
            return "M3 validation failed:\n" + errors.joined(separator: "\n")
        case .duplicateTableName:
            return "Duplicate table name in M3 witness"
        case .unknownColumn(let name):
            return "Unknown column '\(name)'"
        case .constraintViolation(let table, let constraint, let row, let value):
            return "Constraint '\(constraint)' violated in table '\(table)' at row \(row): \(value)"
        case .channelImbalance(let channel, let pushCount, let pullCount):
            return "Channel '\(channel)' imbalance: \(pushCount) pushed vs \(pullCount) pulled"
        case .channelMismatch(let channel, let index):
            return "Channel '\(channel)' multiset mismatch at sorted index \(index)"
        case .logUpZeroDenominator(let channel):
            return "LogUp zero denominator in channel '\(channel)'"
        case .packingMismatch(let packed, let expected, let got):
            return "Packing '\(packed)': expected \(expected) components, got \(got)"
        }
    }
}

// MARK: - Table Builder DSL

/// Convenience builder for constructing M3 tables
public class M3TableBuilder {
    private var name: String
    private var columns: [M3Column] = []
    private var constraints: [M3ZeroConstraint] = []
    private var channelOps: [M3ChannelOp] = []

    public init(name: String) {
        self.name = name
    }

    /// Add a column with GF(2^8) values
    @discardableResult
    public func addGF8Column(_ name: String, values: [UInt8]) -> M3TableBuilder {
        columns.append(M3Column.fromGF8(name: name, values: values))
        return self
    }

    /// Add a column with bit values
    @discardableResult
    public func addBitColumn(_ name: String, values: [Bool]) -> M3TableBuilder {
        columns.append(M3Column.fromBits(name: name, values: values))
        return self
    }

    /// Add a column with GF(2^64) values
    @discardableResult
    public func addGF64Column(_ name: String, values: [UInt64]) -> M3TableBuilder {
        columns.append(M3Column.fromGF64(name: name, values: values))
        return self
    }

    /// Add a column with GF(2^128) values
    @discardableResult
    public func addGF128Column(_ name: String, values: [BinaryTower128]) -> M3TableBuilder {
        columns.append(M3Column(name: name, bitWidth: 128, values: values))
        return self
    }

    /// Add a zero constraint: expr must evaluate to zero on every row
    @discardableResult
    public func addZeroConstraint(_ name: String, _ expr: M3Expr) -> M3TableBuilder {
        constraints.append(M3ZeroConstraint(name: name, expr: expr))
        return self
    }

    /// Push column values into a channel
    @discardableResult
    public func push(channel: String, columns: [String]) -> M3TableBuilder {
        channelOps.append(.push(channel: channel, columns: columns))
        return self
    }

    /// Pull column values from a channel
    @discardableResult
    public func pull(channel: String, columns: [String]) -> M3TableBuilder {
        channelOps.append(.pull(channel: channel, columns: columns))
        return self
    }

    /// Build the table
    public func build() -> M3Table {
        M3Table(name: name, columns: columns,
                zeroConstraints: constraints, channelOps: channelOps)
    }
}
