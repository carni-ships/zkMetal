// Tests for Binius M3 arithmetization engine

import Foundation
import zkMetal

func runBiniusM3Tests() {
    // Initialize binary tower tables (required for NEON ops)
    BinaryTowerNeon.initialize()

    suite("BiniusM3 — Table Validation")
    testM3TableValidation()

    suite("BiniusM3 — Zero Constraints")
    testM3ZeroConstraints()
    testM3ShiftConstraints()

    suite("BiniusM3 — Channel Balance")
    testM3ChannelPushPull()
    testM3ChannelMultiTable()
    testM3ChannelImbalance()

    suite("BiniusM3 — Packing Constraints")
    testM3PackingGF8ToGF16()

    suite("BiniusM3 — Compilation")
    testM3Compile()

    suite("BiniusM3 — Table Builder DSL")
    testM3TableBuilder()

    suite("BiniusM3 — XOR Circuit Example")
    testM3XORCircuit()
}

// MARK: - Table Validation

private func testM3TableValidation() {
    let engine = BiniusM3Engine()

    // Valid table: 4 rows, 2 columns
    let col_a = M3Column.fromGF8(name: "a", values: [1, 2, 3, 4])
    let col_b = M3Column.fromGF8(name: "b", values: [5, 6, 7, 8])
    let table = M3Table(name: "test", columns: [col_a, col_b])

    let errors = table.validate()
    expect(errors.isEmpty, "Valid table has no errors: \(errors)")

    // Invalid: non-power-of-2 row count
    let col_bad = M3Column.fromGF8(name: "c", values: [1, 2, 3])
    let badTable = M3Table(name: "bad", columns: [col_bad])
    let badErrors = badTable.validate()
    expect(!badErrors.isEmpty, "Non-power-of-2 rows should fail validation")

    // Invalid: mismatched row counts
    let col_d = M3Column.fromGF8(name: "d", values: [1, 2, 3, 4])
    let col_e = M3Column.fromGF8(name: "e", values: [1, 2])
    let mismatch = M3Table(name: "mismatch", columns: [col_d, col_e])
    let mismatchErrors = mismatch.validate()
    expect(!mismatchErrors.isEmpty, "Mismatched row counts should fail")

    // Witness generation should fail on invalid tables
    do {
        _ = try engine.generateWitness(tables: [badTable])
        expect(false, "Should have thrown for invalid table")
    } catch {
        expect(true, "Correctly threw for invalid table")
    }

    // Witness generation should succeed for valid tables
    do {
        let witness = try engine.generateWitness(tables: [table])
        expect(witness.tables.count == 1, "Witness has 1 table")
    } catch {
        expect(false, "Should not throw for valid table: \(error)")
    }
}

// MARK: - Zero Constraints

private func testM3ZeroConstraints() {
    let engine = BiniusM3Engine()

    // Table where a + b = c (XOR constraint)
    // In GF(2^8): 0x53 ^ 0xCA = 0x99
    let a_vals: [UInt8] = [0x53, 0x10, 0xFF, 0x00]
    let b_vals: [UInt8] = [0xCA, 0x20, 0xFF, 0x00]
    let c_vals: [UInt8] = [0x99, 0x30, 0x00, 0x00]  // a XOR b

    let col_a = M3Column.fromGF8(name: "a", values: a_vals)
    let col_b = M3Column.fromGF8(name: "b", values: b_vals)
    let col_c = M3Column.fromGF8(name: "c", values: c_vals)

    // Constraint: a + b + c = 0 (in char 2, a + b = c means a + b + c = 0)
    let constraint = M3ZeroConstraint(
        name: "xor_check",
        expr: .add(.add(.column("a"), .column("b")), .column("c"))
    )

    let table = M3Table(name: "xor_table",
                         columns: [col_a, col_b, col_c],
                         zeroConstraints: [constraint])

    do {
        let witness = try engine.generateWitness(tables: [table])
        let ok = try engine.checkConstraints(witness: witness)
        expect(ok, "XOR constraint should pass")
    } catch {
        expect(false, "XOR constraint check failed: \(error)")
    }

    // Now test with wrong values
    let c_wrong: [UInt8] = [0x99, 0x31, 0x00, 0x00]  // row 1 is wrong
    let col_c_wrong = M3Column.fromGF8(name: "c", values: c_wrong)
    let bad_table = M3Table(name: "bad_xor",
                             columns: [col_a, col_b, col_c_wrong],
                             zeroConstraints: [constraint])
    do {
        let witness = try engine.generateWitness(tables: [bad_table])
        _ = try engine.checkConstraints(witness: witness)
        expect(false, "Should have thrown for violated constraint")
    } catch let error as M3Error {
        if case .constraintViolation(_, _, let row, _) = error {
            expect(row == 1, "Violation should be at row 1, got \(row)")
        } else {
            expect(false, "Wrong error type: \(error)")
        }
    } catch {
        expect(false, "Unexpected error type: \(error)")
    }
}

private func testM3ShiftConstraints() {
    let engine = BiniusM3Engine()

    // Fibonacci-like: a[i+1] = a[i] + b[i], b[i+1] = a[i]
    // Use GF(2^8) values (XOR addition). Must satisfy cyclic wrap-around.
    // All zeros is the simplest cyclic solution:
    let a_vals: [UInt8] = [0, 0, 0, 0]
    let b_vals: [UInt8] = [0, 0, 0, 0]

    let col_a = M3Column.fromGF8(name: "a", values: a_vals)
    let col_b = M3Column.fromGF8(name: "b", values: b_vals)

    // Constraint 1: a[next] = a[current] + b[current]
    // => a[next] + a + b = 0
    let fib_constraint = M3ZeroConstraint(
        name: "fib_a",
        expr: .add(.add(.shifted(column: "a", offset: 1), .column("a")), .column("b"))
    )

    // Constraint 2: b[next] = a[current]
    // => b[next] + a = 0
    let shift_constraint = M3ZeroConstraint(
        name: "shift_b",
        expr: .add(.shifted(column: "b", offset: 1), .column("a"))
    )

    let table = M3Table(name: "fib",
                         columns: [col_a, col_b],
                         zeroConstraints: [fib_constraint, shift_constraint])

    do {
        let witness = try engine.generateWitness(tables: [table])
        let ok = try engine.checkConstraints(witness: witness)
        expect(ok, "Fibonacci shift constraints should pass")
    } catch {
        expect(false, "Fibonacci constraint failed: \(error)")
    }
}

// MARK: - Channel Balance

private func testM3ChannelPushPull() {
    let engine = BiniusM3Engine()

    // Producer table pushes values, consumer table pulls same values
    let vals: [UInt8] = [0x10, 0x20, 0x30, 0x40]

    let producer = M3Table(
        name: "producer",
        columns: [M3Column.fromGF8(name: "out", values: vals)],
        channelOps: [.push(channel: "pipe", columns: ["out"])]
    )

    // Consumer pulls same values (possibly reordered -- multiset equality)
    let consumer_vals: [UInt8] = [0x30, 0x10, 0x40, 0x20]  // reordered
    let consumer = M3Table(
        name: "consumer",
        columns: [M3Column.fromGF8(name: "in", values: consumer_vals)],
        channelOps: [.pull(channel: "pipe", columns: ["in"])]
    )

    do {
        let witness = try engine.generateWitness(tables: [producer, consumer])
        let ok = try engine.checkConstraints(witness: witness)
        expect(ok, "Balanced channel (reordered) should pass")
    } catch {
        expect(false, "Channel balance check failed: \(error)")
    }
}

private func testM3ChannelMultiTable() {
    let engine = BiniusM3Engine()

    // Two producers push into the same channel, one consumer pulls everything
    let prod1_vals: [UInt8] = [1, 2]
    let prod2_vals: [UInt8] = [3, 4]

    let prod1 = M3Table(
        name: "prod1",
        columns: [M3Column.fromGF8(name: "x", values: prod1_vals)],
        channelOps: [.push(channel: "ch", columns: ["x"])]
    )
    let prod2 = M3Table(
        name: "prod2",
        columns: [M3Column.fromGF8(name: "x", values: prod2_vals)],
        channelOps: [.push(channel: "ch", columns: ["x"])]
    )

    let consumer_vals: [UInt8] = [4, 2, 1, 3]  // all 4 values, reordered
    let consumer = M3Table(
        name: "consumer",
        columns: [M3Column.fromGF8(name: "y", values: consumer_vals)],
        channelOps: [.pull(channel: "ch", columns: ["y"])]
    )

    do {
        let witness = try engine.generateWitness(tables: [prod1, prod2, consumer])
        let ok = try engine.checkConstraints(witness: witness)
        expect(ok, "Multi-producer channel balance should pass")
    } catch {
        expect(false, "Multi-producer channel failed: \(error)")
    }
}

private func testM3ChannelImbalance() {
    let engine = BiniusM3Engine()

    // Push 4 values, pull only 2 -- should fail
    let prod_vals: [UInt8] = [1, 2, 3, 4]
    let cons_vals: [UInt8] = [1, 2]

    let producer = M3Table(
        name: "producer",
        columns: [M3Column.fromGF8(name: "x", values: prod_vals)],
        channelOps: [.push(channel: "ch", columns: ["x"])]
    )
    let consumer = M3Table(
        name: "consumer",
        columns: [M3Column.fromGF8(name: "y", values: cons_vals)],
        channelOps: [.pull(channel: "ch", columns: ["y"])]
    )

    do {
        let witness = try engine.generateWitness(tables: [producer, consumer])
        _ = try engine.checkConstraints(witness: witness)
        expect(false, "Imbalanced channel should fail")
    } catch let error as M3Error {
        if case .channelImbalance(_, let pushCount, let pullCount) = error {
            expect(pushCount == 4, "Push count should be 4")
            expect(pullCount == 2, "Pull count should be 2")
        } else {
            expect(true, "Got an M3Error (channel-related)")
        }
    } catch {
        expect(false, "Unexpected error: \(error)")
    }
}

// MARK: - Packing Constraints

private func testM3PackingGF8ToGF16() {
    let engine = BiniusM3Engine()

    // Pack two GF(2^8) columns into one GF(2^16) column
    // packed = lo + hi * 2^8  (positional embedding)
    let lo_vals: [UInt8] = [0x34, 0xAB, 0x00, 0xFF]
    let hi_vals: [UInt8] = [0x12, 0xCD, 0x00, 0xFF]
    // packed values: 0x1234, 0xCDAB, 0x0000, 0xFFFF
    let packed_vals: [BinaryTower128] = [
        BinaryTower128(lo: 0x1234, hi: 0),
        BinaryTower128(lo: 0xCDAB, hi: 0),
        BinaryTower128(lo: 0x0000, hi: 0),
        BinaryTower128(lo: 0xFFFF, hi: 0),
    ]

    let table = M3Table(
        name: "pack_test",
        columns: [
            M3Column.fromGF8(name: "lo", values: lo_vals),
            M3Column.fromGF8(name: "hi", values: hi_vals),
            M3Column(name: "packed", bitWidth: 16, values: packed_vals),
        ]
    )

    do {
        let ok = try engine.checkPacking(table: table, packedColumn: "packed",
                                          componentColumns: ["lo", "hi"])
        expect(ok, "GF8->GF16 packing should pass")
    } catch {
        expect(false, "Packing check failed: \(error)")
    }
}

// MARK: - Compilation

private func testM3Compile() {
    let engine = BiniusM3Engine()

    let a_vals: [UInt8] = [0x53, 0x10, 0xFF, 0x00]
    let b_vals: [UInt8] = [0xCA, 0x20, 0xFF, 0x00]
    let c_vals: [UInt8] = [0x99, 0x30, 0x00, 0x00]

    let constraint = M3ZeroConstraint(
        name: "xor",
        expr: .add(.add(.column("a"), .column("b")), .column("c"))
    )

    let table = M3Table(
        name: "xor_table",
        columns: [
            M3Column.fromGF8(name: "a", values: a_vals),
            M3Column.fromGF8(name: "b", values: b_vals),
            M3Column.fromGF8(name: "c", values: c_vals),
        ],
        zeroConstraints: [constraint],
        channelOps: [.push(channel: "result", columns: ["c"])]
    )

    do {
        let witness = try engine.generateWitness(tables: [table])
        let compiled = try engine.compile(witness: witness)

        expect(compiled.columnPolynomials.count == 3, "3 column polynomials")
        expect(compiled.zeroConstraintPolys.count == 1, "1 zero constraint poly")
        expect(compiled.channelConstraints.count == 1, "1 channel constraint")
        expect(compiled.totalElements == 12, "4 rows * 3 columns = 12 total elements")

        // Check that the zero constraint poly is indeed all zeros
        let zeroPoly = compiled.zeroConstraintPolys[0]
        let allZero = zeroPoly.evaluations.allSatisfy { $0.isZero }
        expect(allZero, "Zero constraint polynomial should be all zeros on valid witness")

        // Check column poly metadata
        let colPoly = compiled.columnPolynomials[0]
        expect(colPoly.numVars == 2, "log2(4) = 2 variables")
        expect(colPoly.bitWidth == 8, "GF(2^8) column")
    } catch {
        expect(false, "Compilation failed: \(error)")
    }
}

// MARK: - Table Builder DSL

private func testM3TableBuilder() {
    let engine = BiniusM3Engine()

    let table = M3TableBuilder(name: "dsl_test")
        .addGF8Column("x", values: [1, 2, 3, 4])
        .addGF8Column("y", values: [5, 6, 7, 8])
        .addZeroConstraint("trivial", .add(.column("x"), .column("x")))  // x + x = 0 in char 2
        .build()

    expect(table.name == "dsl_test", "Builder sets name")
    expect(table.columns.count == 2, "Builder adds 2 columns")
    expect(table.zeroConstraints.count == 1, "Builder adds 1 constraint")
    expect(table.channelOps.count == 0, "Builder has no channel ops")

    // The constraint x + x = 0 should always hold in char 2
    do {
        let witness = try engine.generateWitness(tables: [table])
        let ok = try engine.checkConstraints(witness: witness)
        expect(ok, "x + x = 0 constraint passes in char 2")
    } catch {
        expect(false, "Builder table constraint check failed: \(error)")
    }
}

// MARK: - XOR Circuit Example

/// End-to-end test: a simple XOR circuit with lookup via channels.
///
/// Table "xor_lookup": the complete XOR lookup table for 4-bit values
/// Table "computation": the actual computation trace
/// Channel "xor_io": carries (a, b, a^b) tuples from computation to lookup
private func testM3XORCircuit() {
    let engine = BiniusM3Engine()

    // Build the XOR lookup table: all (a, b, a^b) for a,b in [0..3]
    // 4x4 = 16 entries, pad to power of 2 (already 16)
    var lookupA = [UInt8]()
    var lookupB = [UInt8]()
    var lookupC = [UInt8]()
    for a in 0..<4 {
        for b in 0..<4 {
            lookupA.append(UInt8(a))
            lookupB.append(UInt8(b))
            lookupC.append(UInt8(a ^ b))
        }
    }

    let lookupTable = M3TableBuilder(name: "xor_lookup")
        .addGF8Column("la", values: lookupA)
        .addGF8Column("lb", values: lookupB)
        .addGF8Column("lc", values: lookupC)
        .pull(channel: "xor_io", columns: ["la", "lb", "lc"])
        .build()

    // Computation table: 16 rows of actual XOR operations (must use values from lookup)
    // We'll just do all 16 combinations in a different order
    var compA = [UInt8]()
    var compB = [UInt8]()
    var compC = [UInt8]()
    // Reverse order
    for a in stride(from: 3, through: 0, by: -1) {
        for b in stride(from: 3, through: 0, by: -1) {
            compA.append(UInt8(a))
            compB.append(UInt8(b))
            compC.append(UInt8(a ^ b))
        }
    }

    let compTable = M3TableBuilder(name: "computation")
        .addGF8Column("ca", values: compA)
        .addGF8Column("cb", values: compB)
        .addGF8Column("cc", values: compC)
        .addZeroConstraint("xor_verify",
            .add(.add(.column("ca"), .column("cb")), .column("cc")))
        .push(channel: "xor_io", columns: ["ca", "cb", "cc"])
        .build()

    do {
        let witness = try engine.generateWitness(tables: [lookupTable, compTable])

        // Check all constraints (zero + channels)
        let ok = try engine.checkConstraints(witness: witness)
        expect(ok, "XOR circuit constraints pass")

        // Compile
        let compiled = try engine.compile(witness: witness)
        expect(compiled.columnPolynomials.count == 6, "6 columns total (3 per table)")
        expect(compiled.channelConstraints.count == 1, "1 channel")
        expect(compiled.zeroConstraintPolys.count == 1, "1 zero constraint")
        // 16 rows * 3 columns * 2 tables = 96 elements
        expect(compiled.totalElements == 96, "16*6 = 96 total elements")
    } catch {
        expect(false, "XOR circuit failed: \(error)")
    }
}
