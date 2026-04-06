// Tests for GPULogUpEngine -- LogUp lookup argument

import Foundation
import zkMetal

public func runGPULogUpEngineTests() {
    suite("GPULogUpEngine")

    testLogUpSingleTable()
    testLogUpMultiTable()
    testLogUpInvalidLookup()
    testLogUpEmptyTable()
    testLogUpLargeRandom()
    testLogUpRepeatedLookups()
    testLogUpAutoChallenge()
}

// MARK: - Single Table

func testLogUpSingleTable() {
    do {
        let engine = try GPULogUpEngine()
        let table = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let witness = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]
        let beta = frFromInt(12345)

        let proof = try engine.prove(witness: witness, table: table, beta: beta)
        let valid = engine.verify(proof: proof, witness: witness, table: table)
        expect(valid, "Single table LogUp (n=4, N=4)")
        expect(frEqual(proof.finalGrandSum, Fr.zero), "Grand sum closes to zero")
        expect(frEqual(proof.lookupSum, proof.tableSum), "Lookup sum == table sum")
    } catch {
        expect(false, "Single table LogUp threw: \(error)")
    }
}

// MARK: - Multi-Table

func testLogUpMultiTable() {
    do {
        let engine = try GPULogUpEngine()
        let table0: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let table1: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]

        // Witness looks up from both tables
        let witness: [Fr] = [frFromInt(2), frFromInt(20), frFromInt(1), frFromInt(30)]
        let selectors = [0, 1, 0, 1]  // table index for each witness element
        let beta = frFromInt(99999)

        let proof = try engine.proveMultiTable(
            witness: witness, tableSelectors: selectors,
            tables: [table0, table1], beta: beta)
        let valid = engine.verifyMultiTable(
            proof: proof, witness: witness,
            tableSelectors: selectors, tables: [table0, table1])
        expect(valid, "Multi-table LogUp (2 tables)")
        expect(frEqual(proof.finalGrandSum, Fr.zero), "Multi-table grand sum zero")
    } catch {
        expect(false, "Multi-table LogUp threw: \(error)")
    }
}

// MARK: - Invalid Lookup Detection

func testLogUpInvalidLookup() {
    do {
        let engine = try GPULogUpEngine()
        let table = [frFromInt(10), frFromInt(20), frFromInt(30)]
        let validWitness = [frFromInt(10), frFromInt(20)]
        let beta = frFromInt(7777)

        // Valid proof first
        let proof = try engine.prove(witness: validWitness, table: table, beta: beta)
        let valid = engine.verify(proof: proof, witness: validWitness, table: table)
        expect(valid, "Valid lookup passes")

        // Tamper: change a lookup inverse
        var badInverses = proof.lookupInverses
        badInverses[0] = frAdd(badInverses[0], Fr.one)
        let tampered = LogUpProof(
            multiplicities: proof.multiplicities,
            beta: proof.beta,
            lookupInverses: badInverses,
            tableTerms: proof.tableTerms,
            grandSumAccumulator: proof.grandSumAccumulator,
            finalGrandSum: proof.finalGrandSum,
            lookupSum: proof.lookupSum,
            tableSum: proof.tableSum
        )
        let rejected = !engine.verify(proof: tampered, witness: validWitness, table: table)
        expect(rejected, "Tampered inverse rejected")

        // Tamper: wrong final grand sum
        let tampered2 = LogUpProof(
            multiplicities: proof.multiplicities,
            beta: proof.beta,
            lookupInverses: proof.lookupInverses,
            tableTerms: proof.tableTerms,
            grandSumAccumulator: proof.grandSumAccumulator,
            finalGrandSum: Fr.one,  // should be zero
            lookupSum: proof.lookupSum,
            tableSum: proof.tableSum
        )
        let rejected2 = !engine.verify(proof: tampered2, witness: validWitness, table: table)
        expect(rejected2, "Non-zero grand sum rejected")
    } catch {
        expect(false, "Invalid lookup detection threw: \(error)")
    }
}

// MARK: - Empty Table Edge Case (single element)

func testLogUpEmptyTable() {
    do {
        let engine = try GPULogUpEngine()
        // Minimal case: 1-element table, 1-element witness
        let table = [frFromInt(42)]
        let witness = [frFromInt(42)]
        let beta = frFromInt(555)

        let proof = try engine.prove(witness: witness, table: table, beta: beta)
        let valid = engine.verify(proof: proof, witness: witness, table: table)
        expect(valid, "Minimal (1-elem) LogUp")
        expect(frEqual(proof.finalGrandSum, Fr.zero), "Minimal grand sum zero")
        expect(frEqual(proof.multiplicities[0], Fr.one), "Single multiplicity == 1")
    } catch {
        expect(false, "Empty table test threw: \(error)")
    }
}

// MARK: - Large Random Lookups

func testLogUpLargeRandom() {
    do {
        let engine = try GPULogUpEngine()
        let N = 256
        let m = 512
        let table: [Fr] = (0..<N).map { frFromInt(UInt64($0 + 1)) }

        // Deterministic pseudo-random witness (all elements in table)
        var rng: UInt64 = 0xDEAD_BEEF
        var witness = [Fr]()
        witness.reserveCapacity(m)
        for _ in 0..<m {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let idx = Int(rng >> 32) % N
            witness.append(table[idx])
        }

        let proof = try engine.prove(witness: witness, table: table)
        let valid = engine.verify(proof: proof, witness: witness, table: table)
        expect(valid, "Large random LogUp (m=512, N=256)")
        expect(frEqual(proof.finalGrandSum, Fr.zero), "Large random grand sum zero")

        // Check multiplicities sum to m
        var multSum = Fr.zero
        for j in 0..<N {
            multSum = frAdd(multSum, proof.multiplicities[j])
        }
        expect(frEqual(multSum, frFromInt(UInt64(m))), "Multiplicities sum to m")
    } catch {
        expect(false, "Large random threw: \(error)")
    }
}

// MARK: - Repeated Lookups

func testLogUpRepeatedLookups() {
    do {
        let engine = try GPULogUpEngine()
        let table: [Fr] = [frFromInt(5), frFromInt(10), frFromInt(15)]
        // All witness elements are the same value
        let witness: [Fr] = [frFromInt(10), frFromInt(10), frFromInt(10), frFromInt(10)]
        let beta = frFromInt(333)

        let proof = try engine.prove(witness: witness, table: table, beta: beta)
        let valid = engine.verify(proof: proof, witness: witness, table: table)
        expect(valid, "Repeated lookups (all same value)")
        expect(frEqual(proof.multiplicities[1], frFromInt(4)), "Multiplicity of 10 == 4")
        expect(frEqual(proof.multiplicities[0], Fr.zero), "Multiplicity of 5 == 0")
        expect(frEqual(proof.multiplicities[2], Fr.zero), "Multiplicity of 15 == 0")
    } catch {
        expect(false, "Repeated lookups threw: \(error)")
    }
}

// MARK: - Auto Challenge

func testLogUpAutoChallenge() {
    do {
        let engine = try GPULogUpEngine()
        let table: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let witness: [Fr] = [1, 2, 3, 4, 5, 6, 7, 8].map { frFromInt($0) }

        // Auto-derive challenge (beta=nil)
        let proof = try engine.prove(witness: witness, table: table)
        let valid = engine.verify(proof: proof, witness: witness, table: table)
        expect(valid, "Auto Fiat-Shamir challenge")
        expect(!proof.beta.isZero, "Beta is non-zero")
    } catch {
        expect(false, "Auto challenge threw: \(error)")
    }
}
