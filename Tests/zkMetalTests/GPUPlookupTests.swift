// Tests for GPUPlookupEngine -- Plookup lookup argument

import Foundation
import zkMetal

func runGPUPlookupTests() {
    suite("GPUPlookup")

    testPlookupSimple()
    testPlookupRepeated()
    testPlookupAsymmetric()
    testPlookupAccumulatorCloses()
    testPlookupRejectTampered()
    testPlookupAutoChallenge()
    testPlookupLarger()
}

// MARK: - Basic Prove/Verify

func testPlookupSimple() {
    do {
        let engine = try GPUPlookupEngine()
        let table = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let witness = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]
        let beta = frFromInt(12345)
        let gamma = frFromInt(67890)

        let proof = try engine.prove(witness: witness, table: table,
                                      beta: beta, gamma: gamma)
        let valid = engine.verify(proof: proof, witness: witness, table: table)
        expect(valid, "Simple Plookup (n=4, N=4)")
    } catch {
        expect(false, "Simple Plookup threw: \(error)")
    }
}

func testPlookupRepeated() {
    do {
        let engine = try GPUPlookupEngine()
        let table: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let witness: [Fr] = [1, 1, 3, 3, 5, 5, 7, 7].map { frFromInt($0) }

        let proof = try engine.prove(witness: witness, table: table)
        let valid = engine.verify(proof: proof, witness: witness, table: table)
        expect(valid, "Repeated lookups (n=8, N=8)")
    } catch {
        expect(false, "Repeated lookups threw: \(error)")
    }
}

func testPlookupAsymmetric() {
    do {
        let engine = try GPUPlookupEngine()
        let table: [Fr] = (0..<16).map { frFromInt(UInt64($0 * 7 + 3)) }
        let witness: [Fr] = (0..<4).map { table[$0] }

        let proof = try engine.prove(witness: witness, table: table)
        let valid = engine.verify(proof: proof, witness: witness, table: table)
        expect(valid, "Asymmetric (n=4, N=16)")
    } catch {
        expect(false, "Asymmetric threw: \(error)")
    }
}

func testPlookupAccumulatorCloses() {
    do {
        let engine = try GPUPlookupEngine()
        let table = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let witness = [frFromInt(2), frFromInt(4), frFromInt(1), frFromInt(3)]

        let proof = try engine.prove(witness: witness, table: table)
        expect(frEqual(proof.finalAccumulator, Fr.one), "Accumulator Z[n] == 1")
        expect(proof.sortedVector.count == 8, "Sorted vector length == n + N")
        expect(proof.accumulatorZ.count == 8, "Accumulator length == n + N")
        expect(frEqual(proof.accumulatorZ[0], Fr.one), "Accumulator Z[0] == 1")
    } catch {
        expect(false, "AccumulatorCloses threw: \(error)")
    }
}

func testPlookupRejectTampered() {
    do {
        let engine = try GPUPlookupEngine()
        let table = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let witness = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]

        let proof = try engine.prove(witness: witness, table: table)

        // Tamper with the final accumulator
        let tampered = PlookupProof(
            sortedVector: proof.sortedVector,
            accumulatorZ: proof.accumulatorZ,
            beta: proof.beta,
            gamma: proof.gamma,
            finalAccumulator: frAdd(proof.finalAccumulator, Fr.one)
        )
        let rejected = !engine.verify(proof: tampered, witness: witness, table: table)
        expect(rejected, "Reject tampered accumulator")

        // Tamper with sorted vector
        var badSorted = proof.sortedVector
        badSorted[0] = frFromInt(999)
        let tampered2 = PlookupProof(
            sortedVector: badSorted,
            accumulatorZ: proof.accumulatorZ,
            beta: proof.beta,
            gamma: proof.gamma,
            finalAccumulator: proof.finalAccumulator
        )
        let rejected2 = !engine.verify(proof: tampered2, witness: witness, table: table)
        expect(rejected2, "Reject tampered sorted vector")
    } catch {
        expect(false, "RejectTampered threw: \(error)")
    }
}

func testPlookupAutoChallenge() {
    do {
        let engine = try GPUPlookupEngine()
        let table: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let witness: [Fr] = [1, 2, 3, 4, 5, 6, 7, 8].map { frFromInt($0) }

        // Auto-derive challenges (beta=nil, gamma=nil)
        let proof = try engine.prove(witness: witness, table: table)
        let valid = engine.verify(proof: proof, witness: witness, table: table)
        expect(valid, "Auto Fiat-Shamir challenges")

        // Challenges should be non-zero
        expect(!proof.beta.isZero, "Beta is non-zero")
        expect(!proof.gamma.isZero, "Gamma is non-zero")
    } catch {
        expect(false, "AutoChallenge threw: \(error)")
    }
}

func testPlookupLarger() {
    do {
        let engine = try GPUPlookupEngine()
        let N = 256
        let n = 128
        let table: [Fr] = (0..<N).map { frFromInt(UInt64($0 + 1)) }

        var rng: UInt64 = 0xCAFE_BABE
        var witness = [Fr]()
        witness.reserveCapacity(n)
        for _ in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let idx = Int(rng >> 32) % N
            witness.append(table[idx])
        }

        let proof = try engine.prove(witness: witness, table: table)
        let valid = engine.verify(proof: proof, witness: witness, table: table)
        expect(valid, "Larger Plookup (n=128, N=256)")
        expect(frEqual(proof.finalAccumulator, Fr.one), "Larger accumulator closes")
    } catch {
        expect(false, "Larger threw: \(error)")
    }
}
