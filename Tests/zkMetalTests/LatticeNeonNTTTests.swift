// Tests for NEON-accelerated lattice NTT (Kyber q=3329, Dilithium q=8380417)
// Validates NEON implementations against existing CPU reference implementations.

import Foundation
import zkMetal

// MARK: - PRNG for test data

private var neonTestRNG: UInt64 = 0xCAFE_BABE_DEAD_BEEF

private func neonNextRandom() -> UInt64 {
    neonTestRNG ^= neonTestRNG << 13
    neonTestRNG ^= neonTestRNG >> 7
    neonTestRNG ^= neonTestRNG << 17
    return neonTestRNG
}

private func neonResetRNG() {
    neonTestRNG = 0xCAFE_BABE_DEAD_BEEF
}

// MARK: - Test runner

public func runLatticeNeonNTTTests() {
    // ============================================================
    // Kyber NEON NTT round-trip
    // ============================================================
    suite("Lattice NEON NTT: Kyber round-trip")

    // Sequential input
    var seqPoly = (1...256).map { UInt32($0) }
    let seqCopy = seqPoly
    kyberNTTNeon(&seqPoly)
    kyberINTTNeon(&seqPoly)
    var ok = true
    for i in 0..<256 {
        if seqPoly[i] != seqCopy[i] { ok = false; break }
    }
    expect(ok, "Kyber NEON NTT->INTT round-trip (sequential)")

    // Random input
    neonResetRNG()
    var randPoly = [UInt32](repeating: 0, count: 256)
    for i in 0..<256 { randPoly[i] = UInt32(neonNextRandom() % 3329) }
    let randCopy = randPoly
    kyberNTTNeon(&randPoly)
    kyberINTTNeon(&randPoly)
    var randOk = true
    for i in 0..<256 {
        if randPoly[i] != randCopy[i] { randOk = false; break }
    }
    expect(randOk, "Kyber NEON NTT->INTT round-trip (random)")

    // ============================================================
    // Kyber NEON vs CPU reference
    // ============================================================
    suite("Lattice NEON NTT: Kyber NEON vs CPU")

    neonResetRNG()
    var neonInput = [UInt32](repeating: 0, count: 256)
    for i in 0..<256 { neonInput[i] = UInt32(neonNextRandom() % 3329) }
    var cpuInput = neonInput.map { KyberField(value: UInt16($0)) }
    var neonFwd = neonInput
    kyberNTTNeon(&neonFwd)
    kyberNTTCPU(&cpuInput)
    let cpuFwd = cpuInput.map { UInt32($0.value) }

    var fwdOk = true
    for i in 0..<256 {
        if neonFwd[i] != cpuFwd[i] { fwdOk = false; break }
    }
    expect(fwdOk, "Kyber NEON forward NTT matches CPU reference")

    // Inverse
    var neonInv = neonFwd
    kyberINTTNeon(&neonInv)
    kyberInvNTTCPU(&cpuInput)
    let cpuInv = cpuInput.map { UInt32($0.value) }

    var invOk = true
    for i in 0..<256 {
        if neonInv[i] != cpuInv[i] { invOk = false; break }
    }
    expect(invOk, "Kyber NEON inverse NTT matches CPU reference")

    // ============================================================
    // Dilithium NEON NTT round-trip
    // ============================================================
    suite("Lattice NEON NTT: Dilithium round-trip")

    var dilSeq = (1...256).map { UInt32($0) }
    let dilSeqCopy = dilSeq
    dilithiumNTTNeon(&dilSeq)
    dilithiumINTTNeon(&dilSeq)
    var dilOk = true
    for i in 0..<256 {
        if dilSeq[i] != dilSeqCopy[i] { dilOk = false; break }
    }
    expect(dilOk, "Dilithium NEON NTT->INTT round-trip (sequential)")

    neonResetRNG()
    var dilRand = [UInt32](repeating: 0, count: 256)
    for i in 0..<256 { dilRand[i] = UInt32(neonNextRandom() % 8380417) }
    let dilRandCopy = dilRand
    dilithiumNTTNeon(&dilRand)
    dilithiumINTTNeon(&dilRand)
    var dilRandOk = true
    for i in 0..<256 {
        if dilRand[i] != dilRandCopy[i] { dilRandOk = false; break }
    }
    expect(dilRandOk, "Dilithium NEON NTT->INTT round-trip (random)")

    // ============================================================
    // Dilithium NEON vs CPU reference
    // ============================================================
    suite("Lattice NEON NTT: Dilithium NEON vs CPU")

    neonResetRNG()
    var dilNeonIn = [UInt32](repeating: 0, count: 256)
    for i in 0..<256 { dilNeonIn[i] = UInt32(neonNextRandom() % 8380417) }
    var dilCpuIn = dilNeonIn.map { DilithiumField(value: $0) }
    var dilNeonFwd = dilNeonIn
    dilithiumNTTNeon(&dilNeonFwd)
    dilithiumNTTCPU(&dilCpuIn)
    let dilCpuFwd = dilCpuIn.map { $0.value }

    var dilFwdOk = true
    for i in 0..<256 {
        if dilNeonFwd[i] != dilCpuFwd[i] { dilFwdOk = false; break }
    }
    expect(dilFwdOk, "Dilithium NEON forward NTT matches CPU reference")

    var dilNeonInv = dilNeonFwd
    dilithiumINTTNeon(&dilNeonInv)
    dilithiumInvNTTCPU(&dilCpuIn)
    let dilCpuInv = dilCpuIn.map { $0.value }

    var dilInvOk = true
    for i in 0..<256 {
        if dilNeonInv[i] != dilCpuInv[i] { dilInvOk = false; break }
    }
    expect(dilInvOk, "Dilithium NEON inverse NTT matches CPU reference")

    // ============================================================
    // Batch-4 Kyber NTT
    // ============================================================
    suite("Lattice NEON NTT: Kyber batch-4")

    neonResetRNG()
    var polys = [[UInt32]](repeating: [UInt32](repeating: 0, count: 256), count: 4)
    for p in 0..<4 {
        for i in 0..<256 { polys[p][i] = UInt32(neonNextRandom() % 3329) }
    }
    let polysCopy = polys

    // Batch-4 NTT
    let (b0, b1, b2, b3) = kyberNTTNeonBatch4(polys[0], polys[1], polys[2], polys[3])

    // Compare with individual NEON NTTs
    var singles = polys
    for p in 0..<4 { kyberNTTNeon(&singles[p]) }

    var batchFwdOk = true
    let batchResults = [b0, b1, b2, b3]
    for p in 0..<4 {
        for i in 0..<256 {
            if batchResults[p][i] != singles[p][i] { batchFwdOk = false; break }
        }
        if !batchFwdOk { break }
    }
    expect(batchFwdOk, "Kyber batch-4 NTT matches individual NEON NTTs")

    // Batch-4 INTT round-trip
    let (r0, r1, r2, r3) = kyberINTTNeonBatch4(b0, b1, b2, b3)
    let recovered = [r0, r1, r2, r3]

    var batchRtOk = true
    for p in 0..<4 {
        for i in 0..<256 {
            if recovered[p][i] != polysCopy[p][i] { batchRtOk = false; break }
        }
        if !batchRtOk { break }
    }
    expect(batchRtOk, "Kyber batch-4 NTT->INTT round-trip")

    // ============================================================
    // Batch-4 Dilithium NTT
    // ============================================================
    suite("Lattice NEON NTT: Dilithium batch-4")

    neonResetRNG()
    var dPolys = [[UInt32]](repeating: [UInt32](repeating: 0, count: 256), count: 4)
    for p in 0..<4 {
        for i in 0..<256 { dPolys[p][i] = UInt32(neonNextRandom() % 8380417) }
    }
    let dPolysCopy = dPolys

    let (db0, db1, db2, db3) = dilithiumNTTNeonBatch4(dPolys[0], dPolys[1], dPolys[2], dPolys[3])

    var dSingles = dPolys
    for p in 0..<4 { dilithiumNTTNeon(&dSingles[p]) }

    var dBatchFwdOk = true
    let dBatchResults = [db0, db1, db2, db3]
    for p in 0..<4 {
        for i in 0..<256 {
            if dBatchResults[p][i] != dSingles[p][i] { dBatchFwdOk = false; break }
        }
        if !dBatchFwdOk { break }
    }
    expect(dBatchFwdOk, "Dilithium batch-4 NTT matches individual NEON NTTs")

    let (dr0, dr1, dr2, dr3) = dilithiumINTTNeonBatch4(db0, db1, db2, db3)
    let dRecovered = [dr0, dr1, dr2, dr3]

    var dBatchRtOk = true
    for p in 0..<4 {
        for i in 0..<256 {
            if dRecovered[p][i] != dPolysCopy[p][i] { dBatchRtOk = false; break }
        }
        if !dBatchRtOk { break }
    }
    expect(dBatchRtOk, "Dilithium batch-4 NTT->INTT round-trip")

    // ============================================================
    // Edge cases
    // ============================================================
    suite("Lattice NEON NTT: Edge cases")

    // Zero polynomial
    var zeros = [UInt32](repeating: 0, count: 256)
    kyberNTTNeon(&zeros)
    expect(zeros.allSatisfy { $0 == 0 }, "Kyber NEON NTT(zero) = zero")

    // All q-1
    var maxKyber = [UInt32](repeating: 3328, count: 256)
    let maxKyberCopy = maxKyber
    kyberNTTNeon(&maxKyber)
    kyberINTTNeon(&maxKyber)
    var maxOk = true
    for i in 0..<256 {
        if maxKyber[i] != maxKyberCopy[i] { maxOk = false; break }
    }
    expect(maxOk, "Kyber NEON round-trip with all q-1")

    var maxDil = [UInt32](repeating: 8380416, count: 256)
    let maxDilCopy = maxDil
    dilithiumNTTNeon(&maxDil)
    dilithiumINTTNeon(&maxDil)
    var maxDilOk = true
    for i in 0..<256 {
        if maxDil[i] != maxDilCopy[i] { maxDilOk = false; break }
    }
    expect(maxDilOk, "Dilithium NEON round-trip with all q-1")

    // All ones
    var ones = [UInt32](repeating: 1, count: 256)
    let onesCopy = ones
    kyberNTTNeon(&ones)
    kyberINTTNeon(&ones)
    var onesOk = true
    for i in 0..<256 {
        if ones[i] != onesCopy[i] { onesOk = false; break }
    }
    expect(onesOk, "Kyber NEON round-trip with all ones")

    // Constant polynomial [c, 0, 0, ...]
    var constPoly = [UInt32](repeating: 0, count: 256)
    constPoly[0] = 42
    kyberNTTNeon(&constPoly)
    kyberINTTNeon(&constPoly)
    expect(constPoly[0] == 42, "Kyber NEON constant poly coeff 0 preserved")
    expect(constPoly[1...].allSatisfy { $0 == 0 }, "Kyber NEON constant poly rest zero")

    // ============================================================
    // Performance comparison (brief, just to validate no regression)
    // ============================================================
    suite("Lattice NEON NTT: Performance sanity")

    neonResetRNG()
    var perfPoly = [UInt32](repeating: 0, count: 256)
    for i in 0..<256 { perfPoly[i] = UInt32(neonNextRandom() % 3329) }

    let iters = 10000
    let t0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iters {
        kyberNTTNeon(&perfPoly)
    }
    let neonTime = CFAbsoluteTimeGetCurrent() - t0

    neonResetRNG()
    var cpuPerfPoly = (0..<256).map { _ in KyberField(value: UInt16(neonNextRandom() % 3329)) }
    let t1 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iters {
        kyberNTTCPU(&cpuPerfPoly)
    }
    let cpuTime = CFAbsoluteTimeGetCurrent() - t1

    let speedup = cpuTime / neonTime
    // Just check it's not drastically slower (NEON should be at least as fast)
    expect(speedup > 0.5, "Kyber NEON not slower than 2x vs CPU (speedup: \(String(format: "%.2f", speedup))x)")
}
