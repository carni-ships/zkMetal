import Foundation
import zkMetal

// MARK: - PRNG for test data

private var neonTestRNG: UInt64 = 0xCAFE_BABE_DEAD_BEEF

private func nextNeonRandom() -> UInt64 {
    neonTestRNG ^= neonTestRNG << 13
    neonTestRNG ^= neonTestRNG >> 7
    neonTestRNG ^= neonTestRNG << 17
    return neonTestRNG
}

private func resetNeonRNG() {
    neonTestRNG = 0xCAFE_BABE_DEAD_BEEF
}

// MARK: - Test runner

public func runLatticeNTTNeonTests() {
    // ============================================================
    // Kyber int16 NTT round-trip
    // ============================================================
    suite("Lattice NEON: Kyber int16 round-trip")

    // Sequential input
    var kyberSeq = (1...256).map { Int16($0) }
    let kyberSeqOrig = kyberSeq
    kyberNTTNeonS16(&kyberSeq)
    // NTT should change the data
    var nttChanged = false
    for i in 0..<256 {
        if kyberSeq[i] != kyberSeqOrig[i] { nttChanged = true; break }
    }
    expect(nttChanged, "NTT modifies data (Kyber int16)")

    kyberINTTNeonS16(&kyberSeq)
    var seqOk = true
    for i in 0..<256 {
        if kyberSeq[i] != kyberSeqOrig[i] { seqOk = false; break }
    }
    expect(seqOk, "NTT->INTT round-trip (Kyber int16, sequential)")

    // Random input
    resetNeonRNG()
    var kyberRand = [Int16](repeating: 0, count: 256)
    for i in 0..<256 {
        kyberRand[i] = Int16(nextNeonRandom() % 3329)
    }
    let kyberRandOrig = kyberRand
    kyberNTTNeonS16(&kyberRand)
    kyberINTTNeonS16(&kyberRand)
    var randOk = true
    for i in 0..<256 {
        if kyberRand[i] != kyberRandOrig[i] { randOk = false; break }
    }
    expect(randOk, "NTT->INTT round-trip (Kyber int16, random)")

    // ============================================================
    // Kyber int16 vs existing uint32 consistency
    // ============================================================
    suite("Lattice NEON: Kyber int16 vs uint32 consistency")

    resetNeonRNG()
    var s16Input = [Int16](repeating: 0, count: 256)
    var u32Input = [UInt32](repeating: 0, count: 256)
    for i in 0..<256 {
        let v = UInt32(nextNeonRandom() % 3329)
        s16Input[i] = Int16(v)
        u32Input[i] = v
    }

    kyberNTTNeonS16(&s16Input)
    kyberNTTNeon(&u32Input)

    var matchOk = true
    for i in 0..<256 {
        if UInt32(s16Input[i]) != u32Input[i] {
            matchOk = false
            break
        }
    }
    expect(matchOk, "int16 NTT matches uint32 NTT (Kyber)")

    // INTT consistency
    kyberINTTNeonS16(&s16Input)
    kyberINTTNeon(&u32Input)
    var inttMatchOk = true
    for i in 0..<256 {
        if UInt32(s16Input[i]) != u32Input[i] {
            inttMatchOk = false
            break
        }
    }
    expect(inttMatchOk, "int16 INTT matches uint32 INTT (Kyber)")

    // ============================================================
    // Dilithium int32 NTT round-trip
    // ============================================================
    suite("Lattice NEON: Dilithium int32 round-trip")

    // Sequential input
    var dilSeq = (1...256).map { Int32($0) }
    let dilSeqOrig = dilSeq
    dilithiumNTTNeonS32(&dilSeq)
    var dilNttChanged = false
    for i in 0..<256 {
        if dilSeq[i] != dilSeqOrig[i] { dilNttChanged = true; break }
    }
    expect(dilNttChanged, "NTT modifies data (Dilithium int32)")

    dilithiumINTTNeonS32(&dilSeq)
    var dilSeqOk = true
    for i in 0..<256 {
        if dilSeq[i] != dilSeqOrig[i] { dilSeqOk = false; break }
    }
    expect(dilSeqOk, "NTT->INTT round-trip (Dilithium int32, sequential)")

    // Random input
    resetNeonRNG()
    var dilRand = [Int32](repeating: 0, count: 256)
    for i in 0..<256 {
        dilRand[i] = Int32(nextNeonRandom() % 8380417)
    }
    let dilRandOrig = dilRand
    dilithiumNTTNeonS32(&dilRand)
    dilithiumINTTNeonS32(&dilRand)
    var dilRandOk = true
    for i in 0..<256 {
        if dilRand[i] != dilRandOrig[i] { dilRandOk = false; break }
    }
    expect(dilRandOk, "NTT->INTT round-trip (Dilithium int32, random)")

    // ============================================================
    // Dilithium int32 vs existing uint32 consistency
    // ============================================================
    suite("Lattice NEON: Dilithium int32 vs uint32 consistency")

    resetNeonRNG()
    var s32Input = [Int32](repeating: 0, count: 256)
    var du32Input = [UInt32](repeating: 0, count: 256)
    for i in 0..<256 {
        let v = UInt32(nextNeonRandom() % 8380417)
        s32Input[i] = Int32(v)
        du32Input[i] = v
    }

    dilithiumNTTNeonS32(&s32Input)
    dilithiumNTTNeon(&du32Input)

    var dilMatchOk = true
    for i in 0..<256 {
        if UInt32(bitPattern: s32Input[i]) != du32Input[i] {
            dilMatchOk = false
            break
        }
    }
    expect(dilMatchOk, "int32 NTT matches uint32 NTT (Dilithium)")

    dilithiumINTTNeonS32(&s32Input)
    dilithiumINTTNeon(&du32Input)
    var dilInttMatchOk = true
    for i in 0..<256 {
        if UInt32(bitPattern: s32Input[i]) != du32Input[i] {
            dilInttMatchOk = false
            break
        }
    }
    expect(dilInttMatchOk, "int32 INTT matches uint32 INTT (Dilithium)")

    // ============================================================
    // Edge cases
    // ============================================================
    suite("Lattice NEON: Signed NTT edge cases")

    // Zero polynomial
    var kyberZeros = [Int16](repeating: 0, count: 256)
    kyberNTTNeonS16(&kyberZeros)
    let allZeroKyber = kyberZeros.allSatisfy { $0 == 0 }
    expect(allZeroKyber, "NTT(zero) = zero (Kyber int16)")

    var dilZeros = [Int32](repeating: 0, count: 256)
    dilithiumNTTNeonS32(&dilZeros)
    let allZeroDil = dilZeros.allSatisfy { $0 == 0 }
    expect(allZeroDil, "NTT(zero) = zero (Dilithium int32)")

    // All max values (q-1)
    var kyberMax = [Int16](repeating: 3328, count: 256)
    let kyberMaxOrig = kyberMax
    kyberNTTNeonS16(&kyberMax)
    kyberINTTNeonS16(&kyberMax)
    var maxOk = true
    for i in 0..<256 {
        if kyberMax[i] != kyberMaxOrig[i] { maxOk = false; break }
    }
    expect(maxOk, "All q-1 round-trip (Kyber int16)")

    var dilMax = [Int32](repeating: 8380416, count: 256)
    let dilMaxOrig = dilMax
    dilithiumNTTNeonS32(&dilMax)
    dilithiumINTTNeonS32(&dilMax)
    var dilMaxOk = true
    for i in 0..<256 {
        if dilMax[i] != dilMaxOrig[i] { dilMaxOk = false; break }
    }
    expect(dilMaxOk, "All q-1 round-trip (Dilithium int32)")

    // Constant polynomial round-trip
    var kyberConst = [Int16](repeating: 0, count: 256)
    kyberConst[0] = 42
    kyberNTTNeonS16(&kyberConst)
    kyberINTTNeonS16(&kyberConst)
    expect(kyberConst[0] == 42, "Constant poly coeff[0] round-trip (Kyber int16)")
    let kyberConstRest = kyberConst[1...].allSatisfy { $0 == 0 }
    expect(kyberConstRest, "Constant poly rest zero (Kyber int16)")

    var dilConst = [Int32](repeating: 0, count: 256)
    dilConst[0] = 12345
    dilithiumNTTNeonS32(&dilConst)
    dilithiumINTTNeonS32(&dilConst)
    expect(dilConst[0] == 12345, "Constant poly coeff[0] round-trip (Dilithium int32)")
    let dilConstRest = dilConst[1...].allSatisfy { $0 == 0 }
    expect(dilConstRest, "Constant poly rest zero (Dilithium int32)")

    // ============================================================
    // Performance: verify NEON path is exercised (basic timing sanity)
    // ============================================================
    suite("Lattice NEON: Signed NTT performance")

    resetNeonRNG()
    var perfPoly16 = [Int16](repeating: 0, count: 256)
    for i in 0..<256 { perfPoly16[i] = Int16(nextNeonRandom() % 3329) }

    let iterations = 10000
    let t0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        kyberNTTNeonS16(&perfPoly16)
        kyberINTTNeonS16(&perfPoly16)
    }
    let kyberUs = (CFAbsoluteTimeGetCurrent() - t0) * 1_000_000 / Double(iterations)
    print("  Kyber int16 NTT+INTT: \(String(format: "%.2f", kyberUs)) us/pair")
    expect(kyberUs < 100.0, "Kyber int16 NTT+INTT < 100us")

    var perfPoly32 = [Int32](repeating: 0, count: 256)
    for i in 0..<256 { perfPoly32[i] = Int32(nextNeonRandom() % 8380417) }

    let t1 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        dilithiumNTTNeonS32(&perfPoly32)
        dilithiumINTTNeonS32(&perfPoly32)
    }
    let dilUs = (CFAbsoluteTimeGetCurrent() - t1) * 1_000_000 / Double(iterations)
    print("  Dilithium int32 NTT+INTT: \(String(format: "%.2f", dilUs)) us/pair")
    expect(dilUs < 200.0, "Dilithium int32 NTT+INTT < 200us")
}
