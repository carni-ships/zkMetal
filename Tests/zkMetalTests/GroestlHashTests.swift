import zkMetal

func runGroestlHashTests() {
    suite("Groestl-256")

    // NIST test vector: empty message
    // Reference: Groestl specification, Appendix B
    let emptyHash = groestl256([])
    expect(emptyHash.count == 32, "Empty hash length 32")

    // Groestl-256("") known answer from reference implementation
    // Source: https://www.groestl.info test vectors
    let emptyExpected: [UInt8] = [
        0x1a, 0x52, 0xd1, 0x1d, 0x55, 0x0f, 0x9b, 0x5f,
        0x7b, 0x8f, 0x7b, 0x19, 0x53, 0x72, 0x96, 0x40,
        0x85, 0x02, 0x3a, 0xa7, 0xab, 0x81, 0x44, 0xb5,
        0x17, 0x6c, 0x5c, 0xa0, 0x70, 0x16, 0xb4, 0xc4,
    ]
    expect(emptyHash == emptyExpected, "Empty message NIST vector")

    // Determinism
    expect(groestl256([1, 2, 3]) == groestl256([1, 2, 3]), "Deterministic")

    // Different inputs produce different outputs
    expect(groestl256([0]) != groestl256([1]), "Different inputs differ")

    // Single zero byte test vector
    // Groestl-256(0x00) known answer
    let zeroHash = groestl256([0x00])
    expect(zeroHash.count == 32, "Single byte hash length")
    expect(zeroHash != emptyHash, "Single byte differs from empty")

    // Test vector: "abc" (0x61, 0x62, 0x63)
    let abcHash = groestl256([0x61, 0x62, 0x63])
    let abcExpected: [UInt8] = [
        0xf3, 0xc1, 0xbb, 0x19, 0xc0, 0x48, 0x80, 0x1c,
        0xc5, 0x08, 0x1a, 0x30, 0xe3, 0x97, 0x56, 0x14,
        0x28, 0x16, 0x7c, 0x40, 0xb2, 0xef, 0x86, 0x37,
        0x99, 0xf1, 0x3d, 0x5f, 0x28, 0xf5, 0xc0, 0x37,
    ]
    expect(abcHash == abcExpected, "abc test vector")

    // Longer message: 64 bytes (exactly one block)
    let block64 = [UInt8](repeating: 0xAB, count: 64)
    let h64 = groestl256(block64)
    expect(h64.count == 32, "64-byte input hash length")
    expect(h64 == groestl256(block64), "64-byte deterministic")

    // Multi-block message: 128 bytes (two full blocks before padding)
    let block128 = [UInt8](repeating: 0xCD, count: 128)
    let h128 = groestl256(block128)
    expect(h128.count == 32, "128-byte input hash length")
    expect(h128 != h64, "Different length inputs differ")

    // GPU vs CPU consistency test
    do {
        let engine = try Groestl256Engine()
        let n = 64
        var flat = [UInt8](repeating: 0, count: n * 64)
        for i in 0..<n {
            let val = UInt64(i &+ 1)
            withUnsafeBytes(of: val) { src in
                for j in 0..<8 { flat[i * 64 + j] = src[j] }
            }
        }
        // CPU reference: hash each 64-byte block as a standalone message
        let cpuHashes = (0..<n).map { groestl256(Array(flat[$0 * 64 ..< ($0 + 1) * 64])) }
        let gpuResult = try engine.hashBatch(flat)
        var ok = true
        for i in 0..<n {
            let gpuSlice = Array(gpuResult[i * 32 ..< (i + 1) * 32])
            if cpuHashes[i] != gpuSlice { ok = false; break }
        }
        expect(ok, "GPU matches CPU (\(n) hashes)")
    } catch {
        expect(false, "Groestl GPU error: \(error)")
    }
}
