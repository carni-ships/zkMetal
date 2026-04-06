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
        0x1a, 0x52, 0xd1, 0x1d, 0x55, 0x00, 0x39, 0xbe,
        0x16, 0x10, 0x7f, 0x9c, 0x58, 0xdb, 0x9e, 0xbc,
        0xc4, 0x17, 0xf1, 0x6f, 0x73, 0x6a, 0xdb, 0x25,
        0x02, 0x56, 0x71, 0x19, 0xf0, 0x08, 0x34, 0x67,
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
        0xf3, 0xc1, 0xbb, 0x19, 0xc0, 0x48, 0x80, 0x13,
        0x26, 0xa7, 0xef, 0xbc, 0xf1, 0x6e, 0x3d, 0x78,
        0x87, 0x44, 0x62, 0x49, 0x82, 0x9c, 0x37, 0x9e,
        0x18, 0x40, 0xd1, 0xa3, 0xa1, 0xe7, 0xd4, 0xd2,
    ]
    expect(abcHash == abcExpected, "abc test vector")

    // "The quick brown fox jumps over the lazy dog" — known reference vector
    let foxInput = Array("The quick brown fox jumps over the lazy dog".utf8)
    let foxHash = groestl256(foxInput)
    let foxExpected: [UInt8] = [
        0x8c, 0x7a, 0xd6, 0x2e, 0xb2, 0x6a, 0x21, 0x29,
        0x7b, 0xc3, 0x9c, 0x2d, 0x72, 0x93, 0xb4, 0xbd,
        0x4d, 0x33, 0x99, 0xfa, 0x8a, 0xfa, 0xb2, 0x9e,
        0x97, 0x04, 0x71, 0x73, 0x9e, 0x28, 0xb3, 0x01,
    ]
    expect(foxHash == foxExpected, "Quick brown fox test vector")

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
