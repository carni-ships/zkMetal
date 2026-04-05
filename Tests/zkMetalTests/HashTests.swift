import zkMetal

func runHashTests() {
    suite("Poseidon2")
    let zero3 = [Fr.zero, Fr.zero, Fr.zero]
    let perm = poseidon2Permutation(zero3)
    expect(frToInt(perm[0]) != [0, 0, 0, 0], "Permutation non-trivial")
    let p1 = poseidon2Permutation([frFromInt(1), frFromInt(2), frFromInt(3)])
    let p2 = poseidon2Permutation([frFromInt(1), frFromInt(2), frFromInt(3)])
    var detOk = true
    for i in 0..<3 { if frToInt(p1[i]) != frToInt(p2[i]) { detOk = false } }
    expect(detOk, "Deterministic")

    do {
        let engine = try Poseidon2Engine()
        let n = 256
        var flat = [Fr]()
        for i in 0..<n { flat.append(frFromInt(UInt64(i))); flat.append(frFromInt(UInt64(i + n))) }
        let cpuH = (0..<n).map { poseidon2Hash(flat[2*$0], flat[2*$0+1]) }
        let gpuH = try engine.hashPairs(flat)
        var ok = true
        for i in 0..<n { if frToInt(cpuH[i]) != frToInt(gpuH[i]) { ok = false; break } }
        expect(ok, "GPU matches CPU (256)")
    } catch { expect(false, "P2 error: \(error)") }

    suite("Keccak-256")
    let h = keccak256([])
    expect(h.count == 32, "Empty hash length")
    expect(h != [UInt8](repeating: 0, count: 32), "Empty hash non-zero")
    expect(keccak256([1, 2, 3]) == keccak256([1, 2, 3]), "Deterministic")

    do {
        let engine = try Keccak256Engine()
        let n = 256
        var flat = [UInt8](repeating: 0, count: n * 64)
        for i in 0..<n {
            let val = UInt64(i)
            withUnsafeBytes(of: val) { src in
                for j in 0..<8 { flat[i * 64 + j] = src[j] }
            }
        }
        let cpuH = (0..<n).map { keccak256(Array(flat[$0*64..<($0+1)*64])) }
        let gpuH = try engine.hash64(flat)
        var ok = true
        for i in 0..<n {
            let gpuSlice = Array(gpuH[i*32..<(i+1)*32])
            if cpuH[i] != gpuSlice { ok = false; break }
        }
        expect(ok, "GPU matches CPU (256)")
    } catch { expect(false, "Keccak error: \(error)") }

    suite("Merkle Trees")
    let mN = 64
    var leaves = [Fr]()
    for i in 0..<mN { leaves.append(frFromInt(UInt64(i + 1))) }
    let tree = parallelPoseidon2Merkle(leaves)
    var leafOk = true
    for i in 0..<mN { if frToInt(tree[mN + i]) != frToInt(leaves[i]) { leafOk = false; break } }
    expect(leafOk, "P2 Merkle leaves")
    var nodeOk = true
    for i in stride(from: mN - 1, through: 1, by: -1) {
        let exp = poseidon2Hash(tree[2 * i], tree[2 * i + 1])
        if frToInt(tree[i]) != frToInt(exp) { nodeOk = false; break }
    }
    expect(nodeOk, "P2 Merkle internal nodes")
}
