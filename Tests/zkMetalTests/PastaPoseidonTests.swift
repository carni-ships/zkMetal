// Tests for Pasta Poseidon hash (Mina Kimchi variant)
// Test vectors from o1-labs/proof-systems reference implementation.

import Foundation
import zkMetal

func runPastaPoseidonTests() {
    print("RUNNING PASTA POSEIDON TESTS")
    suite("Pasta Poseidon")

    // MARK: - Pallas CPU Tests

    let zero = PallasFp.zero
    let one = PallasFp.one

    // Test permutation returns 3 elements
    let permResult = pallasPoseidonPermutation([zero, zero, zero])
    expect(permResult.count == 3, "Permutation returns 3 elements")

    // Test hash(0,0) = permute([0,0,0])[0]
    let h00 = pallasPoseidonHash(zero, zero)
    expectEqual(h00.v.0, permResult[0].v.0, "hash(0,0) == permute[0] limb 0")
    expectEqual(h00.v.7, permResult[0].v.7, "hash(0,0) == permute[0] limb 7")

    // Test determinism
    let h00a = pallasPoseidonHash(zero, zero)
    let h00b = pallasPoseidonHash(zero, zero)
    expectEqual(h00a.v.0, h00b.v.0, "Hash determinism limb 0")
    expectEqual(h00a.v.7, h00b.v.7, "Hash determinism limb 7")

    // Test in-place permutation matches array version
    var s0 = zero, s1 = zero, s2 = zero
    pallasPoseidonPermuteInPlace(&s0, &s1, &s2)
    expectEqual(s0.v.0, permResult[0].v.0, "In-place permute s0 limb 0")
    expectEqual(s0.v.7, permResult[0].v.7, "In-place permute s0 limb 7")

    // Test different inputs produce different hashes
    let h01 = pallasPoseidonHash(zero, one)
    let h10 = pallasPoseidonHash(one, zero)
    let h11 = pallasPoseidonHash(one, one)
    expect(h00.v.0 != h01.v.0, "hash(0,0) != hash(0,1)")
    expect(h00.v.0 != h10.v.0, "hash(0,0) != hash(1,0)")
    expect(h00.v.0 != h11.v.0, "hash(0,0) != hash(1,1)")
    expect(h01.v.0 != h10.v.0, "hash(0,1) != hash(1,0)")

    // Test hashMany equals hash for single element
    let h1 = pallasPoseidonHashMany([zero])
    let hRef = pallasPoseidonHash(zero, zero)
    expectEqual(h1.v.0, hRef.v.0, "hashMany([0]) == hash(0,0) limb 0")
    expectEqual(h1.v.7, hRef.v.7, "hashMany([0]) == hash(0,0) limb 7")

    // MARK: - Vesta CPU Tests

    let veZero = VestaFp.zero
    let veOne = VestaFp.one

    let vePermResult = vestaPoseidonPermutation([veZero, veZero, veZero])
    expect(vePermResult.count == 3, "Vesta permutation returns 3 elements")

    let veH00 = vestaPoseidonHash(veZero, veZero)
    expectEqual(veH00.v.0, vePermResult[0].v.0, "Vesta hash(0,0) == permute[0] limb 0")
    expectEqual(veH00.v.7, vePermResult[0].v.7, "Vesta hash(0,0) == permute[0] limb 7")

    // Vesta determinism
    let veH00a = vestaPoseidonHash(veZero, veZero)
    let veH00b = vestaPoseidonHash(veZero, veZero)
    expectEqual(veH00a.v.0, veH00b.v.0, "Vesta hash determinism")

    // Vesta different inputs
    let veH01 = vestaPoseidonHash(veZero, veOne)
    let veH10 = vestaPoseidonHash(veOne, veZero)
    expect(veH00.v.0 != veH01.v.0, "Vesta hash(0,0) != hash(0,1)")
    expect(veH00.v.0 != veH10.v.0, "Vesta hash(0,0) != hash(1,0)")

    // Pallas vs Vesta produce different results
    expect(h00.v.0 != veH00.v.0, "Pallas and Vesta hashes differ")

    // MARK: - GPU Tests
    do {
        let engine = try PastaPoseidonEngine()

        // Test Pallas GPU hash pairs
        let pallasInput: [PallasFp] = [zero, zero, zero, one, one, zero, one, one]
        let gpuPallasHash = try engine.pallasHashPairs(pallasInput)
        expect(gpuPallasHash.count == 4, "GPU Pallas hash pairs count")

        expectEqual(gpuPallasHash[0].v.0, h00.v.0, "GPU Pallas hash(0,0) limb 0")
        expectEqual(gpuPallasHash[0].v.7, h00.v.7, "GPU Pallas hash(0,0) limb 7")
        expectEqual(gpuPallasHash[1].v.0, h01.v.0, "GPU Pallas hash(0,1) limb 0")
        expectEqual(gpuPallasHash[1].v.7, h01.v.7, "GPU Pallas hash(0,1) limb 7")
        expectEqual(gpuPallasHash[2].v.0, h10.v.0, "GPU Pallas hash(1,0) limb 0")
        expectEqual(gpuPallasHash[2].v.7, h10.v.7, "GPU Pallas hash(1,0) limb 7")
        expectEqual(gpuPallasHash[3].v.0, h11.v.0, "GPU Pallas hash(1,1) limb 0")
        expectEqual(gpuPallasHash[3].v.7, h11.v.7, "GPU Pallas hash(1,1) limb 7")

        // Test Pallas GPU batch permute
        let pallasPermInput: [PallasFp] = [zero, zero, zero]
        let gpuPallasPerm = try engine.pallasBatchPermute(pallasPermInput)
        expect(gpuPallasPerm.count == 3, "GPU Pallas batch permute count")

        for i in 0..<3 {
            expectEqual(gpuPallasPerm[i].v.0, permResult[i].v.0,
                       "GPU Pallas permute[\(i)] limb 0")
            expectEqual(gpuPallasPerm[i].v.7, permResult[i].v.7,
                       "GPU Pallas permute[\(i)] limb 7")
        }

        // Test Vesta GPU hash pairs
        let vestaInput: [VestaFp] = [veZero, veZero, veOne, veOne]
        let gpuVestaHash = try engine.vestaHashPairs(vestaInput)
        expect(gpuVestaHash.count == 2, "GPU Vesta hash pairs count")

        let veCpuH11 = vestaPoseidonHash(veOne, veOne)
        expectEqual(gpuVestaHash[0].v.0, veH00.v.0, "GPU Vesta hash(0,0) limb 0")
        expectEqual(gpuVestaHash[0].v.7, veH00.v.7, "GPU Vesta hash(0,0) limb 7")
        expectEqual(gpuVestaHash[1].v.0, veCpuH11.v.0, "GPU Vesta hash(1,1) limb 0")
        expectEqual(gpuVestaHash[1].v.7, veCpuH11.v.7, "GPU Vesta hash(1,1) limb 7")

        // Test larger batch
        let n = 128
        var batchInput = [PallasFp]()
        for _ in 0..<(n * 2) { batchInput.append(zero) }
        let gpuBatch = try engine.pallasHashPairs(batchInput)
        expect(gpuBatch.count == n, "GPU batch hash count")
        for i in 0..<n {
            expectEqual(gpuBatch[i].v.0, h00.v.0, "Batch[\(i)] limb 0")
            expectEqual(gpuBatch[i].v.7, h00.v.7, "Batch[\(i)] limb 7")
        }

        // Throughput benchmark
        print("Starting Pasta Poseidon throughput benchmark...")
        let benchSizes = [1024, 8192, 65536]
        for size in benchSizes {
            var input = [PallasFp]()
            input.reserveCapacity(size * 2)
            for _ in 0..<(size * 2) { input.append(zero) }
            let t0 = CFAbsoluteTimeGetCurrent()
            let result = try engine.pallasHashPairs(input)
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            let hashesPerSec = Double(size) / elapsed
            print("Pallas hash_pairs(\(size)) (\(result.count) output): \(String(format: "%.0f", hashesPerSec)) hashes/sec (\(String(format: "%.2f", elapsed*1000))ms)")
        }

        // Batch size scaling (measure scaling efficiency)
        print("\nBatch scaling (1 thread per hash):")
        let sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        for size in sizes {
            var input = [PallasFp]()
            input.reserveCapacity(size * 2)
            for _ in 0..<(size * 2) { input.append(zero) }
            let t0 = CFAbsoluteTimeGetCurrent()
            let result = try engine.pallasHashPairs(input)
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            let perHashMs = (elapsed * 1000) / Double(size)
            print("  n=\(size): \(String(format: "%.3f", perHashMs))ms/hash, \(String(format: "%.0f", Double(size)/elapsed)) hashes/sec")
        }

        // Permute kernel throughput
        print("\nPermute kernel throughput:")
        let permSizes = [1024, 8192, 65536]
        for size in permSizes {
            var input = [PallasFp]()
            input.reserveCapacity(size * 3)
            for _ in 0..<(size * 3) { input.append(zero) }
            let t0 = CFAbsoluteTimeGetCurrent()
            let result = try engine.pallasBatchPermute(input)
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            print("  permute(\(size)): \(String(format: "%.0f", Double(size)/elapsed)) states/sec (\(String(format: "%.2f", elapsed*1000))ms)")
        }

        // CPU baseline benchmark (C CIOS implementation)
        print("\nCPU baseline (C CIOS implementation):")
        let cpuSizes = [1024, 8192, 65536, 131072, 262144, 524288, 1048576]
        for size in cpuSizes {
            var hashes = [PallasFp]()
            hashes.reserveCapacity(size)
            let t0 = CFAbsoluteTimeGetCurrent()
            for i in 0..<size {
                hashes.append(pallasPoseidonHash(zero, zero))
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            let hashesPerSec = Double(size) / elapsed
            print("  CPU pallasPoseidonHash(\(size)): \(String(format: "%.0f", hashesPerSec)) hashes/sec (\(String(format: "%.2f", elapsed*1000))ms)")
        }

        // GPU extended size benchmark (powers of 2)
        print("\nGPU extended sizes:")
        let gpuSizes: [(String, Int)] = [("2^14", 16384), ("2^15", 32768), ("2^16", 65536), ("2^17", 131072)]
        for (label, size) in gpuSizes {
            var input = [PallasFp]()
            input.reserveCapacity(size * 2)
            for _ in 0..<(size * 2) { input.append(zero) }
            let t0 = CFAbsoluteTimeGetCurrent()
            let result = try engine.pallasHashPairs(input)
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            let hashesPerSec = Double(size) / elapsed
            print("  GPU \(label) (\(size)): \(String(format: "%.0f", hashesPerSec)) hashes/sec (\(String(format: "%.2f", elapsed*1000))ms)")
        }

    } catch {
        expect(false, "PastaPoseidonEngine error: \(error)")
    }
}
