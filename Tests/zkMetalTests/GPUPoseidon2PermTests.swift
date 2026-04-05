// GPUPoseidon2PermTests — Tests for GPU Poseidon2 permutation engine
// Tests width-3, width-4, compression, determinism, and performance.

import zkMetal
import Foundation

public func runGPUPoseidon2PermTests() {
    suite("GPU Poseidon2 Permutation")

    guard let engine = try? GPUPoseidon2PermutationEngine() else {
        print("  [SKIP] No GPU available")
        return
    }

    // =========================================================================
    // Test 1: Width-3 GPU permutation matches CPU Poseidon2
    // =========================================================================
    do {
        let n = 128
        var states = [[Fr]]()
        for i in 0..<n {
            states.append([frFromInt(UInt64(i * 3 + 1)),
                          frFromInt(UInt64(i * 3 + 2)),
                          frFromInt(UInt64(i * 3 + 3))])
        }

        let cpuResults = states.map { poseidon2Permutation($0) }
        let gpuResults = try engine.permute(states: states)

        var ok = true
        for i in 0..<n {
            for j in 0..<3 {
                if cpuResults[i][j] != gpuResults[i][j] {
                    ok = false
                    break
                }
            }
            if !ok { break }
        }
        expect(ok, "W3 GPU matches CPU (\(n) permutations)")
    } catch {
        expect(false, "W3 GPU permutation error: \(error)")
    }

    // =========================================================================
    // Test 2: Width-3 GPU permutation with known test vector (all zeros)
    // =========================================================================
    do {
        let zeroState = [[Fr.zero, Fr.zero, Fr.zero]]
        let cpuResult = poseidon2Permutation(zeroState[0])

        // Use CPU path threshold override: pass 128 states to force GPU
        var states = [[Fr]]()
        for _ in 0..<128 { states.append([Fr.zero, Fr.zero, Fr.zero]) }
        let gpuResults = try engine.permute(states: states)

        var ok = true
        for j in 0..<3 {
            if cpuResult[j] != gpuResults[0][j] { ok = false; break }
        }
        expect(ok, "W3 GPU matches CPU on zero state")

        // All GPU results should be identical (same input)
        var allSame = true
        for i in 1..<128 {
            for j in 0..<3 {
                if gpuResults[0][j] != gpuResults[i][j] { allSame = false; break }
            }
            if !allSame { break }
        }
        expect(allSame, "W3 all-zero batch produces identical results")
    } catch {
        expect(false, "W3 zero state error: \(error)")
    }

    // =========================================================================
    // Test 3: Batch compression matches sequential CPU compression
    // =========================================================================
    do {
        let n = 128
        var pairs = [(Fr, Fr)]()
        for i in 0..<n {
            pairs.append((frFromInt(UInt64(i + 1)), frFromInt(UInt64(i + n + 1))))
        }

        let cpuResults = pairs.map { poseidon2Hash($0.0, $0.1) }
        let gpuResults = try engine.compress(pairs: pairs)

        var ok = true
        for i in 0..<n {
            if cpuResults[i] != gpuResults[i] {
                ok = false
                break
            }
        }
        expect(ok, "GPU compress matches CPU (\(n) pairs)")
    } catch {
        expect(false, "GPU compress error: \(error)")
    }

    // =========================================================================
    // Test 4: Width-3 and width-4 produce different outputs
    // =========================================================================
    do {
        // Use the same input values for both widths
        let w3State = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let w4State = [frFromInt(1), frFromInt(2), frFromInt(3), Fr.zero]

        let cpuW3 = poseidon2Permutation(w3State)
        let cpuW4 = engine.cpuPermuteW4(w4State)

        // They must be different (different parameters, different matrices)
        var different = false
        // Compare first 3 elements
        for i in 0..<3 {
            if cpuW3[i] != cpuW4[i] { different = true; break }
        }
        expect(different, "W3 and W4 produce different outputs")

        // W4 output should be non-trivial
        var w4NonTrivial = false
        for i in 0..<4 {
            if cpuW4[i] != Fr.zero { w4NonTrivial = true; break }
        }
        expect(w4NonTrivial, "W4 output is non-trivial")
    }

    // =========================================================================
    // Test 5: Width-4 GPU matches CPU
    // =========================================================================
    do {
        let n = 128
        var states = [[Fr]]()
        for i in 0..<n {
            states.append([frFromInt(UInt64(i * 4 + 1)),
                          frFromInt(UInt64(i * 4 + 2)),
                          frFromInt(UInt64(i * 4 + 3)),
                          frFromInt(UInt64(i * 4 + 4))])
        }

        let cpuResults = states.map { engine.cpuPermuteW4($0) }
        let gpuResults = try engine.permuteWidth4(states: states)

        var ok = true
        for i in 0..<n {
            for j in 0..<4 {
                if cpuResults[i][j] != gpuResults[i][j] {
                    ok = false
                    break
                }
            }
            if !ok { break }
        }
        expect(ok, "W4 GPU matches CPU (\(n) permutations)")
    } catch {
        expect(false, "W4 GPU permutation error: \(error)")
    }

    // =========================================================================
    // Test 6: Determinism — permute(permute(x)) is deterministic
    // =========================================================================
    do {
        let n = 128
        var states = [[Fr]]()
        for i in 0..<n {
            states.append([frFromInt(UInt64(i + 100)),
                          frFromInt(UInt64(i + 200)),
                          frFromInt(UInt64(i + 300))])
        }

        let first = try engine.permute(states: states)
        let second = try engine.permute(states: first)
        let secondAgain = try engine.permute(states: first)

        var ok = true
        for i in 0..<n {
            for j in 0..<3 {
                if second[i][j] != secondAgain[i][j] { ok = false; break }
            }
            if !ok { break }
        }
        expect(ok, "permute(permute(x)) is deterministic")
    } catch {
        expect(false, "Determinism test error: \(error)")
    }

    // =========================================================================
    // Test 7: Zero-copy MTLBuffer API (width-3)
    // =========================================================================
    do {
        let n = 256
        var flat = [Fr]()
        for i in 0..<(n * 3) {
            flat.append(frFromInt(UInt64(i + 1)))
        }

        let stride = MemoryLayout<Fr>.stride
        guard let inBuf = engine.device.makeBuffer(length: flat.count * stride,
                                                    options: .storageModeShared) else {
            expect(false, "Failed to create input buffer")
            return
        }
        flat.withUnsafeBytes { src in
            memcpy(inBuf.contents(), src.baseAddress!, flat.count * stride)
        }

        let outBuf = try engine.permuteBuffer(buf: inBuf, count: n, width: 3)

        // Verify against CPU
        let outPtr = outBuf.contents().bindMemory(to: Fr.self, capacity: n * 3)
        let gpuResult = Array(UnsafeBufferPointer(start: outPtr, count: n * 3))

        var ok = true
        for i in 0..<n {
            let cpu = poseidon2Permutation([flat[i*3], flat[i*3+1], flat[i*3+2]])
            for j in 0..<3 {
                if cpu[j] != gpuResult[i*3+j] { ok = false; break }
            }
            if !ok { break }
        }
        expect(ok, "permuteBuffer W3 matches CPU (\(n))")
    } catch {
        expect(false, "MTLBuffer API error: \(error)")
    }

    // =========================================================================
    // Test 8: Zero-copy MTLBuffer API (width-4)
    // =========================================================================
    do {
        let n = 256
        var flat = [Fr]()
        for i in 0..<(n * 4) {
            flat.append(frFromInt(UInt64(i + 1)))
        }

        let stride = MemoryLayout<Fr>.stride
        guard let inBuf = engine.device.makeBuffer(length: flat.count * stride,
                                                    options: .storageModeShared) else {
            expect(false, "Failed to create input buffer")
            return
        }
        flat.withUnsafeBytes { src in
            memcpy(inBuf.contents(), src.baseAddress!, flat.count * stride)
        }

        let outBuf = try engine.permuteBuffer(buf: inBuf, count: n, width: 4)

        // Verify against CPU
        let outPtr = outBuf.contents().bindMemory(to: Fr.self, capacity: n * 4)
        let gpuResult = Array(UnsafeBufferPointer(start: outPtr, count: n * 4))

        var ok = true
        for i in 0..<n {
            let cpu = engine.cpuPermuteW4([flat[i*4], flat[i*4+1], flat[i*4+2], flat[i*4+3]])
            for j in 0..<4 {
                if cpu[j] != gpuResult[i*4+j] { ok = false; break }
            }
            if !ok { break }
        }
        expect(ok, "permuteBuffer W4 matches CPU (\(n))")
    } catch {
        expect(false, "MTLBuffer W4 API error: \(error)")
    }

    // =========================================================================
    // Test 9: CPU fallback for small batches
    // =========================================================================
    do {
        // Test with batch size below GPU threshold (< 64)
        let n = 16
        var states = [[Fr]]()
        for i in 0..<n {
            states.append([frFromInt(UInt64(i + 50)),
                          frFromInt(UInt64(i + 150)),
                          frFromInt(UInt64(i + 250))])
        }

        let cpuResults = states.map { poseidon2Permutation($0) }
        let engineResults = try engine.permute(states: states)  // should use CPU fallback

        var ok = true
        for i in 0..<n {
            for j in 0..<3 {
                if cpuResults[i][j] != engineResults[i][j] { ok = false; break }
            }
            if !ok { break }
        }
        expect(ok, "CPU fallback matches direct CPU (\(n) < threshold)")
    } catch {
        expect(false, "CPU fallback error: \(error)")
    }

    // =========================================================================
    // Test 10: Large batch performance benchmark (2^14 permutations)
    // =========================================================================
    do {
        let logN = 14
        let n = 1 << logN  // 16384
        var states = [[Fr]]()
        states.reserveCapacity(n)
        var seed: UInt64 = 0xDEAD_BEEF
        for _ in 0..<n {
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            let a = frFromInt(seed >> 32)
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            let b = frFromInt(seed >> 32)
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            let c = frFromInt(seed >> 32)
            states.append([a, b, c])
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        let _ = try engine.permute(states: states)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        let perPermutation = elapsed / Double(n) * 1e6  // microseconds

        expect(true, String(format: "W3 2^%d permutations: %.1fms (%.2f us/perm)",
                           logN, elapsed * 1000, perPermutation))

        // Width-4 benchmark
        var states4 = [[Fr]]()
        states4.reserveCapacity(n)
        seed = 0xCAFE_BABE
        for _ in 0..<n {
            var row = [Fr]()
            for _ in 0..<4 {
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                row.append(frFromInt(seed >> 32))
            }
            states4.append(row)
        }

        let t1 = CFAbsoluteTimeGetCurrent()
        let _ = try engine.permuteWidth4(states: states4)
        let elapsed4 = CFAbsoluteTimeGetCurrent() - t1
        let perPerm4 = elapsed4 / Double(n) * 1e6

        expect(true, String(format: "W4 2^%d permutations: %.1fms (%.2f us/perm)",
                           logN, elapsed4 * 1000, perPerm4))
    } catch {
        expect(false, "Benchmark error: \(error)")
    }
}
