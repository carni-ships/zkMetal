// CPU-optimized benchmark — compares vanilla CPU, parallel CPU, and GPU
import zkMetal
import Foundation

public func runCPUBench() {
    print("=== CPU-Optimized Benchmarks ===")
    print("Cores: \(ProcessInfo.processInfo.activeProcessorCount)")

    // --- NTT BN254 Fr ---
    print("\n--- NTT BN254 Fr: Vanilla vs Parallel CPU vs GPU ---")
    do {
        let engine = try NTTEngine()
        let sizes = [14, 16, 18, 20]
        var rng: UInt64 = 0xDEAD_BEEF

        for logN in sizes {
            let n = 1 << logN
            var data = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                data[i] = frFromInt(rng >> 32)
            }

            // Vanilla CPU
            let cpuT0 = CFAbsoluteTimeGetCurrent()
            let cpuResult = NTTEngine.cpuNTT(data, logN: logN)
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000

            // Parallel CPU (warmup + timed)
            let _ = parallelNTT_Fr(data, logN: logN)
            var parTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = parallelNTT_Fr(data, logN: logN)
                parTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            parTimes.sort()
            let parTime = parTimes[2]

            // Verify parallel matches vanilla
            let parResult = parallelNTT_Fr(data, logN: logN)
            var match = true
            for i in 0..<n {
                if frToInt(cpuResult[i]) != frToInt(parResult[i]) { match = false; break }
            }

            // GPU
            let dataBuf = engine.device.makeBuffer(
                length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr>.stride)
            }
            for _ in 0..<3 { try engine.ntt(data: dataBuf, logN: logN) }
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr>.stride)
            }
            var gpuTimes = [Double]()
            for _ in 0..<5 {
                data.withUnsafeBytes { src in
                    memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr>.stride)
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                try engine.ntt(data: dataBuf, logN: logN)
                gpuTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            gpuTimes.sort()
            let gpuTime = gpuTimes[2]

            let parSpeedup = cpuTime / parTime
            let gpuSpeedup = cpuTime / gpuTime
            let correct = match ? "ok" : "MISMATCH"
            print(String(format: "  2^%-2d | Vanilla: %8.1fms | Parallel: %7.1fms (%.1fx) | GPU: %6.2fms (%.0fx) [%@]",
                        logN, cpuTime, parTime, parSpeedup, gpuTime, gpuSpeedup, correct))
        }
    } catch {
        print("NTT Error: \(error)")
    }

    // --- NTT BabyBear ---
    print("\n--- NTT BabyBear: Vanilla vs Parallel CPU vs GPU ---")
    do {
        let engine = try BabyBearNTTEngine()
        let sizes = [16, 18, 20, 22]
        var rng: UInt32 = 0xDEAD_BEEF

        for logN in sizes {
            let n = 1 << logN
            var data = [Bb](repeating: Bb.zero, count: n)
            for i in 0..<n {
                rng = rng &* 1664525 &+ 1013904223
                data[i] = Bb(v: rng % Bb.P)
            }

            let cpuT0 = CFAbsoluteTimeGetCurrent()
            let cpuResult = BabyBearNTTEngine.cpuNTT(data, logN: logN)
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000

            let _ = parallelNTT_Bb(data, logN: logN)
            var parTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = parallelNTT_Bb(data, logN: logN)
                parTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            parTimes.sort()
            let parTime = parTimes[2]

            let parResult = parallelNTT_Bb(data, logN: logN)
            var match = true
            for i in 0..<n { if cpuResult[i].v != parResult[i].v { match = false; break } }

            // GPU
            let dataBuf = engine.device.makeBuffer(
                length: n * MemoryLayout<Bb>.stride, options: .storageModeShared)!
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Bb>.stride)
            }
            for _ in 0..<3 { try engine.ntt(data: dataBuf, logN: logN) }
            var gpuTimes = [Double]()
            for _ in 0..<5 {
                data.withUnsafeBytes { src in
                    memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Bb>.stride)
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                try engine.ntt(data: dataBuf, logN: logN)
                gpuTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            gpuTimes.sort()
            let gpuTime = gpuTimes[2]

            let parSpeedup = cpuTime / parTime
            let gpuSpeedup = cpuTime / gpuTime
            let correct = match ? "ok" : "MISMATCH"
            print(String(format: "  2^%-2d | Vanilla: %7.1fms | Parallel: %6.1fms (%.1fx) | GPU: %5.2fms (%.0fx) [%@]",
                        logN, cpuTime, parTime, parSpeedup, gpuTime, gpuSpeedup, correct))
        }
    } catch {
        print("BabyBear NTT Error: \(error)")
    }

    // --- NTT BabyBear NEON ---
    print("\n--- NTT BabyBear: NEON C vs Vanilla CPU vs GPU ---")
    do {
        let engine = try BabyBearNTTEngine()
        let sizes = [16, 18, 20, 22]
        var rng: UInt32 = 0xDEAD_BEEF

        for logN in sizes {
            let n = 1 << logN
            var data = [Bb](repeating: Bb.zero, count: n)
            for i in 0..<n {
                rng = rng &* 1664525 &+ 1013904223
                data[i] = Bb(v: rng % Bb.P)
            }

            // Vanilla CPU
            let cpuT0 = CFAbsoluteTimeGetCurrent()
            let cpuResult = BabyBearNTTEngine.cpuNTT(data, logN: logN)
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000

            // NEON C (warmup + timed)
            let _ = neonNTT_Bb(data, logN: logN)
            var neonTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = neonNTT_Bb(data, logN: logN)
                neonTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            neonTimes.sort()
            let neonTime = neonTimes[2]

            // Verify NEON matches vanilla
            let neonResult = neonNTT_Bb(data, logN: logN)
            var match = true
            for i in 0..<n { if cpuResult[i].v != neonResult[i].v { match = false; break } }

            // NEON round-trip correctness
            let neonRoundtrip = neonINTT_Bb(neonResult, logN: logN)
            var rtMatch = true
            for i in 0..<n { if data[i].v != neonRoundtrip[i].v { rtMatch = false; break } }

            // GPU
            let dataBuf = engine.device.makeBuffer(
                length: n * MemoryLayout<Bb>.stride, options: .storageModeShared)!
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Bb>.stride)
            }
            for _ in 0..<3 { try engine.ntt(data: dataBuf, logN: logN) }
            var gpuTimes = [Double]()
            for _ in 0..<5 {
                data.withUnsafeBytes { src in
                    memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Bb>.stride)
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                try engine.ntt(data: dataBuf, logN: logN)
                gpuTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            gpuTimes.sort()
            let gpuTime = gpuTimes[2]

            let neonSpeedup = cpuTime / neonTime
            let gpuSpeedup = cpuTime / gpuTime
            let correct = (match && rtMatch) ? "ok" : (match ? "rt-FAIL" : "MISMATCH")
            print(String(format: "  2^%-2d | Vanilla: %7.1fms | NEON: %6.1fms (%.1fx) | GPU: %5.2fms (%.0fx) [%@]",
                        logN, cpuTime, neonTime, neonSpeedup, gpuTime, gpuSpeedup, correct))
        }
    } catch {
        print("BabyBear NEON NTT Error: \(error)")
    }

    // --- NTT Goldilocks ---
    print("\n--- NTT Goldilocks: Vanilla vs Parallel CPU vs GPU ---")
    do {
        let engine = try GoldilocksNTTEngine()
        let sizes = [16, 18, 20, 22]
        var rng: UInt64 = 0xCAFE_BABE

        for logN in sizes {
            let n = 1 << logN
            var data = [Gl](repeating: Gl.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                data[i] = Gl(v: rng % Gl.P)
            }

            let cpuT0 = CFAbsoluteTimeGetCurrent()
            let cpuResult = GoldilocksNTTEngine.cpuNTT(data, logN: logN)
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000

            let _ = parallelNTT_Gl(data, logN: logN)
            var parTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = parallelNTT_Gl(data, logN: logN)
                parTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            parTimes.sort()
            let parTime = parTimes[2]

            let parResult = parallelNTT_Gl(data, logN: logN)
            var match = true
            for i in 0..<n { if cpuResult[i].v != parResult[i].v { match = false; break } }

            let dataBuf = engine.device.makeBuffer(
                length: n * MemoryLayout<Gl>.stride, options: .storageModeShared)!
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Gl>.stride)
            }
            for _ in 0..<3 { try engine.ntt(data: dataBuf, logN: logN) }
            var gpuTimes = [Double]()
            for _ in 0..<5 {
                data.withUnsafeBytes { src in
                    memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Gl>.stride)
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                try engine.ntt(data: dataBuf, logN: logN)
                gpuTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            gpuTimes.sort()
            let gpuTime = gpuTimes[2]

            let parSpeedup = cpuTime / parTime
            let gpuSpeedup = cpuTime / gpuTime
            let correct = match ? "ok" : "MISMATCH"
            print(String(format: "  2^%-2d | Vanilla: %7.1fms | Parallel: %6.1fms (%.1fx) | GPU: %5.2fms (%.0fx) [%@]",
                        logN, cpuTime, parTime, parSpeedup, gpuTime, gpuSpeedup, correct))
        }
    } catch {
        print("Goldilocks NTT Error: \(error)")
    }

    // --- NTT Goldilocks: optimized C ---
    print("\n--- NTT Goldilocks: Optimized C vs Vanilla CPU vs GPU ---")
    do {
        let engine = try GoldilocksNTTEngine()
        let sizes = [16, 18, 20, 22]
        var rng: UInt64 = 0xCAFE_BABE

        for logN in sizes {
            let n = 1 << logN
            var data = [Gl](repeating: Gl.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                data[i] = Gl(v: rng % Gl.P)
            }

            let cpuT0 = CFAbsoluteTimeGetCurrent()
            let cpuResult = GoldilocksNTTEngine.cpuNTT(data, logN: logN)
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000

            // Optimized C (warmup + timed)
            let _ = cNTT_Gl(data, logN: logN)
            var cTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = cNTT_Gl(data, logN: logN)
                cTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            cTimes.sort()
            let cTime = cTimes[2]

            // Verify C matches vanilla
            let cResult = cNTT_Gl(data, logN: logN)
            var match = true
            for i in 0..<n { if cpuResult[i].v != cResult[i].v { match = false; break } }

            // Round-trip
            let cRt = cINTT_Gl(cResult, logN: logN)
            var rtMatch = true
            for i in 0..<n { if data[i].v != cRt[i].v { rtMatch = false; break } }

            // GPU
            let dataBuf = engine.device.makeBuffer(
                length: n * MemoryLayout<Gl>.stride, options: .storageModeShared)!
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Gl>.stride)
            }
            for _ in 0..<3 { try engine.ntt(data: dataBuf, logN: logN) }
            var gpuTimes = [Double]()
            for _ in 0..<5 {
                data.withUnsafeBytes { src in
                    memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Gl>.stride)
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                try engine.ntt(data: dataBuf, logN: logN)
                gpuTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            gpuTimes.sort()
            let gpuTime = gpuTimes[2]

            let cSpeedup = cpuTime / cTime
            let gpuSpeedup = cpuTime / gpuTime
            let correct = (match && rtMatch) ? "ok" : (match ? "rt-FAIL" : "MISMATCH")
            print(String(format: "  2^%-2d | Vanilla: %7.1fms | C: %7.1fms (%.1fx) | GPU: %5.2fms (%.0fx) [%@]",
                        logN, cpuTime, cTime, cSpeedup, gpuTime, gpuSpeedup, correct))
        }
    } catch {
        print("Goldilocks C NTT Error: \(error)")
    }

    // --- NTT BN254 Fr: optimized C ---
    print("\n--- NTT BN254 Fr: Optimized C vs Vanilla CPU vs GPU ---")
    do {
        let engine = try NTTEngine()
        let sizes = [14, 16, 18]  // Skip 2^20 (too slow for vanilla)
        var rng: UInt64 = 0xDEAD_BEEF

        for logN in sizes {
            let n = 1 << logN
            var data = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                data[i] = frFromInt(rng >> 32)
            }

            let cpuT0 = CFAbsoluteTimeGetCurrent()
            let cpuResult = NTTEngine.cpuNTT(data, logN: logN)
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000

            // Optimized C (warmup + timed)
            let _ = cNTT_Fr(data, logN: logN)
            var cTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = cNTT_Fr(data, logN: logN)
                cTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            cTimes.sort()
            let cTime = cTimes[2]

            // Verify C matches vanilla
            let cResult = cNTT_Fr(data, logN: logN)
            var match = true
            for i in 0..<n {
                if frToInt(cpuResult[i]) != frToInt(cResult[i]) { match = false; break }
            }

            // Round-trip
            let cRt = cINTT_Fr(cResult, logN: logN)
            var rtMatch = true
            for i in 0..<n {
                if frToInt(data[i]) != frToInt(cRt[i]) { rtMatch = false; break }
            }

            // GPU
            let dataBuf = engine.device.makeBuffer(
                length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!
            data.withUnsafeBytes { src in
                memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr>.stride)
            }
            for _ in 0..<3 { try engine.ntt(data: dataBuf, logN: logN) }
            var gpuTimes = [Double]()
            for _ in 0..<5 {
                data.withUnsafeBytes { src in
                    memcpy(dataBuf.contents(), src.baseAddress!, n * MemoryLayout<Fr>.stride)
                }
                let t0 = CFAbsoluteTimeGetCurrent()
                try engine.ntt(data: dataBuf, logN: logN)
                gpuTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            gpuTimes.sort()
            let gpuTime = gpuTimes[2]

            let cSpeedup = cpuTime / cTime
            let gpuSpeedup = cpuTime / gpuTime
            let correct = (match && rtMatch) ? "ok" : (match ? "rt-FAIL" : "MISMATCH")
            print(String(format: "  2^%-2d | Vanilla: %8.1fms | C: %8.1fms (%.1fx) | GPU: %6.2fms (%.0fx) [%@]",
                        logN, cpuTime, cTime, cSpeedup, gpuTime, gpuSpeedup, correct))
        }
    } catch {
        print("BN254 Fr C NTT Error: \(error)")
    }

    // --- MSM BN254 ---
    print("\n--- MSM BN254: Vanilla vs Parallel CPU (Pippenger) vs GPU ---")
    do {
        let engine = try MetalMSM()

        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let gProj = pointFromAffine(PointAffine(x: gx, y: gy))

        let logSizes = [8, 10, 12, 14]
        let maxN = 1 << logSizes.last!

        var projPoints = [PointProjective]()
        projPoints.reserveCapacity(maxN)
        var acc = gProj
        for _ in 0..<maxN {
            projPoints.append(acc)
            acc = pointAdd(acc, gProj)
        }
        let allPoints = batchToAffine(projPoints)
        projPoints = []

        var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
        var allScalars = [[UInt32]]()
        allScalars.reserveCapacity(maxN)
        for _ in 0..<maxN {
            var limbs = [UInt32](repeating: 0, count: 8)
            for j in 0..<8 {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
            }
            allScalars.append(limbs)
        }

        for logN in logSizes {
            let n = 1 << logN
            let points = Array(allPoints.prefix(n))
            let scalars = Array(allScalars.prefix(n))

            // Vanilla CPU (sequential scalar mul)
            let vanillaT0 = CFAbsoluteTimeGetCurrent()
            var vanillaResult = pointIdentity()
            let projPts = points.map { pointFromAffine($0) }
            for i in 0..<n {
                let scalar = frFromLimbs(scalars[i])
                vanillaResult = pointAdd(vanillaResult, pointScalarMul(projPts[i], scalar))
            }
            let vanillaTime = (CFAbsoluteTimeGetCurrent() - vanillaT0) * 1000

            // Parallel CPU (Pippenger)
            let _ = parallelMSM(points: points, scalars: scalars)
            var parTimes = [Double]()
            for _ in 0..<3 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = parallelMSM(points: points, scalars: scalars)
                parTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            parTimes.sort()
            let parTime = parTimes[1]

            // Verify: parallel result matches vanilla
            let parResult = parallelMSM(points: points, scalars: scalars)
            let match = pointEqual(vanillaResult, parResult) ? "ok" : "MISMATCH"

            // GPU
            let _ = try engine.msm(points: points, scalars: scalars)
            var gpuTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.msm(points: points, scalars: scalars)
                gpuTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            gpuTimes.sort()
            let gpuTime = gpuTimes[2]

            let parSpeedup = vanillaTime / parTime
            let gpuSpeedup = vanillaTime / gpuTime
            print(String(format: "  2^%-2d | Vanilla: %8.1fms | Pippenger: %7.1fms (%.1fx) | GPU: %6.1fms (%.0fx) [%@]",
                        logN, vanillaTime, parTime, parSpeedup, gpuTime, gpuSpeedup, match))
        }
    } catch {
        print("MSM Error: \(error)")
    }

    // --- Batch Hashing ---
    print("\n--- Poseidon2 Batch: Vanilla vs Parallel CPU vs GPU ---")
    do {
        let engine = try Poseidon2Engine()
        let sizes = [12, 14, 16]

        for logN in sizes {
            let n = 1 << logN
            var pairs = [(Fr, Fr)]()
            for i in 0..<n {
                pairs.append((frFromInt(UInt64(i)), frFromInt(UInt64(i + n))))
            }

            // Vanilla CPU (sequential)
            let vanillaT0 = CFAbsoluteTimeGetCurrent()
            var vanillaResults = [Fr]()
            vanillaResults.reserveCapacity(n)
            for i in 0..<n {
                vanillaResults.append(poseidon2Hash(pairs[i].0, pairs[i].1))
            }
            let vanillaTime = (CFAbsoluteTimeGetCurrent() - vanillaT0) * 1000

            // Parallel CPU
            let _ = parallelPoseidon2Batch(pairs)
            var parTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = parallelPoseidon2Batch(pairs)
                parTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            parTimes.sort()
            let parTime = parTimes[2]

            // GPU
            var inputData = [Fr]()
            for (a, b) in pairs { inputData.append(a); inputData.append(b) }
            let _ = try engine.hashPairs(inputData)
            var gpuTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.hashPairs(inputData)
                gpuTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            gpuTimes.sort()
            let gpuTime = gpuTimes[2]

            let parSpeedup = vanillaTime / parTime
            let gpuSpeedup = vanillaTime / gpuTime
            print(String(format: "  2^%-2d | Vanilla: %7.0fms | Parallel: %6.0fms (%.1fx) | GPU: %5.1fms (%.0fx)",
                        logN, vanillaTime, parTime, parSpeedup, gpuTime, gpuSpeedup))
        }
    } catch {
        print("Poseidon2 Error: \(error)")
    }

    print("\n--- Keccak-256 Batch: Vanilla vs Parallel CPU vs GPU ---")
    do {
        let engine = try Keccak256Engine()
        let sizes = [14, 16, 18]

        for logN in sizes {
            let n = 1 << logN
            var inputs = [[UInt8]]()
            for i in 0..<n {
                var block = [UInt8](repeating: 0, count: 64)
                let val = UInt64(i)
                withUnsafeBytes(of: val) { block.replaceSubrange(0..<8, with: $0) }
                inputs.append(block)
            }

            // Vanilla CPU
            let vanillaT0 = CFAbsoluteTimeGetCurrent()
            for i in 0..<n { let _ = keccak256(inputs[i]) }
            let vanillaTime = (CFAbsoluteTimeGetCurrent() - vanillaT0) * 1000

            // Parallel CPU
            let _ = parallelKeccak256Batch(inputs)
            var parTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = parallelKeccak256Batch(inputs)
                parTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            parTimes.sort()
            let parTime = parTimes[2]

            // GPU
            let flatInput = inputs.flatMap { $0 }
            let _ = try engine.hash64(flatInput)
            var gpuTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.hash64(flatInput)
                gpuTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            gpuTimes.sort()
            let gpuTime = gpuTimes[2]

            let parSpeedup = vanillaTime / parTime
            let gpuSpeedup = vanillaTime / gpuTime
            print(String(format: "  2^%-2d | Vanilla: %7.0fms | Parallel: %6.0fms (%.1fx) | GPU: %5.1fms (%.0fx)",
                        logN, vanillaTime, parTime, parSpeedup, gpuTime, gpuSpeedup))
        }
    } catch {
        print("Keccak Error: \(error)")
    }
}
