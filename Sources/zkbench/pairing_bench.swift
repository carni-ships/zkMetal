// BN254 Pairing Benchmark: CPU vs GPU comparison
// Tests: single pairing, Groth16 (4 pairings), batch (16 pairings)

import Foundation
import Metal
import zkMetal

public func runPairingBench() {
    fputs("\n--- BN254 Pairing Benchmark (CPU vs GPU) ---\n", stderr)

    let g1 = bn254G1Generator()
    let g2 = bn254G2Generator()

    // Generate test points: [2^k]G1 and [2^k]G2 for various k
    func makeTestPairs(n: Int) -> [(PointAffine, G2AffinePoint)] {
        var pairs = [(PointAffine, G2AffinePoint)]()
        var g1Proj = pointFromAffine(g1)
        var g2Proj = g2FromAffine(g2)
        for _ in 0..<n {
            let g1Aff = pointToAffine(g1Proj)!
            let g2Aff = g2ToAffine(g2Proj)!
            pairs.append((g1Aff, g2Aff))
            g1Proj = pointDouble(g1Proj)
            g2Proj = g2Double(g2Proj)
        }
        return pairs
    }

    // ---- Correctness check first ----
    fputs("\n[1] Correctness\n", stderr)

    let cpuPair = bn254Pairing(g1, g2)
    fputs("  CPU e(G1,G2) != 1: \(!fp12Equal(cpuPair, .one) ? "PASS" : "FAIL")\n", stderr)

    do {
        let engine = try BN254PairingEngine()
        fputs("  Metal shader compiled successfully\n", stderr)

        // GPU single pairing
        let gpuResults = try engine.batchPairing(pairs: [(g1, g2)])
        let gpuPair = gpuResults[0]
        let gpuMatchesCPU = fp12Equal(gpuPair, cpuPair)
        fputs("  GPU e(G1,G2) matches CPU: \(gpuMatchesCPU ? "PASS" : "FAIL")\n", stderr)

        if !gpuMatchesCPU {
            let cpuC0 = fpToInt(cpuPair.c0.c0.c0)
            let gpuC0 = fpToInt(gpuPair.c0.c0.c0)
            fputs("    CPU c0.c0.c0[0]: \(String(format: "0x%016llx", cpuC0[0]))\n", stderr)
            fputs("    GPU c0.c0.c0[0]: \(String(format: "0x%016llx", gpuC0[0]))\n", stderr)

            // Test just the Miller loop
            let cpuML = bn254MillerLoop(g1, g2)
            let gpuMLs = try engine.batchMillerLoop(pairs: [(g1, g2)])
            let gpuML = gpuMLs[0]
            let mlMatch = fp12Equal(gpuML, cpuML)
            fputs("    Miller loop matches: \(mlMatch ? "PASS" : "FAIL")\n", stderr)
            if !mlMatch {
                let cpuMLC0 = fpToInt(cpuML.c0.c0.c0)
                let gpuMLC0 = fpToInt(gpuML.c0.c0.c0)
                fputs("    CPU ML c0.c0.c0[0]: \(String(format: "0x%016llx", cpuMLC0[0]))\n", stderr)
                fputs("    GPU ML c0.c0.c0[0]: \(String(format: "0x%016llx", gpuMLC0[0]))\n", stderr)
            }
        }

        // GPU pairing check: e(G,-H)*e(G,H) = 1
        let negG2 = g2NegateAffine(g2)
        let checkResult = try engine.pairingCheck(pairs: [(g1, g2), (g1, negG2)])
        fputs("  GPU pairing check e(G,H)*e(G,-H)=1: \(checkResult ? "PASS" : "FAIL")\n", stderr)

        // Bilinearity: e(aP, Q) = e(P, aQ)
        let a = frFromInt(7)
        let aG1 = pointToAffine(pointScalarMul(pointFromAffine(g1), a))!
        let aG2 = g2ToAffine(g2ScalarMul(g2FromAffine(g2), frToInt(a)))!
        let gpuLhs = try engine.batchPairing(pairs: [(aG1, g2)])
        let gpuRhs = try engine.batchPairing(pairs: [(g1, aG2)])
        let bilinearOk = fp12Equal(gpuLhs[0], gpuRhs[0])
        fputs("  GPU bilinear e(7G,H)==e(G,7H): \(bilinearOk ? "PASS" : "FAIL")\n", stderr)

        // ---- Benchmark ----
        fputs("\n[2] Performance\n", stderr)

        for n in [1, 4, 16] {
            let pairs = makeTestPairs(n: n)

            // C pairing check (our optimized C implementation)
            var cTimes = [Double]()
            for _ in 0..<10 {
                let t0 = CFAbsoluteTimeGetCurrent()
                _ = cBN254PairingCheck(pairs)
                cTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            cTimes.sort()
            let cMedian = cTimes[cTimes.count / 2]

            // Swift benchmark: multi-Miller + single final exp
            let cpuRuns = n <= 4 ? 3 : 1
            var cpuTimes = [Double]()
            for _ in 0..<cpuRuns {
                let t0 = CFAbsoluteTimeGetCurrent()
                var cpuF = Fp12.one
                for (p, q) in pairs {
                    cpuF = fp12Mul(cpuF, bn254MillerLoop(p, q))
                }
                _ = bn254FinalExponentiation(cpuF)
                cpuTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            cpuTimes.sort()
            let cpuMedian = cpuTimes[cpuTimes.count / 2]

            // GPU: parallel Miller loops + CPU product + CPU final exp
            let gpuRuns = 5
            // Warmup
            let _ = try engine.multiMillerPairing(pairs: pairs)

            var gpuTimes = [Double]()
            for _ in 0..<gpuRuns {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.multiMillerPairing(pairs: pairs)
                gpuTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            gpuTimes.sort()
            let gpuMedian = gpuTimes[gpuTimes.count / 2]

            let speedup = cpuMedian / gpuMedian
            fputs(String(format: "  n=%2d: C %5.1fms | Swift %7.1fms | GPU %7.1fms | C/GPU %.1fx\n",
                        n, cMedian, cpuMedian, gpuMedian, gpuMedian / cMedian), stderr)
        }

        // ---- Groth16 Verification scenario ----
        fputs("\n[3] Groth16 Verification (4 pairings)\n", stderr)

        let groth16Pairs = makeTestPairs(n: 4)

        // CPU Groth16 verify timing
        var cpuVerifyTimes = [Double]()
        for _ in 0..<3 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = bn254PairingCheck(groth16Pairs)
            cpuVerifyTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        cpuVerifyTimes.sort()
        let cpuVerifyMedian = cpuVerifyTimes[cpuVerifyTimes.count / 2]

        // GPU Groth16 verify
        // Warmup
        let _ = try engine.pairingCheck(pairs: groth16Pairs)
        var gpuVerifyTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try engine.pairingCheck(pairs: groth16Pairs)
            gpuVerifyTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        gpuVerifyTimes.sort()
        let gpuVerifyMedian = gpuVerifyTimes[gpuVerifyTimes.count / 2]

        let verifySpeedup = cpuVerifyMedian / gpuVerifyMedian
        fputs(String(format: "  CPU verify: %7.1fms\n", cpuVerifyMedian), stderr)
        fputs(String(format: "  GPU verify: %7.1fms (%.2fx)\n", gpuVerifyMedian, verifySpeedup), stderr)

        // Component breakdown
        fputs("\n[4] Component Breakdown\n", stderr)

        // CPU single Miller loop
        let cpuML0 = CFAbsoluteTimeGetCurrent()
        let ml = bn254MillerLoop(g1, g2)
        let cpuMLTime = (CFAbsoluteTimeGetCurrent() - cpuML0) * 1000
        fputs(String(format: "  CPU Miller loop:    %7.1fms\n", cpuMLTime), stderr)

        // CPU final exp
        let cpuFE0 = CFAbsoluteTimeGetCurrent()
        _ = bn254FinalExponentiation(ml)
        let cpuFETime = (CFAbsoluteTimeGetCurrent() - cpuFE0) * 1000
        fputs(String(format: "  CPU final exp:      %7.1fms\n", cpuFETime), stderr)

        // GPU Miller loops (4 parallel)
        var gpuMLTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try engine.batchMillerLoop(pairs: groth16Pairs)
            gpuMLTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        gpuMLTimes.sort()
        fputs(String(format: "  GPU 4x Miller loop: %7.1fms (vs CPU 4x: %.1fms)\n",
                    gpuMLTimes[gpuMLTimes.count / 2], cpuMLTime * 4), stderr)

        // C path comparison: standard vs precomputed
        let cPairs = [(g1, g2), (g1, g2NegateAffine(g2))]
        var cStdTimes = [Double]()
        var cPreTimes = [Double]()
        for _ in 0..<10 {
            var t0 = CFAbsoluteTimeGetCurrent()
            _ = cBN254PairingCheck(cPairs)
            cStdTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            t0 = CFAbsoluteTimeGetCurrent()
            _ = cBN254PairingCheckPrecomp(cPairs)
            cPreTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        cStdTimes.sort(); cPreTimes.sort()
        let cStdMed = cStdTimes[cStdTimes.count / 2]
        let cPreMed = cPreTimes[cPreTimes.count / 2]
        fputs(String(format: "  C standard (2-pair):  %5.2fms\n", cStdMed), stderr)
        fputs(String(format: "  C precomp  (2-pair):  %5.2fms (%.1fx)\n", cPreMed, cStdMed / cPreMed), stderr)

        // [5] GPU Profiling Breakdown
        fputs("\n[5] GPU Phase Profiling (16 pairings)\n", stderr)
        let profPairs = makeTestPairs(n: 16)
        engine.profilingEnabled = true
        engine.resetProfiling()
        // Warmup
        _ = try engine.multiMillerPairing(pairs: profPairs)
        engine.resetProfiling()
        for _ in 0..<5 {
            _ = try engine.multiMillerPairing(pairs: profPairs)
        }
        engine.printProfile()
        engine.profilingEnabled = false

    } catch {
        fputs("  Error: \(error)\n", stderr)
    }
}
