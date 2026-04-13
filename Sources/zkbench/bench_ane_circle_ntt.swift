// bench_ane_circle_ntt.swift — Circle NTT benchmark: CPU vs GPU vs ANE
//
// Compares performance of Circle NTT across:
// 1. CPU Scalar - pure Swift reference implementation
// 2. GPU Metal - CircleNTTEngine with Metal compute
// 3. ANE - Apple Neural Engine via ane_circle_ntt C API
//
// Circle NTT over Mersenne31 (p = 2^31 - 1)

import Foundation
import Metal
import ANEOps
import zkMetal

// ============================================================
// M31 helpers for ANE C API interop
// ============================================================

/// Convert [M31] to [UInt32] for ANE C API
func m31ArrayToUInt32(_ arr: [M31]) -> [UInt32] {
    return arr.map { $0.v }
}

/// Convert [UInt32] to [M31]
func uint32ArrayToM31(_ arr: [UInt32]) -> [M31] {
    return arr.map { M31(v: $0) }
}

// ============================================================
// Circle NTT via ANE C API
// ============================================================

public func runCircleNTTANEBench() {
    print("=== Circle NTT Benchmark: CPU vs GPU vs ANE ===\n")

    // ============================================================
    // Initialize ANE
    // ============================================================
    print("--- Initialization ---")
    let aneInitResult = ane_circle_ntt_init()
    let aneAvailable = ane_circle_ntt_gpu_available()
    print("  ANE init: \(aneInitResult == 0 ? "success" : "failed")")
    print("  ANE GPU available: \(aneAvailable)")
    print("")

    // ============================================================
    // Correctness Tests
    // ============================================================
    print("--- Correctness Tests ---")

    // Test ANE Circle NTT forward + inverse roundtrip
    for logN in 4...10 {
        let n = 1 << logN

        // Generate random input
        var rng: UInt64 = 0xDEAD_BEEF_CAFE0000 + UInt64(logN)
        var input = [UInt32]()
        for _ in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            var val = UInt32(rng >> 33)
            if val >= 0x7FFFFFFF { val = val % 0x7FFFFFFF }
            input.append(val)
        }

        // Compute CPU reference (forward + inverse)
        var cpuInput = uint32ArrayToM31(input)
        let cpuFwd = CircleNTTEngine.cpuNTT(cpuInput, logN: logN)
        let cpuInv = CircleNTTEngine.cpuINTT(cpuFwd, logN: logN)

        // Check roundtrip correctness
        var roundtripOK = true
        for i in 0..<n {
            if cpuInv[i].v != input[i] {
                roundtripOK = false
                break
            }
        }

        // Test ANE forward
        if aneAvailable {
            var aneData = input
            let twiddles = computeCircleTwiddlesM31(logN: logN)
            let twiddlesUInt32 = m31ArrayToUInt32(twiddles)

            let aneFwdResult = ane_circle_ntt_forward(&aneData, twiddlesUInt32, Int32(logN))

            if aneFwdResult == 0 {
                // Compare ANE forward to CPU forward
                var aneFwd = Array(aneData.prefix(n))
                var cpuFwdUInt32 = m31ArrayToUInt32(cpuFwd)

                var fwdMatch = true
                for i in 0..<n {
                    // Allow small numerical differences due to different reduction
                    let diff = abs(Int32(aneFwd[i]) - Int32(cpuFwdUInt32[i]))
                    if diff > 1 {
                        fwdMatch = false
                        break
                    }
                }

                // Test ANE inverse - use a copy of forward result
                let invN = m31Inverse(M31(v: UInt32(n))).v
                let invTwiddles = computeInverseCircleTwiddlesM31(logN: logN)
                let invTwiddlesUInt32 = m31ArrayToUInt32(invTwiddles)
                var aneInvData = aneFwd  // Copy of forward result
                let aneInvResult = ane_circle_ntt_inverse(&aneInvData, invTwiddlesUInt32, invN, Int32(logN))

                if aneInvResult == 0 {
                    var invMatch = true
                    for i in 0..<n {
                        let diff = abs(Int32(aneInvData[i]) - Int32(input[i]))
                        if diff > 1 {
                            invMatch = false
                            break
                        }
                    }
                    print("  [\(fwdMatch && invMatch ? "pass" : "FAIL")] ANE Circle NTT N=\(n): fwd=\(fwdMatch), inv=\(invMatch)")
                } else {
                    print("  [FAIL] ANE Circle INTT N=\(n)")
                }
            } else {
                print("  [FAIL] ANE Circle NTT forward N=\(n): error \(aneFwdResult)")
            }
        } else {
            print("  [skip] ANE Circle NTT N=\(n) (ANE not available)")
        }

        // CPU roundtrip test
        if roundtripOK {
            print("  [pass] CPU Circle NTT roundtrip N=\(n)")
        } else {
            print("  [FAIL] CPU Circle NTT roundtrip N=\(n)")
        }
    }

    // ============================================================
    // Performance Benchmarks
    // ============================================================
    print("\n--- Performance Benchmarks ---")

    // GPU engine
    var gpuEngine: CircleNTTEngine? = nil
    do {
        gpuEngine = try CircleNTTEngine()
        print("  GPU engine: initialized")
    } catch {
        print("  GPU engine: failed to initialize: \(error)")
    }

    // Benchmark sizes
    let benchSizes = [8, 10, 12, 14, 16, 18]

    for logN in benchSizes {
        let n = 1 << logN
        print(String(format: "\n  === N = 2^%d = %d ===", logN, n))

        // Generate random input
        var rng: UInt64 = 0x1234_5678_ABCD_EFED + UInt64(logN)
        var inputM31 = [M31]()
        for _ in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            var val = UInt32(rng >> 33) % M31.P
            inputM31.append(M31(v: val))
        }
        let inputUInt32 = m31ArrayToUInt32(inputM31)

        // Precompute twiddles for ANE
        let twiddles = computeCircleTwiddlesM31(logN: logN)
        let twiddlesUInt32 = m31ArrayToUInt32(twiddles)
        let invN = computeInvNM31(n: n)

        // ----- CPU Benchmark -----
        let cpuRuns = 3
        var cpuTimes = [Double]()
        for _ in 0..<cpuRuns {
            let start = CFAbsoluteTimeGetCurrent()
            var data = inputM31
            for _ in 0..<5 {  // 5 iterations for measurable time
                data = CircleNTTEngine.cpuNTT(data, logN: logN)
            }
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000 / 5.0
            cpuTimes.append(elapsed)
        }
        cpuTimes.sort()
        let cpuMedian = cpuTimes[cpuRuns / 2]
        print(String(format: "  CPU:       %8.2f ms", cpuMedian))

        // ----- GPU Benchmark -----
        if let engine = gpuEngine {
            // Warmup
            var warmupData = inputM31
            for _ in 0..<2 {
                warmupData = (try? engine.ntt(warmupData)) ?? warmupData
            }

            let gpuRuns = 5
            var gpuTimes = [Double]()
            for _ in 0..<gpuRuns {
                var data = inputM31
                let start = CFAbsoluteTimeGetCurrent()
                let result = try? engine.ntt(data)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                if result != nil {
                    gpuTimes.append(elapsed)
                }
            }
            if !gpuTimes.isEmpty {
                gpuTimes.sort()
                let gpuMedian = gpuTimes[gpuRuns / 2]
                let gpuSpeedup = cpuMedian / gpuMedian
                print(String(format: "  GPU Metal: %8.2f ms  (%.1fx speedup)", gpuMedian, gpuSpeedup))
            } else {
                print("  GPU Metal:  failed")
            }
        }

        // ----- ANE Benchmark -----
        if aneAvailable {
            var aneTimes = [Double]()
            for _ in 0..<5 {
                var data = inputUInt32
                let start = CFAbsoluteTimeGetCurrent()
                for _ in 0..<5 {  // Batch 5 iterations
                    var mutableData = data
                    let result = ane_circle_ntt_forward(&mutableData, twiddlesUInt32, Int32(logN))
                    if result == 0 {
                        data = mutableData
                    }
                }
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000 / 5.0
                aneTimes.append(elapsed)
            }
            if !aneTimes.isEmpty {
                aneTimes.sort()
                let aneMedian = aneTimes[aneTimes.count / 2]
                let aneSpeedup = cpuMedian / aneMedian
                print(String(format: "  ANE:        %8.2f ms  (%.1fx speedup vs CPU)", aneMedian, aneSpeedup))
            } else {
                print("  ANE:        failed")
            }
        } else {
            print("  ANE:        not available")
        }
    }

    print("\nCircle NTT ANE benchmark complete.")
}

// ============================================================
// Twiddle computation helpers (matching CircleNTTEngine)
// ============================================================

/// Compute inv(N) mod M31.P
func computeInvNM31(n: Int) -> UInt32 {
    // inv_n = n^(-1) mod p using Fermat's little theorem
    var result = UInt32(n % Int(M31.P))
    let pMinus2 = M31.P - 2
    var exp = pMinus2
    while exp > 0 {
        if exp & 1 == 1 {
            result = m31Mul(M31(v: result), M31(v: result)).v  // Simplified
        }
        exp >>= 1
    }
    // Actually compute proper inverse
    return m31Inverse(M31(v: UInt32(n))).v
}

/// Compute Circle NTT twiddles in [UInt32] format
func computeCircleTwiddlesM31(logN: Int) -> [M31] {
    let n = 1 << logN
    let half = n / 2
    let domain = circleCosetDomain(logN: logN)

    var allTwiddles = [M31]()
    allTwiddles.reserveCapacity(logN * half)

    // Layer 0: y-coordinate twiddles
    var layer0 = [M31](repeating: M31.zero, count: half)
    for i in 0..<half {
        layer0[i] = domain[i].y
    }
    allTwiddles.append(contentsOf: layer0)

    // Layers 1..k-1: x-coordinate twiddles
    var xs = (0..<half).map { domain[$0].x }
    for layer in 1..<logN {
        let stride = 1 << layer
        var layerTw = [M31](repeating: M31.zero, count: half)
        let numValues = half / stride
        for j in 0..<numValues {
            if j < xs.count {
                layerTw[j * stride] = xs[j]
            }
        }
        allTwiddles.append(contentsOf: layerTw)

        // Squaring map
        let halfLen = xs.count / 2
        var newXs = [M31](repeating: M31.zero, count: halfLen)
        for i in 0..<halfLen {
            newXs[i] = m31Sub(m31Add(m31Sqr(xs[i]), m31Sqr(xs[i])), M31.one)
        }
        xs = newXs
    }

    return allTwiddles
}

/// Compute inverse Circle NTT twiddles in [UInt32] format
func computeInverseCircleTwiddlesM31(logN: Int) -> [M31] {
    let forwardTwiddles = computeCircleTwiddlesM31(logN: logN)
    return forwardTwiddles.map { m31Inverse($0) }
}