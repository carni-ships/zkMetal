import zkMetal
import Foundation

public func runFusedDeepFoldBench() {
    fputs("\n=== FusedDeepFold Benchmark ===\n", stderr)

    // Note: by4 kernel only does 3 rounds, by8 does 7 rounds
    // Using 3 rounds to match actual kernel capability
    let gpuRounds = 3  // actual rounds the GPU kernel processes
    let cpuRounds = 3  // match GPU for correctness comparison
    let m = 1024  // vector size

    // Generate test data
    var rng: UInt64 = 0xDEAD_BEEF
    func randomFr() -> Fr {
        rng = rng &* 6364136223846793005 &+ 1
        return frFromInt(rng)
    }

    let az0 = (0..<m).map { _ in randomFr() }
    let bz0 = (0..<m).map { _ in randomFr() }
    let cz0 = (0..<m).map { _ in randomFr() }

    var instances: [(az: [Fr], bz: [Fr], cz: [Fr])] = []
    for _ in 0..<cpuRounds {
        instances.append((
            (0..<m).map { _ in randomFr() },
            (0..<m).map { _ in randomFr() },
            (0..<m).map { _ in randomFr() }
        ))
    }

    let challenges = (0..<cpuRounds).map { _ in randomFr() }

    // Benchmark CPU reference
    let engine = try! FusedDeepFoldEngine(fusedRounds: 4)  // by4 kernel

    // Warmup
    _ = try! engine.cpuFusedFold(
        az0: az0, bz0: bz0, cz0: cz0,
        instances: instances, challenges: challenges
    )

    // Benchmark CPU
    let runs = 10
    var cpuTimes = [Double]()
    for _ in 0..<runs {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try! engine.cpuFusedFold(
            az0: az0, bz0: bz0, cz0: cz0,
            instances: instances, challenges: challenges
        )
        cpuTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
    }
    cpuTimes.sort()
    fputs("CPU fused (\(cpuRounds) rounds, m=\(m)): \(String(format: "%.2f", cpuTimes[runs/2])) ms\n", stderr)

    // Benchmark GPU
    fputs("\n--- GPU Benchmark ---\n", stderr)
    do {
        let gpuResult = try engine.fusedFold(
            az0: az0, bz0: bz0, cz0: cz0,
            instances: instances, challenges: challenges
        )
        fputs("GPU fused (\(gpuRounds) rounds, m=\(m)): completed\n", stderr)
        fputs("GPU result[0] = \(frToInt(gpuResult.t[0]))\n", stderr)
    } catch {
        fputs("GPU error: \(error)\n", stderr)
    }

    // Correctness check
    fputs("\n--- Correctness Check ---\n", stderr)
    let cpuResult = try! engine.cpuFusedFold(
        az0: az0, bz0: bz0, cz0: cz0,
        instances: instances, challenges: challenges
    )

    do {
        let gpuResult = try engine.fusedFold(
            az0: az0, bz0: bz0, cz0: cz0,
            instances: instances, challenges: challenges
        )

        // Compare first few elements
        var match = true
        let compareCount = min(10, m)
        for i in 0..<compareCount {
            if frToInt(cpuResult.t[i]) != frToInt(gpuResult.t[i]) {
                fputs("Mismatch at index \(i): CPU=\(frToInt(cpuResult.t[i])), GPU=\(frToInt(gpuResult.t[i]))\n", stderr)
                match = false
            }
        }
        fputs("Correctness: \(match ? "PASS" : "FAIL")\n", stderr)
    } catch {
        fputs("GPU correctness check failed: \(error)\n", stderr)
    }

    // Test different sizes
    fputs("\n--- Size Scaling ---\n", stderr)
    let sizes = [256, 1024, 4096]
    for size in sizes {
        let testAz0 = (0..<size).map { _ in randomFr() }
        let testBz0 = (0..<size).map { _ in randomFr() }
        let testCz0 = (0..<size).map { _ in randomFr() }

        var testInstances: [(az: [Fr], bz: [Fr], cz: [Fr])] = []
        for _ in 0..<3 {
            testInstances.append((
                (0..<size).map { _ in randomFr() },
                (0..<size).map { _ in randomFr() },
                (0..<size).map { _ in randomFr() }
            ))
        }
        let testChallenges = (0..<3).map { _ in randomFr() }

        let cpuStart = CFAbsoluteTimeGetCurrent()
        _ = try! engine.cpuFusedFold(
            az0: testAz0, bz0: testBz0, cz0: testCz0,
            instances: testInstances, challenges: testChallenges
        )
        let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuStart) * 1000
        fputs("CPU m=\(size): \(String(format: "%.2f", cpuTime)) ms\n", stderr)
    }
}
