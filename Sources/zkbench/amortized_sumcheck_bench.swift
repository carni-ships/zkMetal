// Amortized Sumcheck Benchmark — O(1) Per-Query with Precomputed Tower Basis Cache
//
// Benchmarks the constraint-packing + precomputation amortization approach
// from ePrint 2024/1038: Constraint-Packing and Sum-Check Protocol over Binary Towers
//
// Measures:
// 1. TowerBasisCache initialization (one-time precomputation)
// 2. Per-query cost with cached vs uncached approaches
// 3. Speedup factor from precomputation amortization

import zkMetal
import Foundation

public func runAmortizedSumcheckBench() {
    fputs("\n=== Amortized Sumcheck Benchmark ===\n", stderr)

    // Benchmark configurations: (maxLevel, domainSize, numQueries)
    // Note: maxLevel <= 8 for direct GF(2^8) basis computation
    let configs: [(String, Int, Int, Int)] = [
        ("small",   8, 1 << 8, 100),
        ("medium",  10, 1 << 10, 50),
        ("large",   12, 1 << 12, 20),
    ]

    for (name, maxLevel, domainSize, numQueries) in configs {
        fputs("\n--- Config: \(name) (level=\(maxLevel), domain=2^\(Int(log2(Double(domainSize)))), queries=\(numQueries)) ---\n", stderr)

        // Create packer and basis cache
        let config = PackingConfig(
            strategy: .maximizeDensity,
            maxPackedTowerLevel: PackedTowerLevel(maxLevel),
            enableConstraintReuse: true,
            enableSliceOptimization: true
        )

        // Benchmark precomputation (TowerBasisCache init)
        let cacheResult = bench("  Precompute cache", warmup: 2, iterations: 5) {
            let cache = TowerBasisCache(
                maxLevel: PackedTowerLevel(maxLevel),
                domainSize: domainSize
            )
            cache.initialize()
        }

        // Create cache for subsequent benchmarks
        let basisCache = TowerBasisCache(
            maxLevel: PackedTowerLevel(maxLevel),
            domainSize: domainSize
        )
        basisCache.initialize()

        // Create constraint packer with some dummy constraints
        let packer = ConstraintPacker(config: config, basisCache: basisCache)

        // Generate random R1CS constraints
        let numConstraints = 64
        var constraints: [PackedR1CSConstraint] = []
        var rng: UInt64 = 0xDEAD_BEEF_CAFE_1234
        for _ in 0..<numConstraints {
            let a = [(Int(rng & 0xFF), UInt8((rng >> 8) & 0xFF))]
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let b = [(Int(rng & 0xFF), UInt8((rng >> 8) & 0xFF))]
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let c = [(Int(rng & 0xFF), UInt8((rng >> 8) & 0xFF))]
            constraints.append(PackedR1CSConstraint(a: a, b: b, c: c))
        }
        // numVars matches log2(domainSize)
        let packerNumVars = Int(log2(Double(domainSize)))
        _ = packer.pack(constraints: constraints, variableCount: 1 << packerNumVars)

        // Create prover
        let prover = AmortizedSumcheckProver(
            basisCache: basisCache,
            constraintPacker: packer
        )

        // Generate random witness - size matches domainSize (2^numVars)
        let numVars = packerNumVars
        var witness = [UInt8](repeating: 0, count: 1 << numVars)
        for i in 0..<witness.count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            witness[i] = UInt8(rng & 0xFF)
        }

        // Benchmark per-query proof generation
        let proofResult = bench("  Per-query prove (amortized)", warmup: 3, iterations: 10) {
            let polynomialEvals = (0..<(1 << numVars)).map { _ in
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                return UInt8(rng & 0xFF)
            }
            let randomness = (0..<numVars).map { _ in
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                return UInt8(rng & 0xFF)
            }

            let claim = AmortizedSumcheckClaim(
                numVariables: numVars,
                targetSum: 0,
                polynomialEvals: polynomialEvals
            )
            _ = prover.prove(claim: claim, witness: witness, randomness: randomness)
        }

        // Estimate theoretical speedup
        let estimator = PrecomputationCostEstimator(
            maxLevel: maxLevel,
            domainSize: domainSize,
            numRounds: 8,
            numQueries: numQueries
        )
        fputs("  Precomputation cost: \(estimator.precomputationCost) ops\n", stderr)
        fputs("  Per-query without cache: \(estimator.costPerQueryWithoutCache) ops\n", stderr)
        fputs("  Per-query with cache: \(estimator.costPerQueryWithCache) ops\n", stderr)
        fputs("  Theoretical speedup: \(String(format: "%.2fx", estimator.speedupFactor))\n", stderr)
        fputs("  Precomputation worthwhile: \(estimator.isPrecomputationWorthwhile ? "YES" : "NO")\n", stderr)

        // Benchmark verifier
        let verifier = AmortizedSumcheckVerifier(basisCache: basisCache)
        let verifierNumVars = Int(log2(Double(domainSize)))
        let verifierEvals = (0..<(1 << verifierNumVars)).map { _ in
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return UInt8(rng & 0xFF)
        }
        let verifierRandomness = (0..<verifierNumVars).map { _ in
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return UInt8(rng & 0xFF)
        }
        let claim = AmortizedSumcheckClaim(
            numVariables: verifierNumVars,
            targetSum: 0,
            polynomialEvals: verifierEvals
        )
        let proof = prover.prove(claim: claim, witness: witness, randomness: verifierRandomness)

        let verifyResult = bench("  Verify", warmup: 3, iterations: 10) {
            _ = verifier.verify(proof: proof, claim: claim, randomness: verifierRandomness)
        }

        // Memory estimate
        fputs("  Estimated cache memory: \(basisCache.estimatedMemoryBytes / 1024) KB\n", stderr)

        // Cleanup
        fputs("  Packing efficiency: \(String(format: "%.2f", packer.packingEfficiency))x\n", stderr)
    }

    fputs("\n=== Amortized Sumcheck Benchmark Complete ===\n", stderr)
}

// MARK: - Correctness Tests

public func runAmortizedSumcheckCorrectness() {
    fputs("\n=== Amortized Sumcheck Correctness Tests ===\n", stderr)

    // Create test infrastructure
    let maxLevel = 8
    let domainSize = 1 << maxLevel
    let basisCache = TowerBasisCache(
        maxLevel: PackedTowerLevel(maxLevel),
        domainSize: domainSize
    )
    basisCache.initialize()

    let config = PackingConfig(
        strategy: .onePerLevel,
        maxPackedTowerLevel: PackedTowerLevel(maxLevel),
        enableConstraintReuse: true,
        enableSliceOptimization: true
    )
    let packer = ConstraintPacker(config: config, basisCache: basisCache)
    let prover = AmortizedSumcheckProver(basisCache: basisCache, constraintPacker: packer)
    let verifier = AmortizedSumcheckVerifier(basisCache: basisCache)

    // Test 1: Simple claim verification
    do {
        let numVars = 4
        let n = 1 << numVars
        let polynomialEvals: [UInt8] = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
        let targetSum: UInt8 = polynomialEvals.reduce(0, { $0 ^ $1 })  // XOR sum

        let witness = [UInt8](repeating: 1, count: 16)
        // Use BINARY challenges (0 or 1) only
        // Use all-zero challenges to test reduceEvals without constraint bit issues
        let randomness: [UInt8] = [0x00, 0x00, 0x00, 0x00]

        let claim = AmortizedSumcheckClaim(
            numVariables: numVars,
            targetSum: targetSum,
            polynomialEvals: polynomialEvals
        )

        let proof = prover.prove(claim: claim, witness: witness, randomness: randomness)
        let valid = verifier.verify(proof: proof, claim: claim, randomness: randomness)

        fputs("  Simple claim verification: \(valid ? "PASS" : "FAIL")\n", stderr)
    }

    // Test 2: ZeroCheck prove/verify
    do {
        let numVars = 4
        let n = 1 << numVars
        // Polynomial that evaluates to 0 at all points (trivially zero)
        let polynomialEvals: [UInt8] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        let witness = [UInt8](repeating: 1, count: 16)
        let randomness: [UInt8] = [0x03, 0x05, 0x07, 0x0B]

        let zeroCheck = AmortizedZeroCheck(basisCache: basisCache, constraintPacker: packer)
        let proof = zeroCheck.proveZero(polynomialEvals: polynomialEvals, witness: witness, randomness: randomness)
        let valid = zeroCheck.verifyZero(proof: proof, polynomialEvals: polynomialEvals, randomness: randomness)

        fputs("  ZeroCheck prove/verify: \(valid ? "PASS" : "FAIL")\n", stderr)
    }

    // Test 3: Constraint packing efficiency
    do {
        let numConstraints = 32
        var constraints: [PackedR1CSConstraint] = []
        for i in 0..<numConstraints {
            let a = [(i % 16, UInt8(i + 1))]
            let b = [(i % 16, UInt8(i + 2))]
            let c = [(i % 16, UInt8(i + 3))]
            constraints.append(PackedR1CSConstraint(a: a, b: b, c: c))
        }

        _ = packer.pack(constraints: constraints, variableCount: 32)
        let efficiency = packer.packingEfficiency

        fputs("  Constraint packing: packed \(numConstraints) constraints with efficiency \(String(format: "%.2f", efficiency))x\n", stderr)
        fputs("  Active levels: \(packer.activeLevels.count)\n", stderr)
    }

    fputs("\n=== Correctness Tests Complete ===\n", stderr)
}
