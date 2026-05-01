# Caching Optimizations Backlog

## Priority: HIGH

### 1. FRI Engine - Omega/Root-of-Unity Computations
- **File**: `Sources/zkMetal/Polynomial/FRIEngine.swift`
- **Lines**: 719-724, 1123-1126, 1463-1466, 1488-1490, 1741-1747, 2196-2201, 2219-2221, 2270-2272
- **Pattern**: `frRootOfUnity(logN)`, `frInverse(omega)`, `frPow(omegaInv, UInt64(quarter))` computed repeatedly
- **Status**: ✅ Done (2026-05-01) - Added `getW4Inv()`, `getW8Inv()`, `getW8Inv3()` cached helpers
- **Impact**: Reduces O(n) repeated root computations per fold round

### 2. Reed-Solomon Engine - Lagrange Interpolation Denominators
- **File**: `Sources/zkMetal/Polynomial/RSEngine.swift`
- **Lines**: 108-121, 145-150, 217-236
- **Pattern**:
  - `omega^i` powers recomputed for each evaluation
  - Lagrange denominators `prod_{j!=i}(x_i - x_j)` computed O(n^2) without caching
- **Status**: ✅ Done (2026-05-01) - Added `getTwiddleCache()` and `frPowOmega()` in BN254Fr.swift
- **Impact**: High for repeated RS encoding/decoding operations

### 3. WHIR Proximity Engine - Domain Weight Recomputation
- **File**: `Sources/zkMetal/Polynomial/WHIRProximityEngine.swift`
- **Lines**: 327-336, 349-367
- **Pattern**: `frPow(omega, UInt64(index))` called in loop for each evaluation point
- **Status**: ✅ Done (2026-05-01) - Uses `frPowOmega()` with cached twiddles + batch inverse
- **Impact**: O(n) to O(1) for domain weight computation

### 4. STARK Provers - Repeated Omega Powers in OOD Evaluation
- **Files**:
  - `Sources/zkMetal/STARK/Stark252STARK.swift` (lines 718-721)
  - `Sources/zkMetal/STARK/GPUBabyBearSTARKProver.swift` (lines 446-451)
- **Pattern**: Omega powers `omega^i` recomputed for each proof
- **Status**: ❌ Low impact (2026-05-01) - Only called a few times per verification, not worth special caching
- **Impact**: Minimal - verification happens rarely compared to proving

### 5. Subproduct Tree - Polynomial Tree Building
- **File**: `Sources/zkMetal/Polynomial/SubproductTree.swift`
- **Lines**: 35-36, 74, 564-659
- **Pattern**: O(n log^2 n) tree rebuilt for same evaluation points
- **Fix**: Cache built tree for repeated evaluations at same points (depends only on points, not coefficients)
- **Impact**: Significant for multi-round protocols with same domain

---

## Priority: MEDIUM

### 6. GPUVanishingPolyEngine - Coset Point Precomputation
- **File**: `Sources/zkMetal/Polynomial/GPUVanishingPolyEngine.swift`
- **Lines**: 354-367, 372-385
- **Pattern**: Coset points `g * omega^i` and vanishing polynomial evaluation recomputed
- **Fix**: Cache coset points for `(logDomain, cosetGen)` combinations
- **Impact**: Called for small domain sizes in CPU path

### 7. Coset Domain Engine - Generator and Omega Powers
- **Pattern**: `frRootOfUnity(logN)` and `frInverse(omega)` computed throughout STARK provers/verifiers
- **Fix**: Global cache keyed by `(field, logN)` for all root-of-unity computations
- **Impact**: Would benefit all STARK operations

### 8. GPU PolyDiv Engine - Coset Points Without Caching
- **File**: `Sources/zkMetal/Polynomial/GPUPolyDivEngine.swift`
- **Lines**: 254-273
- **Pattern**: `frRootOfUnity(logN)` called, coset points `g * omega^i` recomputed
- **Fix**: Cache coset points per `(logDomain, cosetGen)`
- **Impact**: Benefits polynomial division operations

### 9. Challenge Powers in Commitment Batch Engine
- **File**: `Sources/zkMetal/Groth16/GPUGroth16AggregateEngine.swift`
- **Lines**: 260-265
- **Pattern**: Powers of challenge `r^0, r^1, ..., r^{n-1}` recomputed each aggregation
- **Fix**: Cache challenge powers for repeated challenges
- **Impact**: When same challenge used multiple times

### 10. WHIR Prover - Merkle Tree Rebuilding
- **File**: `Sources/zkMetal/Polynomial/WHIRIOPProver.swift`
- **Lines**: 244, 310, 379-399
- **Pattern**: Merkle tree built from same leaves across queries
- **Fix**: Cache intermediate tree levels
- **Impact**: Once per round in WHIR protocol

---

## Priority: LOW (Already Cached - Verify Hits)

### 11. Circle FRI Engine - Inv2y Cache
- **File**: `Sources/zkMetal/Polynomial/CircleFRIEngine.swift`
- **Lines**: 114-131, 137-150
- **Status**: Has `inv2yCache` and `inv2xCache` - verify cache is being hit

### 12. GPU FRI Engine - Inverse Twiddle Cache
- **File**: `Sources/zkMetal/Polynomial/GPUFRIEngine.swift`
- **Lines**: 200-238
- **Status**: Has `invTwiddleCache` - verify cache is being hit

### 13. GPU Quotient Engine - Vanishing Inverse Cache
- **File**: `Sources/zkMetal/Polynomial/GPUQuotientEngine.swift`
- **Lines**: 269-341
- **Status**: Has cache keyed by `(logDomain, logTraceLen, cosetGen, field)` - good pattern!

### 14. Tower Basis Cache - Twiddle Tables
- **File**: `Sources/zkMetal/Sumcheck/ConstraintPacking/TowerBasisCache.swift`
- **Lines**: 266-281
- **Status**: Already caches in `twiddleTables` array - good!

### 15. Precomputed Poly Manager - Lagrange Numerators
- **File**: `Sources/zkMetal/Sumcheck/ConstraintPacking/PrecomputedPolyManager.swift`
- **Lines**: 322-336
- **Status**: Already cached in `lagrangeNumerators` - verify invalidation on challenge change

### 16. FRI Engine - Layer Buffer Cache
- **File**: `Sources/zkMetal/Polynomial/FRIEngine.swift`
- **Lines**: 897-903
- **Status**: Has caching but cache key comparison could be improved

---

## Pre-Existing Build Errors

These files have bugs that must be fixed before caching optimizations:

### SparsePolyCommit.swift
- **Line**: 390
- **Error**: `cannot find 'srsPrefix' in scope`
- **Issue**: Function reference missing

### P1NTTEngine.swift
- **Lines**: 824, 865
- **Error**: `cannot find 'Self' in scope`
- **Issue**: `Self.p1CosetDomainCache` used outside type context

---

## Implementation Notes

### Cache Key Strategies

1. **Root of Unity Cache**: `[(fieldType, logN)] -> Fr` - stores `omega` and `omegaInv`
2. **Omega Powers Cache**: `[(fieldType, logN, count)] -> [Fr]` - stores `omega^i` powers
3. **Lagrange Cache**: `[(fieldType, points.hash)] -> (denominators, interpolation coeffs)`
4. **Merkle Tree Cache**: `[(leaves.hash)] -> (root, intermediate_levels)`

### Thread Safety

- Use `NSLock` or dispatch queues for concurrent cache access
- Consider `DispatchQueue.concurrent` for read-heavy workloads
- Static caches should use lazy initialization

### Memory Management

- Set maximum cache sizes to prevent unbounded growth
- Use LRU eviction for large data structures (coset domains, trees)
- Clear caches when field configuration changes

---

## P1 Rational Function STARKs Optimizations (Apr 21, 2026)

### Priority: HIGH

#### 1. Missing fold-by-8 kernel
- **File**: `Sources/zkMetal/Polynomial/P1FRIEngine.swift`
- **Lines**: 24-26 (only has foldBy2, foldBy4 - no foldBy8)
- **Pattern**: Main FRI engine has `fold-by-8` kernel, P1 FRI only has 2 and 4
- **Impact**: 3-7x speedup for large domains (2^20: 3 dispatches vs ~5 with fold-by-4)
- **Fix**: Implement `p1_fri_fold_by8` kernel in Metal, add `foldBy8Function` pipeline state

#### 2. inv2t buffer allocations in multiFold
- **File**: `Sources/zkMetal/Polynomial/P1FRIEngine.swift`
- **Lines**: 370-371
- **Pattern**: `multiFold` creates new GPU buffer for each round:
  ```swift
  for i in 0..<alphas.count {
      let inv2tData = getInv2tFolded(logN: logN, foldRound: i)
      let inv2tBuf = createM31Buffer(inv2tData)!  // NEW buffer each iteration!
  }
  ```
- **Impact**: 19 GPU buffer allocations per multiFold call at 2^20
- **Fix**: Use existing `getAllInv2t()` + `inv2tBufCache` pattern from `commitPhase`

#### 3. inv2t buffer allocations in commitPhaseFused
- **File**: `Sources/zkMetal/Polynomial/P1FRIEngine.swift`
- **Lines**: 574-584
- **Pattern**: Creates 4 new inv2t buffers per fold-by-4 group
- **Impact**: Repeated allocations in fused path
- **Fix**: Precompute all inv2t arrays and cache GPU buffers before while loop

### Priority: MEDIUM

#### 4. Query phase rebuilds Merkle tree 76x
- **File**: `Sources/zkMetal/Polynomial/P1FRIEngine.swift`
- **Lines**: 739-759
- **Pattern**: For each query index, builds full tree at each layer:
  ```swift
  for qi in 0..<queryIndices.count {
      for layer in 0..<(commitment.layers.count - 1) {
          let path = p1M31MerklePath(evals, index: Int(idx))  // Full tree build
      }
  }
  ```
- **Impact**: 4 queries × 19 layers = 76 full tree builds
- **Fix**: Build trees once upfront, extract paths

#### 5. O(n) Merkle path extraction
- **File**: `Sources/zkMetal/Polynomial/P1FRIEngine.swift`
- **Lines**: 936-956
- **Pattern**: `p1M31MerklePath` builds full binary tree then extracts one path
- **Impact**: O(n) per path extraction, could be O(log n)
- **Fix**: Direct path extraction without building full tree:
  ```swift
  private func p1M31MerklePath(_ leaves: [M31], index: Int) -> [M31] {
      var path = [M31]()
      var idx = n + index
      while idx > 1 {
          path.append(m31Hash(leaves[idx ^ 1 - n]))
          idx >>= 1
      }
      return path
  }
  ```

#### 6. CPU Merkle instead of GPU
- **File**: `Sources/zkMetal/Polynomial/P1FRIEngine.swift`
- **Lines**: 917-956
- **Pattern**: Uses simple CPU-based `p1M31MerkleRoot` with placeholder hash
- **Impact**: Merkle is not the bottleneck (0.4ms vs 183ms for fold), but GPU would be cleaner
- **Fix**: Use GPU Merkle engine if Poseidon2 for M31 exists

### Priority: LOW (Already Optimized)

#### 7. inv2t caching in commitPhase
- **File**: `Sources/zkMetal/Polynomial/P1FRIEngine.swift`
- **Status**: ✅ Already implemented - 65x speedup achieved
