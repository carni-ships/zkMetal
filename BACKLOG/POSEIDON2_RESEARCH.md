# Poseidon2 Optimization Research

**Date**: 2026-04-15
**Status**: Researching
**Parent Issue**: Poseidon2 Merkle profiling shows GPU compute is 100% of time (not memory transfer)

---

## Profiling Data (M3 Pro, 2^20 leaves)

| Tree Type | Time | Notes |
|-----------|------|-------|
| Binary Poseidon2 | 3610ms | 20 levels, 10 fused + 10 level-by-level |
| 4-ary Poseidon2 | 543ms | 10 levels, 2.7x faster than binary |
| Target (pure 4-ary) | ~270ms | 10 levels, 4^10 = 2^20 |

---

## Conjectures for Poseidon2 Speedup

### Algorithmic/Structural

#### 1. Reduced Round Count
- **Idea**: Current uses 64 rounds (8 + 56 + 8). Reduce to 56 (8 + 48 + 8) or 52 (8 + 44 + 8).
- **Impact**: ~12-19% speedup if security holds
- **Risk**: Must verify security margin carefully
- **Status**: Unverified

#### 2. Uniform Round Schedule
- **Idea**: Partial rounds (only s0 gets S-box) create branch divergence. Uniform schedule could improve GPU ILP.
- **Impact**: Unknown, depends on GPU architecture
- **Risk**: May not improve throughput if S-box is bottleneck
- **Status**: Unverified

#### 3. S-box Decomposition Alternatives
- **Idea**: x^5 = x * x^4 = x * (x^2)^2. Different multiplication trees:
  - Current: x2=x*x, x4=x2*x2, x5=x4*x (2 muls, chain dependency)
  - Alternative: x3=x*x2, x5=x2*x3 (2 muls, different scheduling)
  - x^7 = x^3 * x^4 (3 muls) - might have better instruction scheduling
- **Impact**: Minor (instruction scheduling differences)
- **Status**: Low priority

#### 4. Lazy Reduction for Linear Layer
- **Idea**: External linear layer outputs in [0, 2p). Stay lazy for 2-3 rounds before reducing.
- **Impact**: Could reduce reduce operations by 2-3x
- **Risk**: Must ensure no overflow beyond 2^256
- **Status**: Unverified

### GPU-Specific

#### 5. Batch Permutation Processing
- **Idea**: Instead of one permutation per threadgroup, batch N permutations to amortize RC loading.
- **Impact**: Reduces dispatch overhead, increases parallelism
- **Status**: Implemented (poseidon2_permute kernel processes N permutations)

#### 6. RC Prefetch into Registers
- **Idea**: Preload all 192 RC values into registers at kernel start, eliminate constant memory loads per round.
- **Impact**: Reduces constant memory bandwidth
- **Risk**: Register pressure may increase
- **Status**: Unverified

#### 7. Register-Tiled Linear Layer
- **Idea**: Instead of `sum=a+b+c; a+=sum; b+=sum; c+=sum` (sequential), use tree reduction: `t1=a+b; t2=c+sum; result=t1+t2`
- **Impact**: Fewer dependent operations, better ILP
- **Status**: Unverified

#### 8. Alternative S-box (x^7 or x^11)
- **Idea**: Use x^7 = x^3 * x^4 (3 muls) instead of x^5 (2 muls)
- **Impact**: Unknown - more multiplications but different scheduling
- **Status**: Unverified - requires specification change

### Memory/Access Patterns

#### 9. RC Constant Memory Banking
- **Idea**: On M3, constant memory has banking conflicts for sequential accesses. RC array layout could use stride > 1.
- **Impact**: Unknown
- **Status**: Low priority

#### 10. Shared Memory for Working State
- **Idea**: Use shared memory for t=3 state instead of registers. Enables threadblock data sharing.
- **Impact**: Unknown
- **Risk**: Shared memory is slower than registers for this working set
- **Status**: Unverified

### Architectural

#### 11. Mixed-Arity Tree Optimization
- **Idea**: Use 4-ary at base (most nodes), switch to 2-ary at top (few nodes, less impact)
- **Status**: Implemented in Poseidon24aryMerkleEngine

#### 12. Async/Pipelined Execution
- **Idea**: Use multiple command buffers. CPU prepares level N+1 while GPU executes level N.
- **Impact**: Overlaps CPU and GPU time (currently CPU idle during 1500ms GPU execution)
- **Status**: Not implemented
- **Notes**: Metal supports multiple CBs in flight

#### 13. Multi-Pair-Per-Thread for Low Levels
- **Idea**: For upper levels with few pairs, process multiple pairs per thread to improve utilization
- **Status**: Unverified

---

## Priority Ranking (Impact × Confidence / Effort)

| Rank | Idea | Impact | Confidence | Effort | Score |
|------|------|--------|------------|--------|-------|
| 1 | Async pipelining | High | High | Low | 9 |
| 2 | Reduced round count | High | Medium | High | 5 |
| 3 | RC prefetch | Medium | High | Medium | 6 |
| 4 | Lazy reduction | High | Medium | High | 5 |
| 5 | Register-tiled linear layer | Medium | Medium | Medium | 4 |
| 6 | Multi-pair-per-thread | Medium | Medium | Medium | 4 |

---

## Verification Checklist

Before keeping any change:
- [ ] Ordering preserved
- [ ] Numerical outputs unchanged (exact bit-level match)
- [ ] Security properties maintained
- [ ] Works on M3 Pro GPU
- [ ] Benchmarks show improvement (not just theoretical)

---

## Related Files

- `Sources/Shaders/hash/poseidon2.metal` - Main GPU kernel
- `Sources/zkMetal/Hash/Poseidon2Engine.swift` - Swift wrapper
- `Sources/zkMetal/Hash/MerkleEngine.swift` - Merkle tree implementation
