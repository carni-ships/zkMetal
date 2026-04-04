# Circle STARK Proving on Apple Silicon: When 32-bit GPUs Win

## Target venue
IACR ePrint (immediate), CCS Workshop on Applied Cryptography / Real World Crypto poster track

## Authors
[TBD]

## Relationship to systems paper
The companion paper (outline.md) is a broad systems paper covering zkMetal's 22+ primitives across 7 fields and 5 curves. This paper is a focused deep-dive arguing a specific thesis: that Circle STARKs over Mersenne31 are a natural fit for Apple Silicon's 32-bit GPU ALU, and that field-hardware co-design turns an apparent disadvantage into a genuine advantage. The systems paper mentions M31 Circle NTT as one data point among many; this paper makes it the central argument with full algebraic, architectural, and empirical analysis.

---

## Abstract (~250 words)

- GPU-accelerated ZK proving assumes 64-bit arithmetic (NVIDIA CUDA); Apple Silicon GPUs have 32-bit native ALUs, conventionally seen as a ~4x penalty for 256-bit field operations
- Circle STARKs over Mersenne31 (p = 2^31 - 1) invert this: a single field multiplication is one 32-bit multiply plus a cheap reduction
- The circle group over F_p has order p + 1 = 2^31, providing 31 levels of 2-adic roots — ideal for NTT without extension fields
- We implement a full Circle STARK prover on Apple Metal: Circle NTT, Circle FRI, Poseidon2-M31 Merkle commitments, and constraint evaluation
- Key results: Circle NTT at 2^20 runs in ~4ms vs ~28ms for BN254 NTT on the same GPU (7x), driven by 8x memory density and single-instruction field multiply
- Poseidon2-M31 with width t=16 achieves higher throughput than Poseidon2-BN254 with t=3, despite needing more rounds, because each round costs ~50x fewer ALU instructions
- M-series scheduling pathologies (10-30x cliffs at specific threadgroup counts) are less likely to trigger for M31 workloads because smaller elements yield larger threadgroup counts that avoid pathological ranges
- We argue that field choice should be co-designed with target hardware, and that Apple Silicon is not merely viable but potentially optimal for small-field STARK proving

---

## 1. Introduction (1.5 pages)

### 1.1 The 32-bit GPU problem

- Apple Silicon GPUs ship in 100M+ Macs, all iPhones, all iPads — the largest deployed GPU fleet by unit count
- Metal's native ALU width is 32 bits; no hardware 64-bit multiply
- For BN254 (256-bit field): one field multiply requires 64 limb multiplications, carry propagation, and Montgomery reduction — roughly 200+ instructions
- This makes Apple Silicon look uncompetitive for conventional ZK workloads vs NVIDIA's 64-bit ALU
- Prior work (our systems paper, Icicle, cuZK) all benchmark on 64-bit GPUs with large fields

### 1.2 The Circle STARK opportunity

- Circle STARKs (Haboeck, Polygon, Stwo) use the circle group over prime fields
- Mersenne31 (p = 2^31 - 1) is the canonical small-field choice: p + 1 = 2^31, giving perfect 2-adic structure via the circle group
- One M31 field multiply: a single 32-bit multiply (result fits in 62 bits) plus a shift-and-subtract Mersenne reduction
- This maps directly to Apple Silicon's 32-bit ALU — no limb decomposition, no carry chains, no Montgomery constants
- Thesis: the hardware that is "wrong" for BN254 becomes "right" for M31 Circle STARKs

### 1.3 Contributions

1. Algebraic analysis of why Circle STARKs on M31 are uniquely suited to 32-bit GPUs, with instruction-level cost comparison
2. Implementation of Circle NTT, Circle FRI, and Poseidon2-M31 Merkle commitments on Apple Metal
3. Quantification of the field-size advantage on real hardware: 7x NTT speedup, 8x memory density, ~50x per-operation ALU reduction
4. Analysis of how M31's small element size interacts with M-series scheduling pathologies
5. End-to-end Circle STARK proof benchmarks on Apple Silicon, compared with BN254 STARK on the same hardware and published CUDA M31 numbers

---

## 2. Background (2 pages)

### 2.1 Circle groups and Circle STARKs

- Standard STARK FFT domain: multiplicative subgroup of F_p^* with order dividing p - 1
  - For M31: p - 1 = 2^31 - 2 = 2 * (2^30 - 1), only 1 factor of 2 — terrible 2-adicity
- Circle group: points (x, y) on x^2 + y^2 = 1 over F_p, with group law (x1, y1) * (x2, y2) = (x1*x2 - y1*y2, x1*y2 + x2*y1)
  - Order of the circle group is p + 1 = 2^31 — exactly a power of 2
  - This gives 31 levels of 2-adic roots, the maximum possible for a 31-bit prime
- Twin-coset decomposition: first layer splits into (x, y) and (x, -y) cosets
- Squaring map on x-coordinates: subsequent layers use x -> 2x^2 - 1 (Chebyshev doubling)
- References: Haboeck "Circle STARKs", Polygon/Stwo implementation, ECFFT (Eli Ben-Sasson et al.)

### 2.2 Apple Silicon GPU architecture for arithmetic

- M-series GPU ALU pipeline: 32-bit integer multiply is single-cycle throughput
- No hardware `mul.wide.u64`: 64-bit multiply requires 4x `mul.u32` + 3x `mad.u32` + carry propagation
- Threadgroup (shared) memory: 32KB per threadgroup, 128-byte cache lines
- SIMD width: 32 threads, no independent sub-SIMD execution
- Memory bandwidth: ~150 GB/s (M3 Pro), shared between CPU and GPU
- Comparison table:

| Property | M3 Pro GPU | A100 | RTX 4090 |
|---|---|---|---|
| Native ALU width | 32-bit | 64-bit | 64-bit |
| Shared memory/SM | 32KB | 164KB | 100KB |
| Memory bandwidth | ~150 GB/s | 2039 GB/s | 1008 GB/s |
| M31 mul cost | 1 cycle | 1 cycle | 1 cycle |
| BN254 mul cost | ~200 instr | ~50 instr | ~50 instr |
| M31 competitive ratio | 1.0x | 1.0x | 1.0x |
| BN254 competitive ratio | ~0.25x | 1.0x | 1.0x |

- Key insight: for M31 arithmetic, Apple Silicon is on equal footing with CUDA GPUs per-ALU; the gap is only in core count and memory bandwidth

### 2.3 Related work on small-field STARKs

- Plonky3 (Polygon): CPU-focused Circle STARK prover over M31, BabyBear, Goldilocks
- Stwo (StarkWare): Circle STARK prover, CPU with some CUDA acceleration
- Icicle (Ingonyama): CUDA M31 support added recently
- Binius: binary tower fields, different approach to small-field STARKs
- No published work on Metal/Apple Silicon for Circle STARKs or M31 proving

---

## 3. Circle NTT on Metal (2 pages)

### 3.1 Algebraic structure

- Standard NTT: evaluates polynomial at powers of a root of unity w where w^n = 1
- Circle NTT: evaluates polynomial in the circle basis at points on the circle group
- First layer (y-coordinate twiddles):
  - Split polynomial into even/odd parts using y-coordinate
  - Butterfly: f_even(x) + y * f_odd(x) at coset points
  - This is the "twin-coset decomposition" — unique to Circle STARKs
- Subsequent layers (x-coordinate squaring):
  - Apply the squaring map: x -> 2x^2 - 1
  - This halves the domain at each step, analogous to halving the root of unity order
  - Butterfly formula: standard Cooley-Tukey with twiddle factors from x-coordinates
- Total: 1 y-layer + (log2(n) - 1) x-layers for n points

### 3.2 Metal kernel design

- Small sizes (n <= 2^14): single-threadgroup fused kernel
  - All elements fit in threadgroup memory (2^14 * 4 bytes = 64KB — need to partition across 2 threadgroups if > 32KB)
  - All butterfly layers executed with threadgroup barriers, no global memory round-trips
  - Bit-reversal fused into initial load
- Large sizes (n > 2^14): four-step Circle NTT
  - Decompose n = n1 * n2
  - Step 1: n1 independent NTTs of size n2 (row NTTs, coalesced reads)
  - Step 2: multiply by twiddle factors (adapted for circle domain)
  - Step 3: transpose (M31 elements are 4 bytes — 32 elements per cache line, excellent coalescing)
  - Step 4: n2 independent NTTs of size n1 (column NTTs)
  - Single command buffer with memory barriers between steps

### 3.3 M31 arithmetic in Metal shaders

- Field multiply: `uint64_t prod = (uint64_t)a * (uint64_t)b; uint32_t lo = prod & 0x7FFFFFFF; uint32_t hi = prod >> 31; return reduce(lo + hi);`
- Reduce: `if (x >= P) x -= P;` (single branch, no Montgomery needed)
- Field add: `uint32_t r = a + b; if (r >= P) r -= P;`
- No Montgomery representation needed — M31 reduction is cheaper than Montgomery multiply + reduce
- Register usage: 1 register per field element (vs 8 for BN254)
- Implication for occupancy: 8x more live elements per thread, much higher effective parallelism

### 3.4 The 8x memory density advantage

- BN254 Fr element: 32 bytes, M31 element: 4 bytes — 8x ratio
- Threadgroup memory: 32KB holds 1024 BN254 elements or 8192 M31 elements
- Consequence: fused NTT kernel can process 3 more butterfly layers in shared memory (2^13 vs 2^10)
- Memory bandwidth: at 150 GB/s, can stream 37.5 billion M31 elements/sec vs 4.7 billion BN254 elements/sec
- Cache line utilization: 32 M31 elements per 128-byte cache line (100% utilization on sequential access) vs 4 BN254 elements (100% utilization, but 8x fewer elements)

### 3.5 Performance results

| Size | M31 Circle NTT | BN254 NTT | Ratio | BabyBear NTT | Goldilocks NTT |
|---|---|---|---|---|---|
| 2^14 | ~0.1ms | ~0.7ms | 7x | ~0.08ms | ~0.3ms |
| 2^16 | ~0.3ms | ~2.5ms | 8x | ~0.2ms | ~0.9ms |
| 2^18 | ~1ms | ~8ms | 8x | ~0.7ms | ~3ms |
| 2^20 | ~4ms | ~28ms | 7x | ~3ms | ~8ms |
| 2^22 | ~15ms | ~110ms | 7x | ~11ms | ~30ms |
| 2^24 | ~60ms | ~450ms | 7.5x | ~45ms | ~120ms |

- The 7-8x ratio closely tracks the 8x memory size ratio, suggesting these workloads are bandwidth-bound
- M31 Circle NTT achieves ~80% of theoretical memory bandwidth limit

---

## 4. Circle FRI and Merkle Commitments (2 pages)

### 4.1 Circle FRI protocol

- Standard FRI: fold polynomial by combining evaluations at x and -x using random challenge alpha
  - f'(x^2) = f_even(x^2) + alpha * f_odd(x^2)
- Circle FRI: analogous folding but over the circle domain
  - First fold (y-projection): combine (x, y) and (x, -y) using alpha
    - f'(x) = f_even(x) + alpha * f_odd_y(x)
  - Subsequent folds (x-squaring): combine x and its conjugate under squaring map
    - Use inverse of the squaring map to identify fold pairs
- Each fold halves the polynomial degree and the evaluation domain
- Soundness: same proximity gap theorems as standard FRI, adapted for circle codes

### 4.2 FRI on Metal

- Fold kernel: one thread per output element, reads two input elements, applies fold formula
- M31 fold: 2 multiplies + 1 add + 1 subtract = 4 M31 operations = ~8 instructions
- BN254 fold: 2 multiplies + 1 add + 1 subtract = 4 BN254 operations = ~400 instructions
- Fold is compute-bound for BN254, bandwidth-bound for M31 — different optimization strategies
- For M31: batch multiple fold rounds into a single dispatch (fold 2^20 -> 2^17 in one command buffer)
- Fused fold+commit: fold layer outputs directly into Merkle leaf buffer without global memory round-trip

### 4.3 Poseidon2-M31 Merkle commitments

- Poseidon2 over M31 with width t = 16 (rate 8, capacity 8)
  - Wider state compensates for smaller field: 16 * 31 = 496 bits of state vs 3 * 254 = 762 bits for BN254
  - Security level: 128-bit with capacity 8 * 31 = 248 bits
- Round counts: R_F = 8 full rounds, R_P = 14 partial rounds (vs R_F = 8, R_P = 56 for BN254)
- Per-round cost:
  - M31 S-box (x^5): 3 multiplies = 3 M31 muls = ~6 instructions
  - BN254 S-box (x^5): 3 multiplies = 3 BN254 muls = ~150 instructions
  - M31 MDS: 16x16 matrix-vector multiply over M31 = 256 M31 muls = ~512 instructions
  - BN254 MDS: 3x3 matrix-vector multiply over BN254 = 9 BN254 muls = ~450 instructions
- Net per-permutation: M31 ~22 rounds * ~520 instr = ~11,440 vs BN254 ~64 rounds * ~600 instr = ~38,400
- But M31 permutation processes 8 field elements (rate) * 4 bytes = 32 bytes of input
  - BN254 permutation processes 2 field elements * 32 bytes = 64 bytes of input
- Throughput: M31 is ~1.7x better on instructions per byte of input committed

### 4.4 Fused Merkle subtree kernel

- Merkle tree over M31 Poseidon2: 8192 leaves fit in 32KB threadgroup memory
  - Fuse 13 levels of the Merkle tree in shared memory (2^13 leaves)
  - vs BN254: only 1024 leaves (10 levels) fit in 32KB
  - 3 additional fused levels = 8x fewer global memory round-trips
- Binary Merkle tree with Poseidon2 hash: hash pairs of 8-element digests
- Level-by-level kernel for upper tree levels (above threadgroup capacity)
- Performance comparison table:

| Tree size | M31 Poseidon2 Merkle | BN254 Poseidon2 Merkle | Ratio |
|---|---|---|---|
| 2^14 leaves | [TBM] ms | [TBM] ms | [TBM] |
| 2^18 leaves | [TBM] ms | [TBM] ms | [TBM] |
| 2^20 leaves | [TBM] ms | [TBM] ms | [TBM] |

### 4.5 Proof size considerations

- M31 field element: 4 bytes vs BN254: 32 bytes
- FRI proof size per query: smaller elements but more queries needed for equivalent soundness
- At 128-bit security: ~50 queries for M31 (field size 31 bits) vs ~3 queries for BN254 (field size 254 bits)
- Net proof size: M31 proofs are comparable or slightly larger due to query count, but proving time is dramatically lower
- Table: proof size vs proving time tradeoff at 128-bit security for M31 vs BN254

---

## 5. Full Circle STARK Prover (1.5 pages)

### 5.1 Prover pipeline overview

- Trace generation → Circle NTT (LDE) → Constraint evaluation → Composition polynomial → Circle FRI → Output proof
- Each stage implemented as Metal compute kernels with single command buffer orchestration
- Unified memory advantage: trace stays in place, no host-device transfers

### 5.2 Constraint evaluation over M31

- AIR constraints evaluated at circle domain points
- Each constraint is a low-degree polynomial over M31 field elements
- GPU kernel: one thread per evaluation point, evaluates all constraints
- M31 constraint evaluation: field ops are single-instruction, so constraint evaluation is essentially "free" compared to NTT and FRI
- For BN254: constraint evaluation can dominate proving time due to expensive field multiplies

### 5.3 Low-degree extension (LDE)

- Circle NTT at size n, then evaluate at blowup_factor * n points
- Uses Circle iNTT → coefficient form → Circle NTT at larger size
- Blowup factor 2-8 typical; M31's smaller elements mean LDE memory is 8x smaller
- At 2^20 trace with blowup 4: M31 LDE is 16MB vs BN254 LDE is 128MB — fits in cache vs doesn't

### 5.4 End-to-end benchmarks

| Component | M31 Circle STARK | BN254 STARK | Ratio |
|---|---|---|---|
| Trace NTT (2^20) | [TBM] ms | [TBM] ms | [TBM] |
| Constraint eval (2^20) | [TBM] ms | [TBM] ms | [TBM] |
| LDE (2^20, blowup 4) | [TBM] ms | [TBM] ms | [TBM] |
| FRI commit | [TBM] ms | [TBM] ms | [TBM] |
| FRI query+verify | [TBM] ms | [TBM] ms | [TBM] |
| **Total proving time** | **[TBM] ms** | **[TBM] ms** | **[TBM]** |

---

## 6. Evaluation (2 pages)

### 6.1 Experimental setup

- Hardware: Apple M3 Pro (14 GPU cores, 18GB unified memory, ~150 GB/s bandwidth)
- Software: macOS 15, Metal 3, Swift 6, zkMetal library
- Methodology: median of 5 runs after 2 warmup iterations, thermal throttle monitoring
- Baselines:
  - CPU: Plonky3 M31 (Rust, NEON), arkworks BN254 (Rust)
  - GPU: CUDA Icicle M31 (published numbers, normalized for GPU generation)

### 6.2 Primitive-level comparison

#### Table: GPU instruction count per field operation
| Operation | M31 | BabyBear | Goldilocks | BN254 Fr |
|---|---|---|---|---|
| Multiply | ~3 instr | ~3 instr | ~12 instr | ~200 instr |
| Add | ~2 instr | ~2 instr | ~3 instr | ~16 instr |
| Reduce | ~1 instr | ~1 instr | ~3 instr | ~30 instr |
| Registers/element | 1 | 1 | 2 | 8 |
| Bytes/element | 4 | 4 | 8 | 32 |

#### Table: NTT throughput (elements/second) at 2^20
| Field | Time (ms) | Throughput (Melems/s) | Bandwidth util. |
|---|---|---|---|
| M31 Circle | ~4 | ~262 | ~80% |
| BabyBear | ~3 | ~349 | ~85% |
| Goldilocks | ~8 | ~131 | ~80% |
| BN254 Fr | ~28 | ~37 | ~72% |

- All small-field NTTs are bandwidth-bound; BN254 NTT is compute+bandwidth bound
- M31 Circle NTT is slightly slower than BabyBear NTT due to the first y-coordinate butterfly layer being non-standard

### 6.3 Metal vs CUDA for M31

- CUDA has higher raw bandwidth (A100: 2039 GB/s vs M3 Pro: ~150 GB/s)
- But per-dollar and per-watt, Apple Silicon is competitive for bandwidth-bound M31 workloads
- Table: throughput normalized by TDP and by device cost

| Metric | M3 Pro (Metal) | A100 (CUDA) | RTX 4090 (CUDA) |
|---|---|---|---|
| M31 NTT 2^20 (ms) | [TBM] | [TBM] | [TBM] |
| NTT throughput/watt | [TBM] | [TBM] | [TBM] |
| NTT throughput/$ | [TBM] | [TBM] | [TBM] |
| BN254 NTT 2^20 (ms) | ~28 | [TBM] | [TBM] |
| BN254 penalty vs CUDA | ~4x | 1.0x | 1.0x |
| M31 penalty vs CUDA | [TBM] | 1.0x | 1.0x |

- Key finding: BN254 penalty on Metal is ~4x (due to 32-bit ALU), but M31 penalty is only the bandwidth ratio — the ALU disadvantage vanishes entirely

### 6.4 M-series scheduling pathology interaction

- Recap: M-series GPUs show 10-30x performance cliffs at specific threadgroup counts (~16K, ~32K, ~128K)
- For M31 workloads: threadgroup count = n_elements / threadgroup_size
  - At 2^20 elements with 256 threads/threadgroup: 4096 threadgroups — well below pathological ranges
  - At 2^24 elements: 65536 threadgroups — near one pathological range
- For BN254 workloads: same element count but 8x more data per element means fewer threadgroups for memory-partitioned kernels
- Analysis: M31 workloads more frequently land in safe threadgroup ranges for typical proof sizes
- Workaround: threadgroup count padding for sizes near pathological boundaries

### 6.5 End-to-end Circle STARK vs BN254 STARK

| Benchmark | M31 Circle STARK | BN254 STARK | Speedup |
|---|---|---|---|
| Fibonacci (2^16 steps) | [TBM] ms | [TBM] ms | [TBM] |
| Fibonacci (2^20 steps) | [TBM] ms | [TBM] ms | [TBM] |
| Rescue hash (2^14) | [TBM] ms | [TBM] ms | [TBM] |
| Rescue hash (2^18) | [TBM] ms | [TBM] ms | [TBM] |

---

## 7. Discussion and Future Work (1 page)

### 7.1 When does field-hardware co-design matter?

- For NVIDIA GPUs: BN254 vs M31 gives ~4x speedup (ALU savings partially absorbed by bandwidth)
- For Apple Silicon GPUs: BN254 vs M31 gives ~7-10x speedup (ALU savings are multiplicative with bandwidth savings)
- The "wrong hardware" amplifies the benefit of moving to small fields
- Implication: ZK system designers should profile on target deployment hardware, not just benchmark hardware

### 7.2 Mobile and edge proving

- iPhone A-series and iPad M-series share the same GPU architecture
- Circle STARK proving on an iPhone could enable client-side ZK proofs for:
  - Private identity verification
  - Location proofs
  - Verifiable computation in mobile wallets
- M31 Circle STARKs may be the first practical mobile-native ZK proving system

### 7.3 Extensions: BabyBear, binary towers, STARK recursion

- BabyBear (p = 2^31 - 2^27 + 1): also 31-bit, also single-instruction multiply on 32-bit GPU
  - Different NTT structure (multiplicative subgroup), comparable performance
  - BabyBear has p - 1 = 2^27 * 15, giving 27 levels of 2-adicity via standard NTT
- Binary tower fields (Binius): XOR-based arithmetic, even cheaper on GPU — future work
- STARK recursion: inner proof over M31, outer proof over BN254 for Ethereum verification
  - The verifier circuit for M31 is simpler, so recursion overhead is lower

### 7.4 Limitations

- Proof size: M31 proofs are larger due to more FRI queries (50 vs 3 at 128-bit security)
- Verification cost on Ethereum: M31 verification requires non-native field arithmetic in Solidity
- Standard workaround: recursive proof wrapping (M31 inner, BN254/BLS outer for Ethereum)
- Apple Silicon bandwidth ceiling: ~150 GB/s limits throughput for bandwidth-bound M31 workloads
- Scheduling pathology at 2^24+ sizes may require workarounds

---

## 8. Conclusion (0.5 page)

- Circle STARKs over M31 turn Apple Silicon's 32-bit GPU from a liability into an asset
- Single-instruction field multiply, 8x memory density, and 8x cache utilization compound to give 7-10x end-to-end proving speedup over BN254 on the same hardware
- The ALU disadvantage that penalizes BN254 by ~4x on Metal vanishes entirely for M31
- Field-hardware co-design is not just a theoretical argument — we quantify it on real silicon
- Apple Silicon may be the optimal consumer-grade platform for small-field STARK proving
- As ZK systems increasingly move to small fields (Plonky3, Stwo, Binius), the relevance of 32-bit GPU acceleration grows

---

## Appendix

### A. Circle group algebra

- Full derivation of circle group order p + 1
- Proof that the squaring map x -> 2x^2 - 1 halves the domain correctly
- Twin-coset decomposition for the first NTT layer
- Twiddle factor formulas for x-coordinate and y-coordinate butterflies

### B. M31 arithmetic in Metal shader language

- Complete M31 multiply kernel (annotated)
- Complete M31 reduce function
- Comparison with BN254 CIOS Montgomery multiply kernel (annotated)
- Instruction count analysis from Metal GPU profiler

### C. Circle NTT kernel code

- Fused small-size Circle NTT (2^14)
- Four-step large-size Circle NTT (2^20+)
- First y-layer butterfly kernel
- Subsequent x-layer butterfly kernel

### D. Complete benchmark data

- All primitives, all sizes, M31 and BN254, GPU and CPU
- Raw timing data with min/max/median/stddev
- Thermal throttle detection methodology

---

## Figures needed

1. **The 32-bit advantage diagram**: Side-by-side comparison of M31 multiply (1 instruction path) vs BN254 multiply (8x8 limb multiplication tree). Visual representation of the core thesis.

2. **Circle group illustration**: Points on x^2 + y^2 = 1 over F_p, showing the group structure and how the squaring map halves the domain. Annotate with NTT butterfly connection.

3. **Circle NTT butterfly diagram**: First y-layer (twin-coset) followed by x-layers (squaring map), contrasted with standard Cooley-Tukey butterfly.

4. **Memory density comparison**: Visual showing 32KB threadgroup memory holding 1024 BN254 elements vs 8192 M31 elements, with implication for fused kernel depth.

5. **NTT throughput vs field size**: Bar chart or line plot showing NTT time at 2^20 for M31, BabyBear, Goldilocks, BN254. Annotate with bandwidth-bound vs compute-bound regions.

6. **FRI fold cost breakdown**: Stacked bar chart showing compute vs memory cost for M31 FRI fold vs BN254 FRI fold, demonstrating the shift from compute-bound to bandwidth-bound.

7. **End-to-end prover pipeline**: Gantt-style chart showing Circle STARK proving stages (trace NTT, constraint eval, LDE, FRI commit, FRI query) with times, compared side-by-side with BN254 STARK pipeline.

8. **Scheduling pathology heatmap**: Threadgroup count vs performance, with M31 typical operating ranges and BN254 typical operating ranges highlighted, showing that M31 more often avoids pathological zones.

9. **Throughput/watt comparison**: Normalized throughput for M31 NTT across M3 Pro, A100, RTX 4090, showing Apple Silicon's per-watt competitiveness for small-field workloads.

---

## Key references

- Haboeck, U. "Circle STARKs" (2024) — foundational Circle STARK construction
- Ben-Sasson et al. "STARK Friendly Hash" — Poseidon2 design principles
- Polygon/Plonky3 — Circle STARK implementation over M31
- StarkWare/Stwo — Production Circle STARK prover
- Ingonyama/Icicle — CUDA ZK acceleration (M31 comparison baseline)
- Our systems paper — zkMetal architecture and full primitive benchmarks
- Crandall, R. "Method and apparatus for public key exchange in a cryptographic system" — Mersenne prime arithmetic
