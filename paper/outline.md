# zkMetal: Exploiting Unified Memory and 32-bit ALUs for Zero-Knowledge Cryptography on Apple Silicon GPUs

## Paper Outline (USENIX Security / arXiv)

Target length: 12--15 pages, USENIX format (2-column, 10pt).

---

## Abstract (~200 words)

Claim: Apple Silicon's unified memory architecture and 32-bit GPU ALU create a fundamentally different optimization landscape for zero-knowledge cryptography than discrete GPUs with native 64-bit integer units. We present zkMetal, the first comprehensive ZK cryptography library for Metal GPUs, covering 60+ primitives across 18 fields and 10 elliptic curves. We report three categories of findings: (1) a previously undocumented M-series GPU scheduling pathology where specific threadgroup/bucket counts cause 10--30x performance cliffs, requiring workload-aware configuration tables; (2) a systematic methodology for field-hardware co-design, showing that 32-bit ALU limitations become advantages for small fields (BabyBear 8.5B elem/s NTT, Mersenne31 fused kernels) while requiring CIOS Montgomery with explicit carry management for 256-bit fields; (3) a C/NEON acceleration layer that achieves 10--134x over baseline Swift across all CPU primitives. Against ICICLE-Metal (the only other Metal ZK library), zkMetal is 30--500x faster on NTT and 6--40x faster on MSM. We characterize performance relative to hardware theoretical floors, finding primitives ranging from 1.2x (BabyBear NTT) to 9x (BN254 MSM) of limits, and catalog 30+ dead-end optimizations as negative results.

---

## 1. Introduction (1.5 pages)

**What this section covers:**
- Motivation: ZK proofs increasingly run on client devices (mobile wallets, browser proving, edge verification). Apple Silicon powers >50% of premium mobile/laptop market. No mature ZK library exists for Metal GPU.
- Gap: ICICLE-Metal exists but has ~85ms per-call overhead and limited primitive coverage. Arkworks is CPU-only. CUDA libraries (e.g., ICICLE-CUDA) target fundamentally different hardware (native 64-bit int multiply, discrete memory).
- Thesis: Apple Silicon's unified memory eliminates PCIe transfer overhead but its 32-bit ALU decomposes 64-bit multiplies into 4x32-bit ops, creating a novel optimization landscape. We show how to exploit this systematically.
- Scope: 60+ primitives, 18 fields, 10 curves, full proof systems (Circle STARK, Plonk, Groth16, GKR, HyperNova, Spartan), post-quantum (Kyber, Dilithium), and application primitives (ECDSA, BLS, Schnorr, EdDSA).

**Key figure:** Figure 1 -- zkMetal vs ICICLE-Metal vs Arkworks CPU across MSM/NTT/hash at various sizes (bar chart). Shows 30--500x gaps.

**Data points to include:**
- MSM BN254 2^18: zkMetal 45ms vs ICICLE-Metal 1475ms vs Arkworks 266ms
- NTT BN254 2^20: zkMetal 6.1ms vs ICICLE-Metal 194ms
- NTT BabyBear 2^24: zkMetal 2.0ms vs ICICLE-Metal 709ms

---

## 2. Background (1.5 pages)

**What this section covers:**

### 2.1 Apple Silicon GPU Architecture (~0.5 page)
- M-series tile-based deferred rendering GPU. Unified memory (CPU/GPU share physical DRAM, no PCIe). 32-bit ALU (no native 64-bit integer multiply; `mulhi(ulong,ulong)` decomposes to 4 MADs). Threadgroup shared memory: 32KB. SIMD width: 32 threads. Dispatch model: command buffers -> command encoder -> compute pipelines.
- Contrast with NVIDIA: CUDA has native `uint64` multiply, separate VRAM with high bandwidth (900+ GB/s on A100 vs 150 GB/s on M3 Pro), and mature compiler toolchain.

### 2.2 ZK Primitives Overview (~0.5 page)
- MSM (Pippenger), NTT (Cooley-Tukey), algebraic hashing (Poseidon2), FRI, sumcheck -- standard definitions, computational profiles.
- Montgomery CIOS for modular arithmetic. GLV endomorphism for curves with efficient endomorphisms.

### 2.3 Related Work (~0.5 page)
- ICICLE (Ingonyama): CUDA-first, Metal port with performance issues. Cite their benchmarks.
- cuZK, ZPrize submissions (2022, 2023), MoPro v2 (cite their blog benchmarks on M3 Air).
- Arkworks (CPU baseline), Plonky3 (BabyBear focus), gnark (Go).
- No prior work systematically characterizes Metal GPU for ZK.

---

## 3. Architecture (1.5 pages)

**What this section covers:**

### 3.1 Three-Tier Execution Model (~0.5 page)
- **Tier 1: Metal GPU shaders** -- compute-intensive parallel kernels (MSM bucket accumulation, NTT butterflies, hash batches). Metal Shading Language (C++14 dialect). Buffer management via grow-only caching to amortize allocation.
- **Tier 2: C/NEON CPU kernels** -- `__uint128_t` CIOS Montgomery, NEON SIMD for small fields. Linked via Swift Package Manager C targets. Used when GPU dispatch overhead exceeds compute savings (n < crossover threshold, typically 2^12--2^14).
- **Tier 3: Swift orchestration** -- engine layer managing pipeline selection, buffer lifecycle, auto-tuning, command buffer batching.

**Key figure:** Figure 2 -- Architecture diagram showing three tiers with data flow. Annotate unified memory zero-copy paths.

### 3.2 Field Arithmetic on 32-bit ALU (~0.5 page)
- 256-bit fields (BN254, BLS12-381): 8x32-bit limbs, CIOS Montgomery with explicit carry chains. Each `fp_mul` = ~64 MAD instructions. SOS (Sum-of-Squares) squaring optimization saves ~25% (101->77ms in proof context).
- 64-bit fields (Goldilocks): special reduction p = 2^64 - 2^32 + 1 allows single-word overflow handling.
- 31-bit fields (BabyBear, M31): native 32-bit arithmetic. Single MAD per multiply. 8x density advantage over 256-bit fields. This is where Metal's 32-bit ALU becomes an *advantage* over 64-bit CUDA GPUs that waste half their ALU width.
- Binary tower (Binius): XOR addition is literally free on GPU; multiply via lookup tables.

**Data point:** BabyBear NTT 2^24 at 8.5B elem/s vs BN254 NTT 2^24 at 144M elem/s -- 59x throughput ratio for 8x element size ratio, showing super-linear benefit from field size reduction.

### 3.3 Unified Memory Exploitation (~0.5 page)
- Zero-copy buffer sharing: CPU prepares scalars/points, GPU reads without transfer. Eliminates PCIe-equivalent overhead.
- Buffer caching: grow-only pattern across all engines. First call allocates, subsequent calls reuse. Measured: FRI 2.4x, Keccak 28% improvement from caching alone.
- Single command buffer batching: encode multiple dispatches into one CB to amortize ~0.5ms per-CB overhead. NTT BN254 2^16: 2x faster with single-encoder pattern.

---

## 4. GPU Scheduling Pathology (1.5 pages) [NOVEL CONTRIBUTION]

**What this section covers:**

This is the paper's most novel finding. Specific threadgroup counts on M-series GPUs cause catastrophic 10--30x performance degradation. This is not documented by Apple and was discovered empirically across multiple independent optimization efforts.

### 4.1 Phenomenon Description (~0.5 page)
- MSM bucket accumulation dispatches one thread per bucket. At windowBits=14, nBuckets=16384, MSM takes 7000ms instead of expected ~50ms (140x regression).
- secp256k1 at wb=12: 2049 buckets causes 78ms vs wb=13's 21ms at 2^14 (3.7x). wb=11 at 1025 buckets also pathological.
- BLS12-377 at wb=14 with 12-limb Fq: 17000ms vs wb=15's 569ms at 2^17 (30x regression).
- GLV on secp256k1 doubles effective points, hitting 32K-bucket pathology: 13569ms vs 1133ms (12x regression).

**Key figure:** Figure 3 -- Heatmap of MSM latency vs (windowBits, numPoints) across BN254, secp256k1, BLS12-377. Show pathological bands at specific bucket counts.

### 4.2 Root Cause Hypothesis (~0.5 page)
- Pathological counts appear near powers of 2 and their neighbors (1025, 2049, 16384, 32768). Hypothesis: Metal's tile-based scheduler has occupancy cliffs when threadgroup grids align poorly with the GPU's execution unit geometry.
- Register pressure interaction: larger fields (12-limb BLS12-377) exacerbate the effect because each thread uses more registers, reducing occupancy further at the cliff.
- Not reproducible on NVIDIA GPUs -- CUDA's warp scheduler does not exhibit this behavior.

### 4.3 Mitigation (~0.5 page)
- Per-curve, per-size window selection tables derived empirically. BN254: wb=16 for >32K points. secp256k1: wb=13 for <=64K, wb=16 for >64K. BLS12-377: avoid wb=14 entirely.
- Signed-digit decomposition halves effective bucket count, sidestepping some pathological bands.
- Recommendation for Apple: expose threadgroup scheduling hints or document occupancy model.

**Table:** Table 1 -- Pathological (windowBits, curve) combinations and measured regression factors.

---

## 5. Optimization Methodology (2.5 pages)

### 5.1 MSM Optimization (1 page)
- Starting point: naive GPU Pippenger at 270ms (BN254, 2^18).
- Optimization stack (cumulative): signed-digit decomposition (42%), GLV endomorphism (22%), CIOS Montgomery (15%), Z=1 mixed affine addition (8%), single command buffer (13%), precomputation caching (5%). Final: 45ms.
- Signed-digit detail: carry chain converts w-bit digits to [-(2^(w-1)-1), 2^(w-1)], halving bucket count. Sign bit in bit 31 of sorted index; GPU checks sign and negates point.y.
- Metal compiler noinline requirement: large-struct-returning functions (PointProjective with 3x Fp = 96 bytes) are miscompiled when inlined. `__attribute__((noinline))` is mandatory on point_double, point_add_mixed, point_add.
- Timing breakdown at 45ms: GLV 2ms, precomp 1.6ms, sort 2.4ms, GPU reduce 15.5ms, bucket_sum+combine 8.8ms, Horner 0.5ms. GPU reduce (bucket accumulation) is 50% -- the fundamental bottleneck is random-access memory writes during scatter.

**Key figure:** Figure 4 -- Waterfall chart showing MSM optimization progression from 270ms to 45ms with each optimization's contribution.

**Data points:**
- BN254 2^18: 45ms (vs Arkworks CPU 266ms = 5.9x, vs ICICLE-Metal 1475ms = 33x)
- CPU C Pippenger crossover at 2^14 (29ms CPU vs 22ms GPU)
- secp256k1 2^18: 113ms (no GLV due to pathology, signed-digit only)

### 5.2 NTT Optimization (0.75 page)
- Four-step FFT for large sizes (2^20+): split into row NTT + twiddle multiply + column NTT. Avoids single-kernel shared memory limits.
- Fused bitrev+butterfly: eliminate separate bit-reversal pass. 47% faster at 2^16.
- Twiddle fusion: precompute and fuse twiddle factors into butterfly. 4--5x at 2^24.
- Small-field advantage: BabyBear NTT at 2^24 runs in 2.0ms (8.5B elem/s) because each butterfly is a single 32-bit MAD instead of 64 MADs for BN254.
- BN254 2^22 at 26ms is near-optimal for 32B elements: 127x theoretical headroom is unreachable because strided column access at 32B elements x 128B cache lines = 25% utilization is fundamental to the four-step algorithm. Three alternative approaches tested and failed (transpose-before-column, swapped split, phase profiling).

**Key figure:** Figure 5 -- NTT throughput (elements/sec) vs field size (bits) at 2^24, showing super-linear scaling.

**Data points:**
- BabyBear 2^24: 8.5B elem/s
- Goldilocks 2^24: 5.7B elem/s
- BN254 2^24: 144M elem/s
- vs ICICLE-Metal: 30--500x faster across all fields and sizes

### 5.3 C CIOS Acceleration (~0.75 page)
- Methodology: identify hot loops in CPU paths, replace with C using `__uint128_t` for 64-bit CIOS Montgomery multiply (avoids Swift's lack of 128-bit integer support). NEON SIMD for BabyBear (4-wide uint32 Montgomery butterfly).
- Results across primitives: GKR 31x, MSM 347--1024x over Swift Pippenger, BN254 NTT 29--38x, Verkle 24--134x, ECDSA 57x, HyperNova 40x, Spartan 8.6x, Lasso 8.2x.
- Key insight: Swift's optimizer handles 256-bit field arithmetic well (80--90% of C), but is catastrophically bad for inner-loop scalar operations where function call overhead dominates (Pippenger bucket loops, NTT butterflies).
- NEON BabyBear NTT: 4-wide uint32 Montgomery butterfly processes 4 elements per instruction. 5.8x over Swift. Plonky3 technique adapted for ARM64.

**Table:** Table 2 -- C CIOS speedup factors across all primitives with before/after times.

---

## 6. Performance Characterization (2 pages)

### 6.1 Comparison with Baselines (1 page)

**Key tables:**
- Table 3: MSM comparison -- zkMetal vs ICICLE-Metal vs ICICLE CPU vs MoPro v2 vs Arkworks vs ICICLE-CUDA. Sizes 2^16 through 2^20. Note: ICICLE-Metal v3.8 has ~600ms license-server overhead; ICICLE-CUDA on RTX 3090 Ti still wins at ~9ms for 2^16 (native 64-bit mul).
- Table 4: NTT comparison -- zkMetal vs ICICLE-Metal across BN254 and BabyBear. 30--500x gaps.
- Table 5: End-to-end proof systems -- Circle STARK (21ms prove at 2^14), Plonk (49ms prove at 1024 gates), Groth16 (14ms prove at 256 constraints), HyperNova (0.09ms/fold), Spartan (121ms at 2^14), Lasso (56ms prove at 2^18).
- Table 6: Cross-curve MSM -- BN254, BLS12-377, secp256k1, Pallas, Vesta, Ed25519, Grumpkin at 2^14 through 2^18.

**Key figure:** Figure 6 -- Log-scale bar chart of zkMetal vs ICICLE-Metal across all measured primitives.

**Key data points for CUDA comparison:**
- ICICLE-CUDA RTX 3090 Ti MSM 2^16 ~9ms vs zkMetal M3 Pro 27ms (3x gap, expected from memory BW ratio 900/150 = 6x and native 64-bit mul)
- Argument: gap narrows for small fields where 32-bit ALU is not a disadvantage

### 6.2 Theoretical Floor Analysis (1 page)
- Methodology: compute-bound floor = total_ops / 3.6 TFLOPS. Memory-bound floor = total_bytes / 150 GB/s. Dispatch-bound floor = N_dispatches x 0.5ms. Take max.
- Results table (33 primitives ranked by headroom). Highlight:
  - Near-optimal (< 2x headroom): BabyBear NTT 1.2x, Circle NTT 1.3x, HyperNova 1.3x, KZG commit 1.4x, Groth16 prove 1.4x, GKR 1.5x.
  - Moderate headroom (2--5x): Radix sort 2x, Circle STARK 2x, Lasso 2x, ECDSA 3x, Plonk 3x, secp256k1 MSM 4x, Poseidon2 batch 4.5x.
  - Significant headroom (5--10x): MSM BN254 9x, NTT BN254 9x, Sumcheck 9x, Basefold 7x, FRI 7x.
- Discussion: headroom is dominated by memory access patterns (MSM scatter, NTT strided access) not compute inefficiency. Primitives closest to floor are those with regular access patterns or small element sizes.

**Key figure:** Figure 7 -- Scatter plot: current performance vs theoretical floor for all 33 primitives. Diagonal = 1x (optimal). Annotate outliers.

---

## 7. Negative Results and Dead Ends (1.5 pages) [NOVEL CONTRIBUTION]

**What this section covers:**
Systematic catalog of 30+ optimization attempts that failed. Organized by failure mode. This section is intentionally detailed -- negative results prevent duplicated effort and reveal hardware characteristics.

### 7.1 Hardware Mismatch (~0.5 page)
- **64-bit limb CIOS on Metal GPU**: 67% slower despite halving limb count. Root cause: `mulhi(ulong,ulong)` decomposes to 4x32-bit MADs internally -- zero benefit, more register pressure. Demonstrates that Metal has no 64-bit integer hardware at all.
- **Custom AArch64 assembly for Montgomery**: 23% slower than LLVM. Clang already emits optimal MUL+UMULH pairs for `__uint128_t` multiply. Confirmed independently by two repositories. Conclusion: do not hand-optimize ARM64 Montgomery.
- **SIMD cooperative reduce for MSM at w=13**: 2.4x worse due to shuffle overhead for 256-bit points (3x Fp = 96 bytes per shuffle vs 4 bytes for typical GPU values). Metal SIMD shuffles are designed for small types.

### 7.2 Algorithmic Dead Ends (~0.5 page)
- **GLV endomorphism on non-BN254 curves**: Verified correct but regresses on secp256k1 (12x worse, scheduling pathology) and BLS12-377 (1.27x worse, 12-limb point ops dominate). GLV is only net-positive when point addition cost is low relative to scalar width reduction.
- **Radix sort local-binning**: 5 variants tested (serial ranking, SIMD ranking, linear binary-search scatter, atomic-based, sequential SIMD). All slower or same as baseline. Apple Silicon's large L2 cache gives only 6--9x actual write amplification (not theoretical 32x), making coalescing optimization unprofitable.
- **Lazy field reduction in Poseidon2**: Value range exceeds 2^256 after only 2 rounds. Cannot amortize reductions.
- **Fused Poseidon2/Blake3 Merkle subtree kernels**: 80% idle thread waste (tree narrows at upper levels), barrier overhead exceeds benefit for lightweight hash functions (Blake3).
- **Transpose-before-column-FFT for BN254 NTT 2^22**: 4 transposes + 4 blits cost ~50ms, exceeding the coalescing benefit for 32B elements.
- **Circle STARK eliminating traceLDE readback**: 38x regression (21ms->800ms). CPU FRI fold needs data on CPU; the readback itself is only ~0.1ms.

### 7.3 Metal Compiler Issues (~0.5 page)
- **Inline miscompilation**: Functions returning large structs (PointProjective, 96 bytes) produce wrong results when inlined. Requires `__attribute__((noinline))`. Discovered when 5/9 MSM tests failed after removing noinline. This is a genuine Metal compiler bug.
- **Fused reduce+bucket_sum kernel**: Correct but 21% slower because only 4096 threads -- Metal cannot efficiently schedule underutilized grids even when per-thread work is substantial.
- **Uninitialized output buffers via storageModePrivate**: All buffers need CPU access for unified memory; buffer caching already eliminates repeated zeroing.

**Table:** Table 7 -- Dead-end catalog with optimization, expected gain, actual result, root cause, and affected primitive.

---

## 8. Lazy Montgomery and GPU Boundary Correctness (0.5 page)

**What this section covers:**
- Barretenberg (and other libraries) store field elements in lazy-reduced Montgomery form [0, 2p). GPU fp_add/fp_sub assume fully reduced inputs [0, p). This caused ~10% error rate in MSM bucket accumulation.
- Fix: `fp_reduce()` at GPU entry points (point_from_affine, glv_endomorphism kernel). GPU-internal arithmetic stays in [0, p).
- Generalization: any heterogeneous CPU/GPU system must normalize field element representations at boundaries. Unified memory makes this invisible -- data sharing is zero-copy but semantic mismatch is not.
- Also: secp256k1 signed-digit carry overflow bug. Scalars near n (curve order, ~2^256) caused top window digit + carry > halfBuckets. Fix: scalar centering -- if scalar > n/2, use (n-scalar) and negate point.

---

## 9. Proof System Integration (1 page)

**What this section covers:**
Brief treatment of how primitives compose into complete proof systems, with end-to-end numbers.

### 9.1 Circle STARK over Mersenne31
- Full pipeline: Circle NTT for LDE (0.9ms), GPU Keccak Merkle (6ms), GPU constraint eval (2ms), CPU FRI fold (12ms). Total prove: 21ms at 2^14 trace.
- M31 single-word arithmetic makes every kernel compute-dense. Fused constraint evaluation kernel processes entire AIR in one dispatch.
- GPU witness generation: instruction-stream architecture, 877M cells/s for BN254, 1.5B cells/s for M31.

### 9.2 Plonk with KZG
- C CIOS constraint evaluation, batched polynomial ops, Keccak transcript. Prove 49ms at 1024 gates (3.2x over baseline).
- Runtime constraint IR compilation: arbitrary R1CS/AIR constraints -> Metal source -> GPU pipeline. 248M constraints/s.

### 9.3 Folding and Transparent Proofs
- HyperNova CCS folding: 0.09ms/fold. 40x improvement from C CIOS + Keccak transcript + pre-computed affine points.
- Spartan: 121ms prove at 2^14 (8.6x over baseline). Transparent SNARK, no trusted setup.
- Lasso: 56ms prove at 2^18 (8.2x). C-accelerated batch decompose + fused single CB per chunk.

### 9.4 Groth16 and Pairings
- GPU BN254 pairing: 4.7x at batch n=16 (51ms vs 239ms CPU). Projective Miller loop.
- Groth16 prove: 14ms at 256 constraints (107x improvement). Verification valid.
- BLS12-381 full tower: sign 26ms, verify 78ms. BLS aggregate signatures working.

**Table:** Table 8 -- End-to-end proof system benchmarks with component-level breakdown.

---

## 10. Discussion (1 page)

### 10.1 When to Use GPU vs CPU
- Crossover analysis: GPU wins at n >= 2^12--2^14 depending on primitive. Below that, dispatch overhead dominates. C CIOS CPU is competitive for small instances.
- Adaptive selection: zkMetal automatically selects execution path based on input size.
- CPU C Pippenger at 2^16: 68ms (matches Arkworks 69ms). Demonstrates that C CIOS methodology alone achieves state-of-the-art CPU performance.

### 10.2 The Small-Field Thesis
- 32-bit ALU is *not* a limitation for small-field ZK (BabyBear, M31). The trend toward smaller fields in modern proof systems (Plonky3, Circle STARKs, Binius) plays to Apple Silicon's strengths.
- BabyBear NTT at 2^24: 2.0ms. Near hardware floor (1.2x). This is essentially memory-bandwidth-limited, not compute-limited.
- Binary tower (Binius): XOR addition = free on GPU, multiply = table lookup. Metal's 32-bit ALU is ideal.
- Prediction: as proof systems migrate to smaller fields, Apple Silicon's competitiveness vs CUDA narrows.

### 10.3 Unified Memory as Architectural Advantage
- Eliminates transfer cost for streaming verification (task-queue pipeline).
- Enables incremental Merkle updates (CPU append, GPU re-hash) with zero-copy.
- Composability: proof system components share buffers without marshaling.

### 10.4 Limitations
- Apple-only (Metal is proprietary). No cross-platform path except WebGPU (lossy translation).
- No formal verification of GPU arithmetic correctness -- tested empirically with extensive test suites.
- Some primitives crash on certain hardware (Brakedown, Zeromorph signal 139).
- ICICLE-CUDA on RTX 3090 Ti is still faster for large MSM (~9ms vs 27ms at 2^16) due to native 64-bit multiply and higher memory bandwidth (900 vs 150 GB/s).
- Metal compiler opacity: no PTX-level control, must reverse-engineer behavior via empirical testing (noinline bug, scheduling pathology).

---

## 11. Conclusion (0.5 page)

- Summarize contributions: scheduling pathology discovery, field-hardware co-design methodology, comprehensive performance characterization, negative result catalog.
- Main message: Apple Silicon is a viable ZK proving platform, especially for small-field systems (BabyBear, M31, binary towers) where 32-bit ALU is advantageous, not limiting.
- Impact: 60+ primitives, 18 fields, 10 curves -- most comprehensive Metal ZK implementation by an order of magnitude.
- Future work: Rust FFI for integration with SP1/Halo2/Plonky3, constraint IR compilation for general-purpose STARK proving, WebGPU port for browser proving, multi-GPU support for M3/M4 Ultra.

---

## Appendix A: Complete Benchmark Tables (1--2 pages)

All benchmark data from README in tabular form:
- MSM (all 7 curves, all sizes, GPU + C Pippenger + Swift Pippenger + vanilla CPU)
- NTT (4 fields, GPU and CPU, plus ICICLE-Metal comparison)
- Hashing (Poseidon2, Keccak, Blake3, SHA-256 -- batch and Merkle)
- FRI (fold-by-2/4/8, commit phase)
- Sumcheck (dense + sparse)
- Polynomial ops (multiply, multi-eval)
- KZG / Batch KZG / Basefold
- Circle STARK / Plonk / Groth16 / GKR
- Advanced protocols (HyperNova, Spartan, Lasso, LogUp, IPA, Verkle, Tensor, WHIR)
- CPU optimizations (C CIOS, NEON, batch field ops)
- Application primitives (Kyber, Dilithium, ECDSA, BLS, Schnorr, EdDSA, Reed-Solomon, HE NTT)
- Witness generation, constraint IR, streaming verification

## Appendix B: Dead-End Catalog (0.5--1 page)

Full table of 30+ failed optimizations with columns:
- Optimization attempted
- Primitive affected
- Expected improvement
- Actual result
- Root cause of failure
- Time invested (sessions)

---

## Figure/Table Summary

| ID | Type | Content | Section |
|----|------|---------|---------|
| Fig 1 | Bar chart | zkMetal vs ICICLE-Metal vs Arkworks (MSM/NTT/hash) | 1 |
| Fig 2 | Diagram | Three-tier architecture with unified memory data flow | 3 |
| Fig 3 | Heatmap | MSM latency vs (windowBits, numPoints) showing pathological bands | 4 |
| Fig 4 | Waterfall | MSM optimization progression 270ms -> 45ms | 5 |
| Fig 5 | Line chart | NTT throughput vs field size (bits) at 2^24 | 5 |
| Fig 6 | Bar chart | Log-scale zkMetal vs ICICLE-Metal all primitives | 6 |
| Fig 7 | Scatter | Current perf vs theoretical floor (33 primitives) | 6 |
| Tab 1 | Table | Pathological (windowBits, curve) combinations + regression factors | 4 |
| Tab 2 | Table | C CIOS speedup factors across all primitives | 5 |
| Tab 3 | Table | MSM comparison (6 implementations, 3 sizes) | 6 |
| Tab 4 | Table | NTT comparison (zkMetal vs ICICLE-Metal, 2 fields, 5 sizes) | 6 |
| Tab 5 | Table | End-to-end proof system benchmarks | 6 |
| Tab 6 | Table | Cross-curve MSM comparison (7 curves) | 6 |
| Tab 7 | Table | Dead-end catalog (30+ entries) | 7 |
| Tab 8 | Table | Proof system component breakdown | 9 |

---

## Page Budget

| Section | Pages |
|---------|-------|
| Abstract | 0.25 |
| 1. Introduction | 1.5 |
| 2. Background | 1.5 |
| 3. Architecture | 1.5 |
| 4. GPU Scheduling Pathology | 1.5 |
| 5. Optimization Methodology | 2.5 |
| 6. Performance Characterization | 2.0 |
| 7. Negative Results | 1.5 |
| 8. Lazy Montgomery Correctness | 0.5 |
| 9. Proof System Integration | 1.0 |
| 10. Discussion | 1.0 |
| 11. Conclusion | 0.5 |
| References | 0.75 |
| **Total body** | **~14.5** |
| Appendix A (benchmarks) | 1.5 |
| Appendix B (dead ends) | 0.75 |

---

## Writing Priority Order

Sections ranked by novelty and reviewer impact (write these first):
1. **Section 4** (scheduling pathology) -- most novel, no prior documentation
2. **Section 7** (negative results) -- high value for community, unusual for systems papers
3. **Section 5.1** (MSM optimization) -- detailed case study with quantified contributions
4. **Section 3.2** (field arithmetic on 32-bit ALU) -- core technical insight
5. **Section 6.2** (theoretical floor analysis) -- rigorous characterization methodology
6. Remaining sections in order

---

## Key Claims to Verify Before Submission

- [ ] ICICLE-Metal v3.8 overhead is license-server, not architectural (check if newer version fixes)
- [ ] RTX 3090 Ti MSM ~9ms at 2^16 (source: Ingonyama GitHub, verify specific configuration)
- [ ] MoPro v2 benchmarks (source: zkmopro.org/blog, verify M3 Air configuration)
- [ ] All zkMetal benchmarks re-run on clean system with thermal management
- [ ] Scheduling pathology reproduced on M1/M2/M4 to confirm cross-generation
- [ ] noinline miscompilation tested on latest Xcode/Metal compiler version
