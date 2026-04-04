# zkMetal: GPU-Accelerated Zero-Knowledge Proving on Apple Silicon

## Target venue
IACR ePrint (immediate), then submit to IEEE S&P / USENIX Security / CCS (applied crypto track)

## Authors
[TBD]

---

## Abstract (~250 words)
- Zero-knowledge proof systems are computationally intensive; GPU acceleration is critical for practical proving times
- All existing GPU ZK implementations target NVIDIA CUDA; Apple Silicon GPUs (M-series) are unexplored despite wide deployment
- We present zkMetal, a comprehensive GPU-accelerated ZK proving library for Apple Metal
- Key challenges: 32-bit native ALU width (vs CUDA's 64-bit), 32KB shared memory limit, ~5ms dispatch overhead, M-series GPU scheduling pathologies at specific workload sizes
- We implement 22+ primitives spanning field arithmetic, MSM, NTT, hash functions, polynomial commitments, and proof protocols across 7 fields and 5 elliptic curves
- Results: MSM 1300× over CPU, NTT up to 300×, witness generation 7800×, constraint evaluation 2400×
- Circle NTT over Mersenne31 achieves 7× speedup over BN254 NTT on the same hardware, quantifying the field-size advantage on real GPU silicon
- We report a previously undocumented M-series GPU scheduling pathology causing 10-30× performance cliffs at specific bucket counts
- Our work demonstrates that Apple Silicon GPUs are viable — and in some configurations competitive — for ZK proving workloads

---

## 1. Introduction (2 pages)

### 1.1 Motivation
- ZK proofs are the bottleneck for rollups, privacy systems, verifiable computation
- Proving time dominates: minutes for complex circuits, needs to reach seconds
- GPU acceleration is the primary path (ZPrize competitions, Icicle, etc.)
- NVIDIA GPUs dominate research and deployment, but Apple Silicon is ubiquitous on developer machines (100M+ M-series Macs shipped)
- No comprehensive study of ZK proving on Metal/Apple GPU exists

### 1.2 Challenges unique to Metal/Apple Silicon
- 32-bit native ALU: no hardware 64-bit multiply, forcing 8×32-bit limb decomposition for 256-bit fields
- Unified memory architecture: no PCIe transfer overhead, but shared bandwidth
- 32KB threadgroup (shared) memory: constrains fused kernel designs
- Command buffer dispatch overhead: ~5ms per dispatch, orders of magnitude higher than CUDA kernel launch
- No equivalent to CUDA's cooperative groups, warp-level primitives are limited
- Metal compiler opacity: no PTX-level control, must guide via attributes

### 1.3 Contributions
1. First comprehensive GPU ZK library for Apple Metal (22+ primitives, 7 fields, 5 curves)
2. Architectural analysis of M-series GPU for cryptographic workloads
3. Discovery of M-series scheduling pathology at specific bucket counts
4. Optimization techniques: fused Merkle subtrees, bit-interleaved Keccak, dispatch batching, Metal compiler workarounds
5. Performance comparison: GPU vs CPU, Metal vs CUDA (where published numbers exist)
6. Open-source implementation with Rust FFI for ecosystem integration

---

## 2. Background (2 pages)

### 2.1 Zero-knowledge proof primitives
- Field arithmetic (Montgomery multiplication, CIOS method)
- Multi-scalar multiplication (MSM) — Pippenger's bucket method
- Number theoretic transform (NTT) — Cooley-Tukey / Gentleman-Sande
- Hash functions — algebraic (Poseidon2) and symmetric (Keccak-256, Blake3)
- Polynomial commitment schemes — KZG, FRI, IPA, Basefold

### 2.2 Apple Silicon GPU architecture
- M-series GPU core structure: execution units, SIMD width 32, ALU pipeline
- Memory hierarchy: registers, threadgroup memory (32KB), device memory, unified RAM
- Command buffer model: encoder → commit → schedule → execute
- Comparison table: M3 Pro vs A100 vs RTX 4090 (specs relevant to ZK)

### 2.3 Related work
- CUDA: Icicle (Ingonyama), cuZK, PipeZK, GZKP
- WebGPU: ZPrize 2023 submissions
- FPGA: various academic works
- CPU: arkworks, blst, Plonky3
- Gap: no comprehensive Metal/Apple Silicon study

---

## 3. System Design (3 pages)

### 3.1 Architecture overview
- Swift orchestration layer → Metal compute shaders → GPU execution
- Shader compilation: runtime source concatenation (field arithmetic + kernel)
- Buffer management: grow-only caching to amortize allocation
- Figure: system architecture diagram

### 3.2 Field arithmetic on 32-bit GPU
- CIOS Montgomery multiplication with 8×32-bit limbs
- Carry chain analysis: why 32-bit CIOS works (fits in 256 bits between reductions)
- Lazy reduction: [0, 2p) intermediate values to skip reductions between operations
- Comparison: 8×32-bit CIOS vs theoretical 4×64-bit (which Metal can't do natively)
- Table: instruction count per field operation across field sizes

### 3.3 Multi-scalar multiplication
- Pippenger's bucket method adapted for Metal
- Signed-digit scalar recoding with GLV endomorphism
- GPU pipeline: scalar decomposition → counting sort → bucket accumulation → reduction
- Single command buffer batching to minimize dispatch overhead
- Window size selection: w=16 for BN254 (optimal on M3 Pro)

### 3.4 Number theoretic transform
- Four-step FFT for large sizes (> threadgroup memory)
- Fused bitreverse + butterfly kernel for small sizes
- Twiddle factor fusion: precompute and embed in butterfly
- Circle NTT for Mersenne31: adapted FFT over the circle group

### 3.5 Hash functions and Merkle trees
- Poseidon2: inline permutation, fused Merkle subtree kernel (1024 leaves in shared memory)
- Keccak-256: bit-interleaved representation for 32-bit native width
- Blake3: direct mapping of 4×4 state matrix to threadgroup computation
- Variable-size fused kernels: single dispatch for trees of 2-1024 leaves

### 3.6 Polynomial commitment and proof protocols
- FRI with fold-by-2 and fold-by-4
- Basefold: NTT-free multilinear commitment via sumcheck folding
- KZG with batch openings
- GPU witness generation via instruction-stream architecture
- Runtime constraint compilation: IR → Metal source → GPU pipeline

---

## 4. Optimization Techniques (3 pages)

### 4.1 Dispatch overhead mitigation
- Problem: each Metal dispatch costs ~5ms (vs ~10μs for CUDA kernel launch)
- Solution 1: single command buffer with memory barriers between dependent phases
- Solution 2: batch independent work into fewer dispatches
- Solution 3: fused kernels that do multiple levels in shared memory
- Quantification: FRI commit from 14 dispatches (72ms) to 3 dispatches (20ms)

### 4.2 Metal compiler management
- Register pressure: Metal compiler inlines aggressively, causing spills
- Fix: `__attribute__((noinline))` on helper functions (Poseidon2 S-box)
- Measured impact: 15% regression when compiler over-inlines
- Fast math: enabling `fastMathEnabled` for non-security-critical paths

### 4.3 Memory access patterns
- BN254 NTT at 2^22: 32-byte elements × 128-byte cache lines = 25% utilization on strided access
- Four-step FFT to improve locality: row-NTT (coalesced) → transpose → column-NTT
- Unified memory advantage: no explicit host↔device transfers

### 4.4 Threadgroup memory exploitation
- Fused Merkle subtrees: 1024 Fr elements = 32KB, exactly fills threadgroup memory
- Barrier-free levels within shared memory (threadgroup_barrier only)
- Variable subtree sizes for upper tree levels

### 4.5 CPU-GPU work partitioning
- Small workloads (< dispatch overhead threshold) run on CPU
- NEON-optimized CPU paths: BabyBear NTT (5.8×), Blake3 (97×), Keccak (11×), batch field ops (60-78×)
- Auto-selection based on problem size

---

## 5. M-series GPU Scheduling Pathology (1.5 pages)

### 5.1 Discovery
- MSM performance varies non-monotonically with bucket count
- 10-30× slowdowns at specific counts: ~16K, ~32K, ~128K buckets
- Reproducible across M3 Pro, observed independently by other researchers

### 5.2 Characterization
- Triggered by specific threadgroup counts interacting with GPU scheduler
- Not related to occupancy, register pressure, or memory bandwidth
- Hypothesis: internal work distribution granularity in M-series command processor
- Table: MSM timing vs bucket count (smooth sweep from 1K to 256K)

### 5.3 Workarounds
- Window size selection to avoid pathological bucket counts
- Signed-digit recoding to halve effective bucket count
- GLV endomorphism interaction: doubles point count, can push into pathological range

### 5.4 Implications
- Affects any GPU algorithm with data-dependent threadgroup counts on Apple Silicon
- Radix sort, histogram operations, and other scatter-based algorithms also affected
- Recommendation: benchmark at target bucket/threadgroup counts before deploying

---

## 6. Evaluation (3 pages)

### 6.1 Experimental setup
- Hardware: Apple M3 Pro (14 GPU cores, 18GB unified memory)
- Software: macOS 15, Metal 3, Swift 6
- Methodology: median of 5 runs, warmup, thermal management
- Baseline: CPU implementations (Swift, C with NEON, arkworks where comparable)

### 6.2 Primitive-level benchmarks

#### Table: GPU speedup over CPU by primitive
| Primitive | Size | GPU time | CPU time | Speedup |
|---|---|---|---|---|
| MSM BN254 | 2^18 | 44ms | — | 1300× over naive |
| NTT BN254 | 2^16 | 0.7ms | 34ms | 47× |
| NTT M31 Circle | 2^20 | 4ms | — | 7× vs BN254 NTT |
| Poseidon2 Merkle | 2^16 | 17ms | — | — |
| Keccak Merkle | 2^20 | 9.5ms | 82ms | 9× |
| FRI fold | 2^22 | 6.5ms | 2100ms | 312× |
| Witness gen | 2^20 | 10ms | — | 7800× |
| Constraint eval | 2^14 | — | — | 2400× |

#### Table: Field size impact on NTT throughput
| Field | Element size | 2^20 NTT | Relative |
|---|---|---|---|
| BabyBear (31-bit) | 4 bytes | ~1ms | 1.0× |
| Mersenne31 (31-bit) | 4 bytes | 4ms | ~4× |
| Goldilocks (64-bit) | 8 bytes | ~8ms | ~8× |
| BN254 Fr (256-bit) | 32 bytes | ~28ms | ~28× |

### 6.3 End-to-end protocol benchmarks
- FRI commit+query+verify at 2^14, 2^18
- KZG commit+open+verify
- Sumcheck protocol
- Full Lasso lookup proof

### 6.4 Comparison with published CUDA results
- Table: zkMetal vs Icicle/cuZK on comparable operations (normalized by GPU generation)
- Analysis: where Metal is competitive, where CUDA dominates, and why
- Key factors: dispatch overhead (Metal loses), unified memory (Metal wins for small transfers), 32-bit ALU (Metal loses on large fields, competitive on small fields)

### 6.5 Cross-field analysis
- BN254 (256-bit) vs BabyBear (31-bit) vs Mersenne31 (31-bit) on same GPU
- Quantifies the "small field advantage" on real hardware
- Insight: field size reduction is worth more than any single GPU optimization

---

## 7. Discussion (1 page)

### 7.1 When to use Metal for ZK
- Developer machines: local proving during development (no cloud GPU needed)
- Mobile/edge: iOS devices with A-series chips
- Small-to-medium circuits: where dispatch overhead is amortized
- Small fields (BabyBear, M31): where 32-bit ALU is not a disadvantage

### 7.2 Limitations
- 32-bit ALU: ~4× overhead for 256-bit field arithmetic vs 64-bit GPU
- Dispatch overhead: 500× higher than CUDA, limits fine-grained parallelism
- No multi-GPU: M3 Ultra's dual GPU dies not yet exploited
- Compiler opacity: cannot inspect or optimize at ISA level

### 7.3 Future work
- Full STARK prover over Mersenne31 (Circle STARK pipeline)
- Groth16 prover with BLS12-381 pairings
- WebGPU port (Metal is the WebGPU backend on Apple)
- Multi-GPU support for M3 Ultra/M4 Ultra
- Binary tower fields (Binius) — XOR-based arithmetic maps well to GPU

---

## 8. Conclusion (0.5 page)
- First comprehensive study of ZK proving on Apple Silicon GPU
- Demonstrates viability: 100-7800× speedups across primitives
- Identifies architectural challenges and solutions specific to Metal
- Reports novel hardware scheduling pathology
- Open-source contribution: 22+ primitives with Rust FFI
- Small field results (M31 Circle NTT) suggest future ZK systems should co-design field choice with target hardware

---

## Appendix

### A. Montgomery constant derivation
- R, R², p_inv computation for each field

### B. Complete benchmark tables
- All primitives, all sizes, GPU and CPU

### C. Metal shader code excerpts
- CIOS multiplication kernel
- Fused Merkle subtree kernel
- Circle NTT butterfly

### D. Scheduling pathology raw data
- Full sweep of bucket counts vs MSM timing

---

## Figures needed
1. System architecture diagram
2. MSM GPU pipeline (decompose → sort → accumulate → reduce)
3. Fused Merkle subtree shared memory layout
4. Scheduling pathology plot (bucket count vs time, showing cliffs)
5. Field size vs NTT throughput (log-log plot)
6. Dispatch overhead breakdown (FRI commit phases)
7. Comparison bar chart: zkMetal vs CUDA published numbers
