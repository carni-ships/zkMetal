# Metal GPU Profiling and Tuning Guide for ZK Workloads

Practical guide for optimizing zero-knowledge cryptography on Apple Silicon GPUs,
based on lessons learned building zkMetal on the M3 Pro.

---

## 1. M3 Pro GPU Architecture for ZK

### ALU: 32-bit Only

Apple GPU cores have **no native 64-bit integer multiply**. A single `uint64 * uint64`
compiles to multiple 32-bit operations. This is the single most important constraint
for ZK on Metal: 256-bit field arithmetic that runs in 4 limbs on x86 (via `__uint128_t`)
requires **8x uint32 limbs** on GPU.

A Montgomery CIOS multiplication on BN254 Fr (256-bit) performs ~128 `uint32 * uint32`
multiply-accumulate operations per field mul. Compare: CUDA on RTX 3090 Ti uses native
64-bit `mul.hi`/`mad.lo.cc` and completes the same operation in ~16 multiplies.

**Implication**: GPU wins on ZK not through per-thread throughput but through massive
parallelism. The crossover point where GPU beats an optimized C CPU path is typically
around 2^14 points for MSM and 2^14 elements for NTT.

### Threadgroup Shared Memory: 32 KB

Each threadgroup gets 32 KB of shared memory (`threadgroup` address space). For BN254 Fr
(32 bytes per element), this limits shared-memory working sets to **1024 field elements**.

This directly constrains the fused NTT kernel: `ntt_butterfly_fused` can process at most
10 butterfly stages locally (2^10 = 1024 elements) before requiring device-memory
round-trips for remaining stages.

```metal
threadgroup Fr shared[1024]; // max 1024 Fr elements = 32KB
```

For smaller fields this constraint relaxes: BabyBear (4 bytes) fits 8192 elements,
Goldilocks (8 bytes) fits 4096 elements.

### SIMD Width: 32 Threads

Apple GPUs execute in SIMD groups of 32 threads (analogous to CUDA warps). All 32 threads
execute the same instruction; divergent branches serialize. This matters critically for:

- **MSM bucket reduce**: Buckets with vastly different point counts cause threads in the
  same SIMD group to idle while the longest-running thread finishes its loop.
- **Variable-length loops**: Any kernel where loop count varies per thread risks
  SIMD divergence penalties.

The `simd_shuffle_down` intrinsic enables register-level communication within a SIMD group
without shared memory, used for tree reductions:

```metal
for (uint off = 16; off > 0; off >>= 1) {
    PointProjective other = simd_shuffle_down_point(acc, off);
    if (lid < off) acc = point_add(acc, other);
}
```

### Memory Bandwidth: ~150 GB/s Unified

M3 Pro unified memory delivers ~150 GB/s. The CPU and GPU share the same physical memory,
so there is **zero copy overhead** for buffer sharing -- `MTLBuffer` contents are directly
accessible from both sides. This makes GPU/CPU hybrid strategies viable: decide at runtime
whether to use GPU or CPU based on workload size.

For bandwidth-bound kernels, the theoretical floor is:
- NTT 2^20 BN254: `2^20 * 32B * 2 (read+write) = 67 MB` -> 0.45 ms at 150 GB/s
- Sumcheck round: `2^20 * 32B = 33 MB` -> 0.22 ms at 150 GB/s

In practice, actual NTT 2^20 takes 6.1 ms (compute-bound, not bandwidth-bound).

---

## 2. Key Optimization Patterns

### 2.1 Montgomery CIOS on 32-bit GPU ALU

All 256-bit field arithmetic uses Coarsely Integrated Operand Scanning (CIOS) Montgomery
multiplication with 8x uint32 limbs. The inner loop structure:

```metal
Fr fr_mul(Fr a, Fr b) {
    uint t[10];
    for (int i = 0; i < 8; i++) {
        // Multiply-accumulate: a[i] * b[j] for all j
        for (int j = 0; j < 8; j++) {
            carry += ulong(t[j]) + ulong(a.v[i]) * ulong(b.v[j]);
            t[j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        // Montgomery reduction step
        uint m = t[0] * FR_INV;
        // Reduce: m * P[j] for all j
        ...
    }
}
```

Key details:
- `#pragma unroll` on inner loops is critical -- without it the compiler may spill to
  device memory
- The `ulong` intermediates (`uint64`) are synthesized from 32-bit ops but the compiler
  handles this efficiently for multiply-accumulate patterns
- **Dedicated squaring** (`fr_sqr`) exploits `a*a` symmetry: 36 cross-term multiplies
  vs 64 for general multiplication, saving ~22% of ALU work
- **Lazy addition** (`fr_add_lazy`) skips modular reduction when the result feeds directly
  into `fr_mul`, which handles unreduced inputs up to 2^256

### 2.2 Four-Step FFT for NTT

The NTT uses a hybrid approach:

1. **Fused sub-block kernel** (`ntt_butterfly_fused`): First log2(1024) = 10 stages
   execute entirely in threadgroup shared memory. One load from device memory, all
   butterfly passes in shared memory with `threadgroup_barrier`, one store back.

2. **Per-stage kernels** (`ntt_butterfly`, `ntt_butterfly_radix4`): Remaining stages
   dispatch one kernel per stage (or per 2 stages for radix-4), reading/writing device
   memory each time.

3. **Fused bitrev + butterfly** (`ntt_butterfly_fused_bitrev`): Eliminates the separate
   bit-reversal permutation pass by loading with bit-reversed indices directly into
   shared memory.

The four-step structure (sub-blocks in shared memory, then global stages) reduces device
memory traffic by ~10x for the first 10 stages. For BabyBear (32-bit field), shared memory
fits 8192 elements, allowing 13 fused stages.

**Radix-4 butterfly**: Processes 2 stages at once, requiring 6 field multiplies per
4-element group instead of 4 multiplies across 2 separate radix-2 stages. The
reduced dispatch count saves ~0.5 ms of command buffer overhead per eliminated stage.

### 2.3 Pippenger MSM with Signed-Digit + GLV

The MSM pipeline has four GPU phases in a merged command buffer:

1. **GLV decomposition** (GPU kernel): Splits each 256-bit scalar `k` into two ~128-bit
   half-scalars `(k1, k2)` where `k = k1 + k2*lambda mod r`. Halves the effective
   scalar bit-width, halving bucket count.

2. **Signed-digit recoding + GPU counting sort**: Recodes digits to the range
   `[-(2^w-1), 2^w-1]` with carry propagation, eliminating bucket 2^w. GPU radix sort
   groups points by bucket ID for coalesced memory access. Three kernels: histogram,
   prefix sum, scatter.

3. **Bucket reduce** (`msm_reduce_cooperative`): One SIMD group (32 threads) per bucket.
   Each thread accumulates points at stride 32, then SIMD shuffle tree reduction merges
   partial sums. Uses `point_add_mixed_unsafe` (skips identity checks) since random
   point collision probability is ~10^-65.

4. **Bucket sum + Horner combine**: Weighted running sum per window segment, then
   Horner's method combines window results: `result = sum(window[i] * 2^(i*w))`.

**Count-sorted dispatch**: The `count_sorted_map` buffer reorders bucket processing so
that SIMD groups handle buckets with similar point counts, minimizing divergence. Bucket
index is packed as `(window << 16) | bucket` in a uint32.

### 2.4 Buffer Caching and Grow-Only Patterns

Metal buffer allocation (`device.makeBuffer`) is expensive (~0.1-0.5 ms per call).
zkMetal caches all GPU buffers and reuses them across operations:

- **SRS point buffer**: Allocated once at engine init, persists for all MSMs
- **Twiddle factor buffer**: Computed once per NTT size, cached for reuse
- **Grow-only scratch buffers**: Bucket arrays, sort temporaries, etc. are allocated
  at the high-water mark and never shrunk. New allocations only happen when a larger
  size is needed.
- **MTLBinaryArchive shader cache**: Compiled pipeline states are persisted to disk,
  eliminating ~50-200 ms of shader compilation on subsequent launches

For the Barretenberg prover integration, `DontZeroMemory` is used on polynomial allocations
where every element will be overwritten before reading:
```cpp
Polynomial(size, active_size, Polynomial::DontZeroMemory::FLAG)
```
This saved ~192 ms on sumcheck partial evaluation (60+ polynomials * 524K elements).

### 2.5 Fused Kernels

Kernel fusion eliminates device-memory round-trips between logically sequential operations:

- **NTT + constraint evaluation** (`fused_ntt_fib_constraint`): Performs NTT on trace
  columns in shared memory, then immediately evaluates AIR constraints on the transformed
  data without writing intermediate NTT results to device memory. Two columns * 1024
  elements fit in 32 KB shared memory.

- **Hash + Merkle subtree** (Keccak, Blake3): Each threadgroup hashes a batch of leaves
  and builds several levels of the Merkle tree locally before writing intermediate nodes.
  Reduces global memory writes by a factor of the subtree depth.

- **GLV + sort pipeline**: Six encoders in a single command buffer: GLV decompose, GLV
  endomorphism, blit zero, histogram, prefix sum, scatter. Eliminates 5 GPU round-trips.

- **Radix-4 butterfly**: Fuses 2 NTT stages into one kernel dispatch, halving command
  buffer submission overhead for the non-fused NTT stages.

---

## 3. Common Pitfalls

### 3.1 GPU Scheduling Pathology at Certain Bucket Counts

The M3 Pro GPU exhibits **catastrophic 10-30x slowdowns** at specific Pippenger window
sizes:

| Window | Buckets | Behavior |
|--------|---------|----------|
| w=13   | 8K      | Fast     |
| w=14   | 16K     | **10-30x slow** |
| w=15   | 32K     | **10-30x slow** |
| w=16   | 64K     | Fast     |
| w=17   | 128K    | **10-30x slow** |
| w=18   | 256K    | 3-5x slow |

This is **not fixable with count-sorting** alone. Tested: w=15 with SIMD-uniform
count-sorted dispatch still takes ~309 ms vs ~50 ms at w=16. The pathology appears to
be hardware-level register spilling behavior at certain bucket counts and is not purely
a thread-divergence issue.

**Mitigation**: Always use w=16 for n > 32K points. Add per-window imbalance bailout:
if any single window has >10% of points concentrated in one bucket (common with
GLV-decomposed structured polynomials), fall back to CPU for that MSM.

### 3.2 ICICLE-Metal License Server Overhead

ICICLE-Metal v3.8.0 adds ~600 ms constant overhead per MSM/NTT call due to a license
server check. This makes it appear ~30-90x slower than zkMetal on NTT benchmarks and
~10-20x slower on MSM. When comparing against ICICLE, subtract ~600 ms from their
reported times to estimate their actual kernel performance, or measure with license
caching if available.

### 3.3 Register Spilling with Keccak

Keccak-f[1600] has a 25-element state of 64-bit lanes. On Apple GPU with 32-bit ALU,
this becomes 50 uint32 registers just for the state, plus temporaries for theta/rho/pi/chi
steps. Total register pressure can exceed the GPU's per-thread register file.

**Solution**: Bit-interleaved representation. Each 64-bit lane is split into even-bits
and odd-bits, each stored in a 32-bit word. This:
- Keeps all registers at native 32-bit width
- Simplifies 64-bit rotations (which are expensive as 2x shift + OR on 32-bit ALU)
  into 32-bit rotations that map to single-cycle barrel shifter ops
- Reduces register pressure since the compiler can better schedule 32-bit values

```metal
// Standard: 25 ulongs = 50 uint32 registers + rotation temporaries
// Bit-interleaved: 50 uint32 registers but rotations are single-cycle
constant uint2 KECCAK_RC_IL[24] = { ... }; // Pre-split round constants
```

### 3.4 Strided Memory Access in NTT Column Phase

The four-step FFT's global stages access memory with power-of-2 strides, which can
cause bank conflicts and poor cache utilization. For stage `s`, the stride is `2^s`,
meaning at stage 18 of a 2^20 NTT, two butterfly elements are 256K elements apart
(8 MB for BN254).

This is the primary reason NTT 2^22 BN254 takes 26 ms vs the 2.9 ms theoretical
compute floor -- the large stages are bandwidth-bound with poor spatial locality.

**Partial mitigation**: The fused sub-block kernel handles the first 10 stages (small
strides) in shared memory. Only stages 10+ hit device memory with large strides.
Radix-4 butterflies reduce the number of large-stride passes by half.

### 3.5 Lazy Montgomery Reduction at GPU Boundaries

Barretenberg (and some other libraries) store field elements in lazy-reduced Montgomery
form: values in [0, 2p) rather than [0, p). GPU `fp_add`/`fp_sub` assume fully reduced
inputs. If both inputs are in [p, 2p), the single conditional subtraction is insufficient.

**Fix**: Apply `fp_reduce()` at GPU entry points (e.g., `point_from_affine`,
`glv_endomorphism` kernel). GPU-internal values remain in [0, p) after correct
reduction. GPU outputs are fully reduced and safe for host consumption.

```metal
Fp fp_reduce(Fp a) {
    uint borrow;
    Fp reduced = fp_sub_raw(a, fp_modulus(), borrow);
    return borrow ? a : reduced;
}
```

This was the root cause of ~10% MSM error rate before being identified. Any integration
with external libraries must normalize field elements at GPU boundaries.

### 3.6 Degenerate Scalar Distributions

Mock benchmarks and certain polynomial evaluations (e.g., z_perm in Barretenberg) can
produce nearly all-zero scalar vectors. The GPU sort histogram will show >90% of points
in bucket 0 (identity), which is both wasteful and can crash the counting sort.

**Fix**: Check histogram sparsity before GPU sort. If fewer than 10% of expected points
are non-zero, fall back to CPU.

---

## 4. Profiling with Metal System Trace

### Instruments Setup

1. Open **Instruments.app** (Xcode -> Open Developer Tool -> Instruments)
2. Choose the **Metal System Trace** template
3. Target your process (e.g., `swift run -c release zkbench msm`)
4. Record for the duration of the benchmark

### What to Look For

**GPU Timeline**: Shows command buffer submissions, encoder boundaries, and kernel
execution. Look for:
- Gaps between kernel executions (command buffer scheduling overhead, ~0.5 ms each)
- Long-running single threads (SIMD divergence -- one lane holding up 31 others)
- Memory transfer stalls (visible as GPU idle periods between compute)

**GPU Counters** (Metal HUD or `MTLCounterSampleBuffer`):
- **ALU Utilization**: Should be >80% for compute-bound kernels (field arithmetic).
  Low ALU utilization on NTT suggests bandwidth bottleneck.
- **Occupancy**: Threads in flight vs maximum. Register-heavy kernels (Keccak, field mul)
  may limit occupancy. Target >50%.
- **Memory Read/Write Bandwidth**: Compare against the 150 GB/s theoretical maximum.
  Bandwidth-bound kernels (sumcheck, large-stride NTT) should approach this.
- **Register Spills**: Any spills to device memory are catastrophic for performance.
  If the Metal compiler reports high register usage, reduce per-thread state or split
  the kernel.

### Command-Line Profiling

```bash
# Quick GPU timing with Metal timestamps
export METAL_DEVICE_WRAPPER_TYPE=1

# Per-kernel timing via command buffer completed handlers
# (built into zkMetal's benchmark harness)
swift run -c release zkbench msm --verbose

# System-level GPU metrics
sudo powermetrics --samplers gpu_power -i 1000
```

### Targeted Kernel Analysis

To profile a specific kernel:

1. Add `MTLCaptureManager` bracketing around the dispatch of interest
2. Use `gpu_time` from `MTLCommandBuffer.gpuStartTime` / `gpuEndTime`
3. Compare `gpuEndTime - gpuStartTime` (actual GPU execution) against
   wall-clock time (includes scheduling and CPU overhead)

A large gap between GPU time and wall time indicates command buffer scheduling overhead.
Merging multiple encoders into a single command buffer eliminates this. In zkMetal, the
MSM pipeline went from 5 separate command buffers to 1 merged buffer, saving ~2 ms of
scheduling overhead.

---

## 5. Performance Comparison Methodology

### Benchmarking Rules

1. **Warm-up runs**: Always discard the first 1-2 iterations. Metal pipeline state
   compilation, shader cache population, and GPU buffer first-allocation are one-time
   costs that should not be included in steady-state measurements.

2. **Minimum of N runs**: Report the **minimum** of 5+ runs, not the average. GPU
   scheduling jitter, thermal throttling, and OS background tasks add noise. The minimum
   represents the true capability; the variance represents system interference.

3. **Apples-to-apples**: When comparing against other libraries:
   - Use the **same point/scalar data** (random, not structured)
   - Measure **end-to-end** including host-device transfer (for non-unified-memory systems)
   - Note whether the comparison includes one-time setup (SRS loading, pipeline compile)
   - For ICICLE-Metal, note the ~600 ms license overhead separately

4. **Report hardware**: Always state: chip (M3 Pro), core count (6P+6E), memory (18 GB),
   macOS version. GPU performance varies significantly across M1/M2/M3/M4 and
   base/Pro/Max/Ultra tiers due to different GPU core counts and memory bandwidth.

### What to Report

| Metric | How to Measure |
|--------|----------------|
| Wall clock | `CFAbsoluteTimeGetCurrent()` or `mach_absolute_time()` bracketing |
| GPU time | `MTLCommandBuffer.gpuEndTime - gpuStartTime` |
| Throughput | Elements/sec or ops/sec for the specific primitive |
| vs. theoretical | Current / (total_ops / peak_flops) or (total_bytes / peak_bw) |

### Theoretical Floor Calculation

For any primitive, compute both the compute floor and the bandwidth floor. The actual
floor is `max(compute_floor, bandwidth_floor)`:

```
Compute floor:
  ops_per_element = (e.g., 128 uint32 muls for one BN254 fr_mul)
  total_ops = N * ops_per_element * ops_per_kernel_invocation
  compute_time = total_ops / 3.6 TFLOPS

Bandwidth floor:
  bytes_transferred = N * element_size * (reads + writes)
  bandwidth_time = bytes_transferred / 150 GB/s

Dispatch floor:
  dispatch_time = num_kernel_dispatches * 0.5 ms
```

Example for MSM 2^18 BN254:
- Compute: 262K points * ~1000 field muls/point * 128 ops/mul = 33.5 Gops -> ~9 ms
- Bandwidth: 262K * 96B (affine point) = 25 MB read -> 0.17 ms
- Dispatch: 4 phases * 0.5 ms = 2 ms
- Floor: ~9 ms (compute-bound). Actual: 45 ms. Headroom: ~5x.

### Near-Optimal Primitives (< 1.5x of Floor)

These are within noise of hardware limits and need algorithmic breakthroughs for further
gains:

- BabyBear NTT 2^24: 2.0 ms (floor ~1.7 ms) -- bandwidth-bound, 32-bit native
- Circle NTT 2^20: 4.0 ms (floor ~3 ms) -- single-word arithmetic
- HyperNova fold: 0.09 ms (floor ~0.07 ms) -- already 40x optimized
- KZG commit 2^10: 0.7 ms (floor ~0.5 ms) -- C Horner + fused eval/div

### GPU vs CPU Crossover Points

| Primitive | GPU Wins Above | Reason |
|-----------|---------------|--------|
| MSM BN254 | 2^14 (16K points) | Dispatch overhead dominates below |
| NTT BN254 | 2^12 (4K elements) | Shared-memory fused path efficient |
| Keccak batch | 2^12 (4K hashes) | Bit-interleaved 32-bit very efficient |
| Poseidon2 batch | 2^12 (4K hashes) | High arithmetic intensity |
| Blake3 batch | 2^14 (16K hashes) | Lower arithmetic intensity |

Below these thresholds, use the C CPU path (NEON SIMD, `__uint128_t` Montgomery).
zkMetal automatically selects GPU or CPU based on input size.

---

## Appendix: Quick Reference

### Metal Shader Optimization Checklist

- [ ] Use `#pragma unroll` on all inner loops with known bounds
- [ ] Prefer `uint32` over `uint64` for all intermediate computations
- [ ] Keep threadgroup shared memory usage under 32 KB
- [ ] Use `constant` address space for uniform data (round constants, twiddles broadcast)
- [ ] Use `simd_shuffle_down` for intra-SIMD reductions instead of shared memory
- [ ] Use `point_add_mixed_unsafe` when identity probability is negligible
- [ ] Merge sequential kernels into a single command buffer with multiple encoders
- [ ] Cache all `MTLBuffer` objects and grow-only
- [ ] Add `fp_reduce()` at GPU entry points when interfacing with external libraries
- [ ] Profile with Metal System Trace before and after changes

### File Map

```
Sources/Shaders/fields/     Field arithmetic (CIOS Montgomery, lazy add, SOS squaring)
Sources/Shaders/geometry/   Elliptic curve ops (affine, projective, mixed add)
Sources/Shaders/msm/        Pippenger kernels (reduce, bucket_sum, GLV, sort)
Sources/Shaders/ntt/         NTT kernels (butterfly, fused sub-block, radix-4)
Sources/Shaders/hash/       Hash functions (Poseidon2, Keccak bit-interleaved, Blake3)
Sources/Shaders/constraint/ Fused NTT+constraint evaluation
Sources/Shaders/fri/        FRI folding kernels
Sources/Shaders/sumcheck/   Sumcheck round reduction
Sources/NeonFieldOps/       C/ARM64 CPU paths (__uint128_t CIOS, NEON SIMD)
Sources/zkMetal/            Swift engine (buffer caching, dispatch logic, auto-tuning)
```
