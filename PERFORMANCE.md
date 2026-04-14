# Performance Benchmarks

All benchmarks on Apple M3 Pro (6P+6E cores). Run `swift run -c release zkbench all` to reproduce.

## MSM (BN254 G1)

| Points | C Pippenger CPU | GPU (Metal) |
|--------|-----------------|-------------|
| 2^8 | 350ms | 0.8ms |
| 2^10 | 1.4s | 2.5ms |
| 2^12 | 5.6s | 21.0ms |
| 2^14 | 23.4s | 33.6ms |
| 2^16 | -- | 46.3ms |
| 2^17 | -- | 54.8ms |
| 2^18 | -- | 72.7ms |
| 2^20 | -- | 175.3ms |

**Comparison to other implementations (BN254 MSM):**

| Points | zkMetal (M3 Pro) | ICICLE-Metal (M3 Pro) | ICICLE CPU (M3 Pro) | MoPro v2 (M3 Air) | Arkworks CPU (M3 Air) | ICICLE CUDA |
|--------|---------|-------------|-----------|-------------|-----------|-----------|
| 2^16 | **31ms** | 1,083ms | 114ms | 253ms | 69ms | ~9ms |
| 2^18 | **53ms** | 1,475ms | 556ms | 678ms | 266ms | -- |
| 2^20 | **137ms** | 2,590ms | 2,349ms | 1,702ms | 592ms | -- |

## NTT

**Multi-field NTT comparison (GPU):**

| Size | BN254 Fr (256-bit) | BLS12-377 Fr (253-bit) | Goldilocks (64-bit) | BabyBear (31-bit) |
|------|-------------------|----------------------|--------------------|--------------------|
| 2^16 | 0.69ms | 1.4ms | 0.15ms | 0.26ms |
| 2^18 | 2.9ms | 2.1ms | 0.21ms | 0.54ms |
| 2^20 | 10.8ms | 5.8ms | 0.70ms | 0.79ms |
| 2^22 | 28ms | 25ms | 4.3ms | 3.0ms |
| 2^24 | 113ms | 110ms | 3.1ms | 2.3ms |

**BN254 NTT (GPU):**

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^14 | 0.71ms | 5.8ms | 8x |
| 2^16 | 0.85ms | 26.6ms | 31x |
| 2^18 | 1.72ms | 119.8ms | 70x |
| 2^20 | 6.06ms | 528ms | 87x |
| 2^22 | 26.0ms | 2262ms | 87x |
| 2^24 | 110.9ms | 9861ms | 89x |

**Comparison to ICICLE-Metal v3.8 NTT (M3 Pro):**

| Size | zkMetal BN254 | ICICLE BN254 | zkMetal BabyBear | ICICLE BabyBear |
|------|------------|-------------|----------------|---------------- |
| 2^16 | **0.85ms** | 89ms | **0.28ms** | 86ms |
| 2^18 | **1.72ms** | 108ms | **0.26ms** | 92ms |
| 2^20 | **6.06ms** | 194ms | **0.66ms** | 108ms |
| 2^22 | **26.1ms** | 915ms | **2.67ms** | 181ms |
| 2^24 | **110.9ms** | 3,892ms | **1.33ms** | 709ms |

## Hashing

| Primitive | Batch Size | Vanilla CPU | Optimized CPU | GPU (Metal) | GPU vs Opt CPU |
|-----------|-----------|-------------|--------------|-------------|----------------|
| Poseidon2 | 2^12 | 523ms | 19ms (C CIOS) | 2.3ms | **8x** |
| Poseidon2 | 2^14 | 2.0s | 75ms (C CIOS) | 2.3ms | **33x** |
| Poseidon2 | 2^16 | 8.0s | 302ms (C CIOS) | 8.5ms | **36x** |
| Pasta Poseidon | 2^16 | 16.1s | 1.1s (C CIOS) | 117ms (~0.6M hash/s) | **9x** |
| Pasta Poseidon | 2^18 | -- | 4.2s (C CIOS) | 212ms (~0.6M hash/s) | **20x** |
| Keccak-256 | 2^14 | 100ms | 23ms (parallel) | 0.20ms | **500x** |
| Keccak-256 | 2^16 | 387ms | 89ms (parallel) | 0.45ms | **860x** |
| Keccak-256 | 2^18 | 1.6s | 360ms (parallel) | 1.4ms | **1143x** |

## Merkle Trees

| Backend | Leaves | GPU | CPU | Speedup |
|---------|--------|-----|-----|---------|
| Poseidon2 | 2^10 | 7.3ms | 6ms | **1x** |
| Poseidon2 | 2^12 | 8.7ms | 23ms | **3x** |
| Poseidon2 | 2^14 | 10ms | 91ms | **9x** |
| Poseidon2 | 2^16 | 21ms | 364ms | **17x** |
| Poseidon2 | 2^18 | 45ms | 1.4s | **32x** |
| Poseidon2 | 2^20 | 129ms | -- | -- |
| Keccak-256 | 2^12 | 0.37ms | 44ms | **119x** |
| Keccak-256 | 2^14 | 0.51ms | 155ms | **304x** |
| Keccak-256 | 2^16 | 1.4ms | 783ms | **559x** |
| Keccak-256 | 2^18 | 4.5ms | 3.0s | **667x** |
| Keccak-256 | 2^20 | 13ms | -- | -- |
| Blake3 | 2^12 | 0.72ms | 4ms | **6x** |
| Blake3 | 2^14 | 0.92ms | 16ms | **17x** |
| Blake3 | 2^16 | 1.3ms | 101ms | **78x** |
| Blake3 | 2^18 | 3.9ms | 345ms | **88x** |
| Blake3 | 2^20 | 12ms | -- | -- |

## FRI Folding (BN254 Fr)

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^14 | 0.22ms | 8.9ms | **41x** |
| 2^16 | 0.35ms | 35ms | **99x** |
| 2^18 | 0.92ms | 137ms | **149x** |
| 2^20 | 1.96ms | 542ms | **276x** |
| 2^22 | 7.52ms | 2.2s | **295x** |

**FRI commit phase (fold + Merkle, full protocol):**

| Size | Fold-by-2 | Fold-by-4 | Fold-by-8 | 8/2 speedup |
|------|-----------|-----------|-----------|-------------|
| 2^15 | 68ms | 37ms | 20ms | **3.5x** |
| 2^16 | 81ms | 37ms | 33ms | **2.5x** |
| 2^18 | 137ms | 59ms | 36ms | **3.9x** |
| 2^20 | 392ms | 132ms | 121ms | **3.2x** |

## Sumcheck (BN254 Fr)

| Variables | GPU | C Kernel | Vanilla | Best vs Vanilla |
|-----------|-----|----------|---------|----------------|
| 2^14 | 16.0ms | 0.50ms | 0.3ms | C 1x |
| 2^16 | 16.0ms | 1.04ms | 1.3ms | C 1x |
| 2^18 | 16.0ms | 2.64ms | 3.6ms | C 1x |
| 2^20 | 24.0ms | 9.55ms | 14.0ms | C 1x |
| 2^22 | 84.6ms | 33.8ms | 84.9ms | C 3x |

## Polynomial Ops (BN254 Fr)

| Operation | Size | Vanilla CPU | GPU (Metal) | GPU vs Vanilla |
|-----------|------|-------------|-------------|----------------|
| Multiply (NTT) | deg 2^10 | 57ms | 1.7ms | **34x** |
| Multiply (NTT) | deg 2^12 | 218ms | 2.0ms | **109x** |
| Multiply (NTT) | deg 2^14 | 1.1s | 3.3ms | **328x** |
| Multiply (NTT) | deg 2^16 | 2.4s | 7.7ms | **319x** |

## GPU Additive FFT (GF(2^8))

| Size | Elements | Time | Throughput | Notes |
|------|----------|------|------------|-------|
| 2^16 | 65,536 | ~8ms | ~8 M elem/s | |
| 2^18 | 262,144 | ~9ms | ~30 M elem/s | |
| 2^20 | 1,048,576 | ~11ms | ~95 M elem/s | |
| 2^22 | 4,194,304 | ~13ms | ~320 M elem/s | |

**Optimization in progress**: Precomputed GF(2^8) multiplication LUT (256-entry) could reduce
multiply from ~176 primitive ops to 1 table lookup. Target: 3-6x speedup (13ms → 2-4ms).
Combined with SIMD shuffle: 6-10x total speedup potential.

## KZG Commitments (BN254 G1)

| Operation | Size | Vanilla CPU | GPU (Metal) | GPU vs Vanilla |
|-----------|------|-------------|-------------|----------------|
| Commit | deg 2^8 | 261ms | 0.3ms | **813x** |
| Commit | deg 2^10 | 1.0s | 0.7ms | **1396x** |
| Open (eval + witness) | deg 2^8 | 381ms | 0.9ms | **446x** |
| Open (eval + witness) | deg 2^10 | 1.6s | 2.5ms | **669x** |

## Batch KZG (BN254 G1)

| N Polys | Deg | N Individual Opens | 1 Batch Open | Speedup |
|---------|-----|-------------------|--------------|---------|
| 4 | 256 | 14.4ms | 9.8ms | **1.5x** |
| 8 | 256 | 24.5ms | 13.0ms | **1.9x** |
| 16 | 256 | 47.3ms | 21.5ms | **2.2x** |
| 32 | 256 | 75.2ms | 34.5ms | **2.2x** |

## Basefold PCS (BN254 Fr)

| Size | Commit | Open | Verify | Total |
|------|--------|------|--------|-------|
| 2^10 | 7.4ms | 31ms | 0.00ms | 38ms |
| 2^14 | 10ms | 64ms | 0.00ms | 74ms |
| 2^18 | 46ms | 138ms | 0.00ms | 184ms |

## Circle STARK (Mersenne31)

| Trace Size | Prove | Verify | Proof Size |
|-----------|-------|--------|------------|
| 2^8 | 5.8ms | 9ms | 40 KB |
| 2^10 | 4.9ms | 14ms | 54 KB |
| 2^12 | 7.6ms | 16ms | 70 KB |
| 2^14 | 17ms | 20ms | 89 KB |

## Plonk (BN254, KZG)

| Gates | Setup | Prove | Verify |
|-------|-------|-------|--------|
| 16 | 8ms | 3ms | 2ms |
| 64 | 14ms | 9ms | 2ms |
| 256 | 15ms | 15ms | 2ms |
| 1024 | 31ms | 50ms | 2ms |

## Groth16 (BN254)

| Constraints | Setup | Prove | Verify |
|-------------|-------|-------|--------|
| 8 | 107ms | 11ms | 4ms |
| 64 | 568ms | 12ms | 4ms |
| 256 | 2.3s | 14ms | 4ms |

## GKR (BN254 Fr, Layered Circuits)

| Circuit | Prove | Verify |
|---------|-------|--------|
| 2^4 width, d=4 | 0.09ms | 0.06ms |
| 2^5 width, d=4 | 0.15ms | 0.08ms |
| 2^6 width, d=4 | 0.29ms | 0.13ms |
| 2^6 width, d=8 | 0.57ms | 0.25ms |
| 2^8 width, d=4 | 1.24ms | 0.38ms |
| 2^8 width, d=8 | 2.72ms | 0.71ms |
| 2^10 width, d=4 | 6.38ms | 1.38ms |

## GPU Radix Sort

| Size | GPU | CPU | Speedup |
|------|-----|-----|---------|
| 2^16 | 0.7ms | 2.9ms | **4x** |
| 2^18 | 1.3ms | 13ms | **10x** |
| 2^20 | 2.1ms | 59ms | **28x** |
| 2^22 | 6.4ms | 278ms | **43x** |

## Other Curve MSM

| Points | BN254 GPU | BLS12-377 GPU | secp256k1 GPU | secp256k1 C Pip | Pallas GPU | Vesta GPU | Grumpkin GPU |
|--------|-----------|---------------|---------------|-----------------|------------|-----------|--------------|
| 2^8 | 1.1ms | 9ms | 0.8ms | 0.8ms | 5.3ms | 4.9ms | 3.3ms |
| 2^10 | 3.0ms | 35ms | 2.4ms | 2.4ms | 12ms | 10ms | -- |
| 2^12 | 8.1ms | 23ms | 56ms | 6.8ms | 17ms | 17ms | 20ms |
| 2^14 | 22ms | 29ms | 42ms | 21ms | 20ms | 20ms | 258ms |
| 2^16 | 27ms | 55ms | 72ms | 64ms | 39ms | 39ms | 48ms |
| 2^18 | 45ms | 119ms | 200ms | 221ms | 66ms | 65ms | -- |

## CPU Optimizations

| Primitive | Size | Vanilla | Optimized | Speedup | Notes |
|-----------|------|---------|-----------|---------|-------|
| BN254 NTT (C CIOS) | 2^20 | 7.5s | 247ms | **30x** | `__uint128_t` unrolled Montgomery |
| BabyBear NTT (NEON) | 2^22 | 202ms | 37ms | **5.4x** | 4-wide SIMD Montgomery |
| Goldilocks NTT (C) | 2^20 | 53ms | 25ms | **2.1x** | `__uint128_t` pipelining |
| BN254 Fr mul (C CIOS) | single | 2500ns | 16ns | **156x** | Zero-copy Swift↔C bridge |
| BN254 Fr add (C) | single | ~50ns | 4.5ns | **11x** | Branchless modular add |
| BN254 batch mul (C) | 100K | 250ms | 1.3ms | **192x** | 13.4 ns/op CIOS |
| BN254 batch inverse (C) | 100K | 1.0s | 1.6ms | **625x** | Montgomery's trick |
| BN254 batch axpy (C) | 100K | 270ms | 1.5ms | **180x** | Fused scalar*vec + accumulate |
| BN254 inner product (C) | 100K | 250ms | 1.3ms | **192x** | Dot product + vector_sum |
| BN254 fold_interleaved (C) | 2^18 | 1.3s | 5.2ms | **250x** | In-place fold |
| BN254 Horner eval (C) | deg 2^16 | 163ms | 1.0ms | **163x** | Prefetch + branchless |
| ECDSA batch 64 (CPU) | 64 sigs | -- | 1.7ms | **57x** | 0.03ms/sig |

## Supporting Primitives

| Primitive | Metric | Value |
|-----------|--------|-------|
| Transcript (Keccak) | 1K absorb+squeeze | 0.89ms (2.2M ops/s) |
| Transcript (Poseidon2) | 1K absorb+squeeze | 9.9ms (202K ops/s) |
| KZG proof size | -- | 138 B |
| IPA proof size (8 rounds) | -- | 1586 B |
| FRI commitment (2^14) | -- | 1025 KB |
| Blake3 batch GPU | 2^20 | 0.001 us/hash (**900x** vs CPU) |

## Advanced Protocols

| Primitive | Key Benchmark | Notes |
|-----------|---------------|-------|
| HyperNova fold | 0.09ms/fold (1000 steps) | Keccak256 transcript + C CIOS |
| Supernova fold | 0.67ms/fold (16-step) | Multi-circuit IVC with pc routing |
| Nova fold (100-fold) | 0.60ms/fold | GPU folds + sparse matvec |
| Nova fold (256c x 50) | 5.6ms/fold | GPU folds + sparse matvec |
| Basefold open 2^18 | 61ms | Fold-by-4 + pipelined Merkle |
| IPA prove n=256 | 11.8ms | C CIOS batch fold + Blake3 NEON |
| Verkle Trees (CPU) | 3.8ms proof | C CIOS Pedersen+IPA |
| Tensor compress 2^18 | 3.3ms compress | **460.7x** compression ratio |
| Lasso 2^18 | 29ms prove, 26ms verify | C-accelerated |
| BLS12-381 pairing | 1.0ms | **78×** faster |
| Schnorr BIP 340 | Verify 0.11ms | x-only pubkeys |

## Theoretical Floor Analysis

| Rank | Primitive | Current | Floor | Headroom | Status |
|------:|------------|--------:|------:|----------:|--------|
| 1 | GPU Additive FFT 2^22 | 13ms | ~0.5ms | ~26x | Optimization in progress (LUT) |
| 2 | MSM BN254 2^18 | 73ms | ~5ms | ~11x | |
| 3 | NTT BN254 2^22 | 26ms | ~3ms | ~9x | |
| 4 | secp256k1 MSM (GPU) | ~260ms | ~30ms | ~8x | GPU sort + bucket-interleaved in progress |
| 5 | FRI Fold 2^20 | 2.1ms | ~0.3ms | ~7x | |
| 6 | Nova fold (256c) | ~5.6ms | ~1ms | ~5x | GPU sparse matvec integrated |
