# External Baseline Benchmarks

How to reproduce comparison numbers for the paper. All benchmarks on the same Apple M3 Pro machine.

## arkworks (Rust CPU)

```bash
# Install
git clone https://github.com/arkworks-rs/algebra.git
cd algebra
rustup install nightly

# MSM benchmark (BN254 G1)
cargo +nightly bench --bench msm -- bn254

# NTT benchmark
cargo +nightly bench --bench fft -- bn254

# For optimized assembly backend:
RUSTFLAGS="-C target-feature=+bmi2,+adx" cargo +nightly bench --features asm
```

Also: https://github.com/weikengchen/ark-bench (Apple Silicon specific benchmarks)

**Published numbers (M3 Air):**
| Primitive | Size | arkworks | zkMetal GPU | Speedup |
|-----------|------|----------|-------------|---------|
| MSM BN254 G1 | 2^16 | 69ms | 27ms | **2.6x** |
| MSM BN254 G1 | 2^18 | 266ms | 45ms | **5.9x** |
| MSM BN254 G1 | 2^20 | 592ms | 119ms | **5.0x** |

## gnark (Go CPU)

```bash
# Install
go install github.com/consensys/gnark-crypto/...@latest

# Run benchmarks
cd $GOPATH/pkg/mod/github.com/consensys/gnark-crypto@*/ecc/bn254
go test -bench BenchmarkMultiExp -benchtime=10s -count=5

# For Groth16 end-to-end
cd $GOPATH/pkg/mod/github.com/consensys/gnark@*/backend/groth16
go test -bench BenchmarkProver -benchtime=10s -count=5
```

## ICICLE-Metal

```bash
# Install via Mopro or direct
git clone https://github.com/ingonyama-zk/icicle.git
cd icicle
# Follow Metal backend instructions

# Or use Mopro (wraps ICICLE)
git clone https://github.com/zkmopro/mopro.git
```

**Published numbers (M3 Pro):**
| Primitive | Size | ICICLE-Metal | zkMetal GPU | Speedup |
|-----------|------|-------------|-------------|---------|
| MSM BN254 | 2^16 | 1,083ms | 27ms | **40x** |
| MSM BN254 | 2^18 | 1,475ms | 45ms | **33x** |
| MSM BN254 | 2^20 | 2,590ms | 119ms | **22x** |
| NTT BN254 | 2^16 | 89ms | 0.76ms | **117x** |
| NTT BN254 | 2^20 | 194ms | 6.1ms | **32x** |
| NTT BabyBear | 2^16 | 86ms | 0.18ms | **478x** |
| NTT BabyBear | 2^24 | 709ms | 2.0ms | **355x** |

## zk-Bench (Comprehensive Framework)

Reference paper: IACR ePrint 2023/1503

```bash
git clone https://github.com/zkCollective/zk-Bench.git
cd zk-Bench
# Covers 9 libraries × 13 curves
# Includes MSM, NTT, field ops, Groth16, Plonk
```

## Reproducing zkMetal Numbers

```bash
cd /path/to/zkMetal

# All benchmarks
swift run -c release zkbench all

# Specific primitives
swift run -c release zkbench msm
swift run -c release zkbench ntt
swift run -c release zkbench hash
swift run -c release zkbench merkle
swift run -c release zkbench groth16
swift run -c release zkbench plonk
swift run -c release zkbench fri
swift run -c release zkbench sumcheck
swift run -c release zkbench circle_stark
```

## Comparison Summary (for paper tables)

### MSM BN254 G1 (Apple M3 Pro)

| Size | zkMetal | arkworks CPU | gnark CPU | ICICLE-Metal | ICICLE-CUDA (A100) |
|------|---------|-------------|-----------|-------------|-------------------|
| 2^16 | **27ms** | 69ms | TBD | 1,083ms | ~9ms |
| 2^18 | **45ms** | 266ms | TBD | 1,475ms | TBD |
| 2^20 | **119ms** | 592ms | TBD | 2,590ms | TBD |

### NTT BN254 Fr (Apple M3 Pro)

| Size | zkMetal | arkworks CPU | ICICLE-Metal |
|------|---------|-------------|-------------|
| 2^16 | **0.76ms** | TBD | 89ms |
| 2^20 | **6.1ms** | TBD | 194ms |

*TBD = need to run on our machine for apples-to-apples comparison*
