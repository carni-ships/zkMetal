# Jolt Integration Guide

This document covers integrating zkMetal into Jolt-based ZK provers for BN254 operations.

## Current Features

### GPU MSM (G1) - Implemented
- `BN254G1MSMEngine.swift` - Full GPU Pippenger MSM via Metal
- GPU path with CPU fallback for small inputs

### GPU MSM (G2) - Implemented
- `BN254G2MSMEngine.swift` - Full GPU Pippenger MSM for G2 via Metal
- Used for Dory commitment proofs and Groth16 batch verification

### Rust Bindings
- Full arkworks compatibility layer in `bindings/rust/src/arkworks.rs`
- `ArkMSM` wrapper for GPU MSM with arkworks types

## Scalar Conversion

### Important Discovery

Through testing, we found that `ark_bn254::Fr.into_bigint()` returns the **standard (non-Montgomery) integer representation**. This means the conversion is simple - just unpack the limbs.

### Correct Conversion Function

The source of truth is `bindings/rust/src/arkworks.rs`:

```rust
/// Convert arkworks Fr to Pippenger scalar format (8 x u32 limbs, little-endian).
pub fn ark_fr_to_pippenger_scalar(ark_fr: &ArkFr) -> [u32; 8] {
    let bigint: BigInteger256 = (*ark_fr).into_bigint();
    let limbs = bigint.0;
    // Just unpack 4 x u64 into 8 x u32 (little-endian)
    [
        limbs[0] as u32,
        (limbs[0] >> 32) as u32,
        limbs[1] as u32,
        (limbs[1] >> 32) as u32,
        limbs[2] as u32,
        (limbs[2] >> 32) as u32,
        limbs[3] as u32,
        (limbs[3] >> 32) as u32,
    ]
}
```

### Verification

The conversion is correct if:
- `Fr(1)` → `[1, 0, 0, 0, 0, 0, 0, 0]`
- `Fr(2)` → `[2, 0, 0, 0, 0, 0, 0, 0]`

All tests pass with this implementation.

### Simpler Alternative: Use ArkMSM Directly

The `ArkMSM` wrapper in `arkworks.rs` handles this automatically:

```rust
use ark_bn254::{Fr, G1Affine};
use zkmetal::ArkMSM;

let points: Vec<G1Affine> = /* ... */;
let scalars: Vec<Fr> = /* ... */;
let result = ArkMSM::msm(&points, &scalars).unwrap();
```

## C FFI Alternative

The function `bn254_fr_batch_to_limbs()` in `Sources/NeonFieldOps/bn254_msm.c`:

```c
// Just unpacks 4 x u64 into 8 x u32 (little-endian)
void bn254_fr_batch_to_limbs(const uint64_t *fr, uint32_t *limbs, int n);
```

## Benchmark Results (Apple M3 Pro)

| Size | Points | GPU Time | CPU Time | Speedup |
|------|--------|----------|----------|---------|
| 2^8 | 256 | 1.2ms | 401.5ms | 330x |
| 2^10 | 1,024 | 4.1ms | 1,559.9ms | 378x |
| 2^12 | 4,096 | 1.3ms | 5,940.2ms | 4,441x |
| 2^14 | 16,384 | 1.9ms | 24,062.5ms | 12,552x |
| 2^16 | 65,536 | 4.9ms | - | - |
| 2^17 | 131,072 | 9.0ms | - | - |
| 2^18 | 262,144 | 16.5ms | - | - |
| 2^20 | 1,048,576 | 61.2ms | - | - |

## Feature Flags

### Rust (Cargo.toml)

```toml
[dependencies]
zkmetal = { path = "path/to/zkmetal/bindings/rust", features = ["gpu", "arkworks"] }

# Or for CPU-only (NEON):
zkmetal = { path = "path/to/zkmetal/bindings/rust", features = ["neon", "arkworks"] }
```

### Available Features

| Feature | Description |
|---------|-------------|
| `neon` | ARM NEON SIMD operations (required for Apple Silicon CPU ops) |
| `gpu` | Metal GPU kernels (requires macOS with Metal) |
| `arkworks` | arkworks compatibility layer |

## Build Configuration

### macOS (Apple Silicon)

```bash
# With GPU support:
cargo build --features "gpu neon arkworks"
```

### Linux ARM64

Same as macOS - NEON feature works on Linux ARM64.

## Common Issues

### 1. Verification Failures

**Symptom**: Proofs verify correctly on CPU but fail with GPU MSM.

**Cause**: Incorrect scalar conversion.

**Fix**: Use `ArkMSM::msm()` which handles conversion automatically.

### 2. Feature Flag Issues

**Symptom**: `neon` feature not found.

**Fix**: Ensure `zkmetal` feature flag is added to jolt-core's `prover` feature list:

```toml
# In jolt-core/Cargo.toml
[features]
prover = ["zkmetal/neon", "other-deps..."]
```

## Testing

```bash
# Run all Rust tests
cargo test --features "neon arkworks"

# Run scalar conversion tests
cargo test --features arkworks scalar_conversion
```