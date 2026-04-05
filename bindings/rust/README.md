# zkmetal-sys

Raw FFI bindings and safe Rust wrappers for [zkMetal](https://github.com/example/zkMetal) --
GPU-accelerated ZK primitives on Apple Silicon (Metal GPU + ARM NEON CPU).

## Supported operations

### GPU (Metal) -- `gpu` feature (default)

| Operation | Curves/Fields | Engine API | Auto API |
|-----------|---------------|-----------|----------|
| MSM (256-bit, u8, u16, u32 scalars) | BN254 G1 | `MsmEngine` | `bn254_msm_auto` |
| Batch MSM | BN254 G1 | `MsmEngine::msm_batch` | `bn254_msm_batch_auto` |
| NTT / iNTT | BN254 Fr | `NttEngine` | `bn254_ntt_auto` |
| Poseidon2 hash | BN254 Fr | `Poseidon2Engine` | `bn254_poseidon2_hash_pairs_auto` |
| Keccak-256 | - | `KeccakEngine` | `keccak256_hash_auto` |
| FRI fold | BN254 Fr | `FriEngine` | `fri_fold_auto` |
| Batch pairing | BN254 | `PairingEngine` | `bn254_batch_pairing_auto` |

### CPU (ARM NEON) -- `neon` feature

| Operation | Curves/Fields |
|-----------|---------------|
| Field arithmetic (add/sub/mul/neg/inv/sqr) | BN254 Fr, BLS12-381 Fr/Fp, Pallas, Vesta, Ed25519, Stark252 |
| Batch field ops (NEON-vectorized) | BN254 Fr, Goldilocks |
| NTT / iNTT | BN254 Fr, BabyBear, Goldilocks, BLS12-377 Fr, Stark252 |
| Pippenger MSM | BN254, BLS12-381 G1, BLS12-377 G1, Pallas, Vesta, Grumpkin, secp256k1, Ed25519 |
| Pairing | BLS12-381, BN254 (CPU path) |
| EdDSA verify | Ed25519 |

## Prerequisites

Build the zkMetal Swift package first:

```bash
cd <zkMetal-repo>
swift build -c release
```

This produces the shared library at `.build/release/libzkMetal-ffi.dylib`.

## Building

```bash
# From the bindings/rust/ directory:
cargo build

# Or with explicit library path:
ZKMETAL_LIB_DIR=/path/to/zkMetal/.build/release cargo build

# CPU-only (no Metal GPU dependency):
cargo build --no-default-features --features neon

# Both GPU and CPU:
cargo build --features neon
```

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
zkmetal-sys = { path = "path/to/zkMetal/bindings/rust" }
```

### GPU MSM example

```rust
use zkmetal_sys::{bn254_msm_auto, gpu_available};

fn main() {
    assert!(gpu_available(), "No Metal GPU found");

    let n = 1024u32;
    let points = vec![0u8; n as usize * 64];   // affine points, Montgomery form
    let scalars = vec![0u8; n as usize * 32];   // standard form scalars

    let (x, y, z) = bn254_msm_auto(&points, &scalars, n).unwrap();
    println!("MSM result: {} bytes projective", x.len() + y.len() + z.len());
}
```

### GPU NTT example

```rust
use zkmetal_sys::{bn254_ntt_auto, bn254_intt_auto};

fn main() {
    let log_n = 12u32;
    let n = 1usize << log_n;
    let mut data = vec![0u8; n * 32]; // BN254 Fr elements, Montgomery form

    bn254_ntt_auto(&mut data, log_n).unwrap();
    bn254_intt_auto(&mut data, log_n).unwrap(); // round-trip
}
```

### Engine API (advanced, avoids re-initialization)

```rust
use zkmetal_sys::{MsmEngine, NttEngine};

fn main() {
    let msm_engine = MsmEngine::new().unwrap();
    let ntt_engine = NttEngine::new().unwrap();

    // Reuse engines across many calls (RAII -- dropped automatically).
    for _ in 0..100 {
        let points = vec![0u8; 256 * 64];
        let scalars = vec![0u8; 256 * 32];
        let result = msm_engine.msm(&points, &scalars, 256).unwrap();
    }
}
```

### CPU field arithmetic (neon feature)

```rust
use zkmetal_sys::bn254::{Fr, fr_add, fr_mul, fr_inverse};

fn main() {
    let a = Fr::from_raw([1, 0, 0, 0]);
    let b = Fr::from_raw([2, 0, 0, 0]);

    let c = fr_add(&a, &b);
    let d = fr_mul(&a, &b);
    let inv = fr_inverse(&a);
}
```

### CPU NTT (neon feature)

```rust
use zkmetal_sys::ntt::{babybear_ntt, babybear_intt, goldilocks_ntt_neon};

fn main() {
    // BabyBear NTT
    let log_n = 16u32;
    let mut data = vec![0u32; 1 << log_n];
    babybear_ntt(&mut data, log_n);
    babybear_intt(&mut data, log_n);

    // Goldilocks NTT (NEON-vectorized)
    let mut gl_data = vec![0u64; 1 << log_n];
    goldilocks_ntt_neon(&mut gl_data, log_n);
}
```

## Integration with Rust ZK crates

This crate is designed to slot into existing Rust ZK frameworks:

- **arkworks**: Implement `ark_ec::msm::VariableBaseMSM` by calling `bn254_msm_auto`
- **halo2**: Plug into `halo2_proofs::poly::commitment::MSM` via the engine API
- **plonky3**: Use `babybear_ntt` / `goldilocks_ntt_neon` for field-native NTTs
- **SP1**: Use the CPU NTT + MSM for BabyBear/BN254 acceleration

## Data format reference

| Type | Size | Format |
|------|------|--------|
| BN254 Fr (GPU) | 32 bytes | 8 x u32 LE, Montgomery form |
| BN254 Fr (CPU) | 4 x u64 | 4 x u64 LE, Montgomery form |
| BN254 G1 affine (GPU) | 64 bytes | x(32B) \|\| y(32B), Montgomery |
| BN254 G1 affine (CPU) | 8 x u64 | x[4] \|\| y[4], Montgomery |
| BN254 scalar | 32 bytes / 8 x u32 | Standard form (NOT Montgomery) |
| BN254 G1 projective (GPU) | 3 x 32 bytes | X, Y, Z each 32B Montgomery |
| BN254 G1 projective (CPU) | 12 x u64 | x[4], y[4], z[4] Montgomery |
| BLS12-381 Fp | 6 x u64 | Montgomery form |
| Goldilocks | 1 x u64 | Standard form, [0, p) |
| BabyBear | 1 x u32 | Standard form, [0, p) |

## License

MIT OR Apache-2.0
