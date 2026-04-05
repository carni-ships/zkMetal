//! Safe Rust wrappers for Metal-accelerated ZK cryptographic primitives.
//!
//! This crate wraps the raw FFI in `zkmetal-sys` with safe, idiomatic Rust types.
//! All operations are hardware-accelerated via ARM NEON intrinsics and optionally
//! Apple Metal GPU on Apple Silicon (aarch64).
//!
//! # Modules
//!
//! - [`bn254`] - BN254 Fr/Fq field elements and G1 curve operations
//! - [`bls12_381`] - BLS12-381 Fr/Fp/Fp2 fields and G1/G2 curve operations
//! - [`msm`] - Multi-scalar multiplication (Pippenger) for all supported curves
//! - [`ntt`] - Number Theoretic Transform for BN254, BLS12-377, Goldilocks, BabyBear, Stark252
//! - [`hash`] - Poseidon2, Keccak-256, Blake3 hash functions
//! - [`pairing`] - BN254 and BLS12-381 optimal Ate pairing

#![cfg(target_arch = "aarch64")]

pub mod bn254;
pub mod bls12_381;
pub mod msm;
pub mod ntt;
pub mod hash;
pub mod pairing;

pub use zkmetal_sys;
