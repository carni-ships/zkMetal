//! # zkmetal-sys
//!
//! Raw FFI bindings + safe wrappers for zkMetal -- GPU-accelerated ZK primitives
//! on Apple Silicon (Metal GPU + ARM NEON CPU).
//!
//! ## Features
//!
//! - `gpu` (default) -- Metal GPU kernels via `zkmetal.h` (MSM, NTT, Poseidon2, Keccak, FRI, Pairing)
//! - `neon` -- ARM NEON CPU kernels via `NeonFieldOps.h` (field arithmetic, NTT, MSM for many curves)
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use zkmetal_sys::{bn254_ntt_auto, bn254_msm_auto, gpu_available};
//!
//! assert!(gpu_available());
//!
//! // NTT: mutable slice of BN254 Fr elements (32 bytes each, Montgomery form)
//! let mut data = vec![0u8; 32 * 1024]; // 1024 elements
//! bn254_ntt_auto(&mut data, 10).unwrap(); // log2(1024) = 10
//!
//! // MSM: points (64B each) + scalars (32B each) -> projective result
//! let points = vec![0u8; 64 * 256];
//! let scalars = vec![0u8; 32 * 256];
//! let (x, y, z) = bn254_msm_auto(&points, &scalars, 256).unwrap();
//! ```

use std::fmt;

pub mod bn254;
pub mod msm;
pub mod ntt;

#[cfg(feature = "neon")]
pub mod neon;

// Re-export commonly used items at crate root.
pub use bn254::*;
pub use msm::*;
pub use ntt::*;

// ============================================================================
// Error types
// ============================================================================

/// Error codes returned by zkMetal C FFI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ZkMetalError {
    /// No Metal GPU available on this system.
    NoGpu = -1,
    /// Invalid input (e.g., zero points, bad size).
    InvalidInput = -2,
    /// GPU execution error.
    GpuError = -3,
    /// Memory allocation failed.
    AllocFailed = -4,
    /// Unknown error code from C FFI.
    Unknown = -99,
}

impl fmt::Display for ZkMetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZkMetalError::NoGpu => write!(f, "No Metal GPU available"),
            ZkMetalError::InvalidInput => write!(f, "Invalid input"),
            ZkMetalError::GpuError => write!(f, "GPU execution error"),
            ZkMetalError::AllocFailed => write!(f, "Memory allocation failed"),
            ZkMetalError::Unknown => write!(f, "Unknown zkMetal error"),
        }
    }
}

impl std::error::Error for ZkMetalError {}

/// Result type for zkMetal operations.
pub type Result<T> = std::result::Result<T, ZkMetalError>;

/// Map a C status code to a Rust Result.
pub(crate) fn check_status(status: i32) -> Result<()> {
    match status {
        0 => Ok(()),
        -1 => Err(ZkMetalError::NoGpu),
        -2 => Err(ZkMetalError::InvalidInput),
        -3 => Err(ZkMetalError::GpuError),
        -4 => Err(ZkMetalError::AllocFailed),
        _ => Err(ZkMetalError::Unknown),
    }
}

// ============================================================================
// Raw C FFI -- GPU engine API (zkmetal.h)
// ============================================================================

#[cfg(feature = "gpu")]
pub(crate) mod ffi {
    use std::os::raw::c_char;

    extern "C" {
        // -- Engine lifecycle --
        pub fn zkmetal_msm_engine_create(out: *mut *mut std::ffi::c_void) -> i32;
        pub fn zkmetal_msm_engine_destroy(engine: *mut std::ffi::c_void);
        pub fn zkmetal_ntt_engine_create(out: *mut *mut std::ffi::c_void) -> i32;
        pub fn zkmetal_ntt_engine_destroy(engine: *mut std::ffi::c_void);
        pub fn zkmetal_poseidon2_engine_create(out: *mut *mut std::ffi::c_void) -> i32;
        pub fn zkmetal_poseidon2_engine_destroy(engine: *mut std::ffi::c_void);
        pub fn zkmetal_keccak_engine_create(out: *mut *mut std::ffi::c_void) -> i32;
        pub fn zkmetal_keccak_engine_destroy(engine: *mut std::ffi::c_void);
        pub fn zkmetal_fri_engine_create(out: *mut *mut std::ffi::c_void) -> i32;
        pub fn zkmetal_fri_engine_destroy(engine: *mut std::ffi::c_void);
        pub fn zkmetal_pairing_engine_create(out: *mut *mut std::ffi::c_void) -> i32;
        pub fn zkmetal_pairing_engine_destroy(engine: *mut std::ffi::c_void);

        // -- MSM (full 256-bit scalars) --
        pub fn zkmetal_bn254_msm(
            engine: *mut std::ffi::c_void,
            points: *const u8, scalars: *const u8, n_points: u32,
            result_x: *mut u8, result_y: *mut u8, result_z: *mut u8,
        ) -> i32;
        pub fn zkmetal_bn254_msm_auto(
            points: *const u8, scalars: *const u8, n_points: u32,
            result_x: *mut u8, result_y: *mut u8, result_z: *mut u8,
        ) -> i32;

        // -- Small-scalar MSM --
        pub fn zkmetal_bn254_msm_u8(
            engine: *mut std::ffi::c_void,
            points: *const u8, scalars: *const u8, n_points: u32,
            result_x: *mut u8, result_y: *mut u8, result_z: *mut u8,
        ) -> i32;
        pub fn zkmetal_bn254_msm_u16(
            engine: *mut std::ffi::c_void,
            points: *const u8, scalars: *const u8, n_points: u32,
            result_x: *mut u8, result_y: *mut u8, result_z: *mut u8,
        ) -> i32;
        pub fn zkmetal_bn254_msm_u32(
            engine: *mut std::ffi::c_void,
            points: *const u8, scalars: *const u8, n_points: u32,
            result_x: *mut u8, result_y: *mut u8, result_z: *mut u8,
        ) -> i32;
        pub fn zkmetal_bn254_msm_u8_auto(
            points: *const u8, scalars: *const u8, n_points: u32,
            result_x: *mut u8, result_y: *mut u8, result_z: *mut u8,
        ) -> i32;
        pub fn zkmetal_bn254_msm_u16_auto(
            points: *const u8, scalars: *const u8, n_points: u32,
            result_x: *mut u8, result_y: *mut u8, result_z: *mut u8,
        ) -> i32;
        pub fn zkmetal_bn254_msm_u32_auto(
            points: *const u8, scalars: *const u8, n_points: u32,
            result_x: *mut u8, result_y: *mut u8, result_z: *mut u8,
        ) -> i32;

        // -- Batch MSM --
        pub fn zkmetal_bn254_msm_batch(
            engine: *mut std::ffi::c_void,
            all_points: *const u8, all_scalars: *const u8,
            counts: *const u32, n_msms: u32, results: *mut u8,
        ) -> i32;
        pub fn zkmetal_bn254_msm_batch_auto(
            all_points: *const u8, all_scalars: *const u8,
            counts: *const u32, n_msms: u32, results: *mut u8,
        ) -> i32;

        // -- NTT --
        pub fn zkmetal_bn254_ntt(engine: *mut std::ffi::c_void, data: *mut u8, log_n: u32) -> i32;
        pub fn zkmetal_bn254_intt(engine: *mut std::ffi::c_void, data: *mut u8, log_n: u32) -> i32;
        pub fn zkmetal_bn254_ntt_auto(data: *mut u8, log_n: u32) -> i32;
        pub fn zkmetal_bn254_intt_auto(data: *mut u8, log_n: u32) -> i32;

        // -- Poseidon2 --
        pub fn zkmetal_bn254_poseidon2_hash_pairs(
            engine: *mut std::ffi::c_void,
            input: *const u8, n_pairs: u32, output: *mut u8,
        ) -> i32;
        pub fn zkmetal_bn254_poseidon2_hash_pairs_auto(
            input: *const u8, n_pairs: u32, output: *mut u8,
        ) -> i32;

        // -- Keccak-256 --
        pub fn zkmetal_keccak256_hash(
            engine: *mut std::ffi::c_void,
            input: *const u8, n_inputs: u32, output: *mut u8,
        ) -> i32;
        pub fn zkmetal_keccak256_hash_auto(
            input: *const u8, n_inputs: u32, output: *mut u8,
        ) -> i32;

        // -- FRI Fold --
        pub fn zkmetal_fri_fold(
            engine: *mut std::ffi::c_void,
            evals: *const u8, log_n: u32, beta: *const u8, result: *mut u8,
        ) -> i32;
        pub fn zkmetal_fri_fold_auto(
            evals: *const u8, log_n: u32, beta: *const u8, result: *mut u8,
        ) -> i32;

        // -- Batch Pairing --
        pub fn zkmetal_bn254_batch_pairing(
            engine: *mut std::ffi::c_void,
            g1_points: *const u8, g2_points: *const u8, n_pairs: u32,
            result: *mut u8,
        ) -> i32;
        pub fn zkmetal_bn254_pairing_check(
            engine: *mut std::ffi::c_void,
            g1_points: *const u8, g2_points: *const u8, n_pairs: u32,
        ) -> i32;
        pub fn zkmetal_bn254_batch_pairing_auto(
            g1_points: *const u8, g2_points: *const u8, n_pairs: u32,
            result: *mut u8,
        ) -> i32;
        pub fn zkmetal_bn254_pairing_check_auto(
            g1_points: *const u8, g2_points: *const u8, n_pairs: u32,
        ) -> i32;

        // -- Utility --
        pub fn zkmetal_set_shader_dir(path: *const c_char);
        pub fn zkmetal_gpu_available() -> i32;
        pub fn zkmetal_version() -> *const c_char;
    }
}

// ============================================================================
// GPU engine RAII handles
// ============================================================================

/// BN254 Poseidon2 hash engine handle (GPU). Dropped automatically.
#[cfg(feature = "gpu")]
pub struct Poseidon2Engine {
    raw: *mut std::ffi::c_void,
}

#[cfg(feature = "gpu")]
unsafe impl Send for Poseidon2Engine {}

#[cfg(feature = "gpu")]
impl Poseidon2Engine {
    /// Create a new Poseidon2 engine (compiles GPU shaders, allocates resources).
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check_status(unsafe { ffi::zkmetal_poseidon2_engine_create(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Batch hash pairs: input is `2 * n_pairs * 32` bytes, output is `n_pairs * 32` bytes.
    pub fn hash_pairs(&self, input: &[u8], n_pairs: u32, output: &mut [u8]) -> Result<()> {
        assert_eq!(input.len(), n_pairs as usize * 64);
        assert_eq!(output.len(), n_pairs as usize * 32);
        check_status(unsafe {
            ffi::zkmetal_bn254_poseidon2_hash_pairs(
                self.raw, input.as_ptr(), n_pairs, output.as_mut_ptr(),
            )
        })
    }
}

#[cfg(feature = "gpu")]
impl Drop for Poseidon2Engine {
    fn drop(&mut self) {
        unsafe { ffi::zkmetal_poseidon2_engine_destroy(self.raw) }
    }
}

/// Keccak-256 GPU engine handle. Dropped automatically.
#[cfg(feature = "gpu")]
pub struct KeccakEngine {
    raw: *mut std::ffi::c_void,
}

#[cfg(feature = "gpu")]
unsafe impl Send for KeccakEngine {}

#[cfg(feature = "gpu")]
impl KeccakEngine {
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check_status(unsafe { ffi::zkmetal_keccak_engine_create(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Batch hash: each input is 64 bytes, each output is 32 bytes.
    pub fn hash(&self, input: &[u8], n: u32, output: &mut [u8]) -> Result<()> {
        assert_eq!(input.len(), n as usize * 64);
        assert_eq!(output.len(), n as usize * 32);
        check_status(unsafe {
            ffi::zkmetal_keccak256_hash(self.raw, input.as_ptr(), n, output.as_mut_ptr())
        })
    }
}

#[cfg(feature = "gpu")]
impl Drop for KeccakEngine {
    fn drop(&mut self) {
        unsafe { ffi::zkmetal_keccak_engine_destroy(self.raw) }
    }
}

/// FRI fold GPU engine handle. Dropped automatically.
#[cfg(feature = "gpu")]
pub struct FriEngine {
    raw: *mut std::ffi::c_void,
}

#[cfg(feature = "gpu")]
unsafe impl Send for FriEngine {}

#[cfg(feature = "gpu")]
impl FriEngine {
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check_status(unsafe { ffi::zkmetal_fri_engine_create(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Single FRI fold: `evals` has `2^log_n * 32` bytes, `result` has `2^(log_n-1) * 32` bytes.
    /// `beta` is a 32-byte field element (Montgomery form).
    pub fn fold(&self, evals: &[u8], log_n: u32, beta: &[u8; 32], result: &mut [u8]) -> Result<()> {
        let n = 1usize << log_n;
        assert_eq!(evals.len(), n * 32);
        assert_eq!(result.len(), (n / 2) * 32);
        check_status(unsafe {
            ffi::zkmetal_fri_fold(self.raw, evals.as_ptr(), log_n, beta.as_ptr(), result.as_mut_ptr())
        })
    }
}

#[cfg(feature = "gpu")]
impl Drop for FriEngine {
    fn drop(&mut self) {
        unsafe { ffi::zkmetal_fri_engine_destroy(self.raw) }
    }
}

/// BN254 Pairing GPU engine handle. Dropped automatically.
#[cfg(feature = "gpu")]
pub struct PairingEngine {
    raw: *mut std::ffi::c_void,
}

#[cfg(feature = "gpu")]
unsafe impl Send for PairingEngine {}

#[cfg(feature = "gpu")]
impl PairingEngine {
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check_status(unsafe { ffi::zkmetal_pairing_engine_create(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Batch pairing: compute product of e(g1[i], g2[i]).
    ///
    /// - `g1_points`: `n * 64` bytes -- affine G1 points in Montgomery form
    /// - `g2_points`: `n * 128` bytes -- affine G2 points (x0,x1,y0,y1 Montgomery Fp2)
    ///
    /// Returns 384-byte Fp12 result in Montgomery form.
    pub fn batch_pairing(&self, g1_points: &[u8], g2_points: &[u8], n: u32) -> Result<[u8; 384]> {
        assert_eq!(g1_points.len(), n as usize * 64);
        assert_eq!(g2_points.len(), n as usize * 128);
        let mut result = [0u8; 384];
        check_status(unsafe {
            ffi::zkmetal_bn254_batch_pairing(
                self.raw, g1_points.as_ptr(), g2_points.as_ptr(), n, result.as_mut_ptr(),
            )
        })?;
        Ok(result)
    }

    /// Pairing check: verify product of e(g1[i], g2[i]) == 1 (Gt identity).
    /// Returns `Ok(true)` if check passes, `Ok(false)` if it fails.
    pub fn pairing_check(&self, g1_points: &[u8], g2_points: &[u8], n: u32) -> Result<bool> {
        assert_eq!(g1_points.len(), n as usize * 64);
        assert_eq!(g2_points.len(), n as usize * 128);
        let status = unsafe {
            ffi::zkmetal_bn254_pairing_check(self.raw, g1_points.as_ptr(), g2_points.as_ptr(), n)
        };
        match status {
            0 => Ok(true),
            -2 => Ok(false),
            _ => check_status(status).map(|_| unreachable!()),
        }
    }
}

#[cfg(feature = "gpu")]
impl Drop for PairingEngine {
    fn drop(&mut self) {
        unsafe { ffi::zkmetal_pairing_engine_destroy(self.raw) }
    }
}

// ============================================================================
// Convenience (auto) API -- GPU lazy singleton
// ============================================================================

/// Convenience Poseidon2 batch hash pairs (GPU, lazy singleton).
#[cfg(feature = "gpu")]
pub fn bn254_poseidon2_hash_pairs_auto(input: &[u8], n_pairs: u32, output: &mut [u8]) -> Result<()> {
    assert_eq!(input.len(), n_pairs as usize * 64);
    assert_eq!(output.len(), n_pairs as usize * 32);
    check_status(unsafe {
        ffi::zkmetal_bn254_poseidon2_hash_pairs_auto(input.as_ptr(), n_pairs, output.as_mut_ptr())
    })
}

/// Convenience Keccak-256 batch hash (GPU, lazy singleton).
/// Each input is 64 bytes, each output is 32 bytes.
#[cfg(feature = "gpu")]
pub fn keccak256_hash_auto(input: &[u8], n: u32, output: &mut [u8]) -> Result<()> {
    assert_eq!(input.len(), n as usize * 64);
    assert_eq!(output.len(), n as usize * 32);
    check_status(unsafe {
        ffi::zkmetal_keccak256_hash_auto(input.as_ptr(), n, output.as_mut_ptr())
    })
}

/// Convenience FRI fold (GPU, lazy singleton).
#[cfg(feature = "gpu")]
pub fn fri_fold_auto(evals: &[u8], log_n: u32, beta: &[u8; 32], result: &mut [u8]) -> Result<()> {
    let n = 1usize << log_n;
    assert_eq!(evals.len(), n * 32);
    assert_eq!(result.len(), (n / 2) * 32);
    check_status(unsafe {
        ffi::zkmetal_fri_fold_auto(evals.as_ptr(), log_n, beta.as_ptr(), result.as_mut_ptr())
    })
}

/// Convenience batch pairing (GPU, lazy singleton).
#[cfg(feature = "gpu")]
pub fn bn254_batch_pairing_auto(g1_points: &[u8], g2_points: &[u8], n: u32) -> Result<[u8; 384]> {
    assert_eq!(g1_points.len(), n as usize * 64);
    assert_eq!(g2_points.len(), n as usize * 128);
    let mut result = [0u8; 384];
    check_status(unsafe {
        ffi::zkmetal_bn254_batch_pairing_auto(
            g1_points.as_ptr(), g2_points.as_ptr(), n, result.as_mut_ptr(),
        )
    })?;
    Ok(result)
}

/// Convenience pairing check (GPU, lazy singleton).
/// Returns `Ok(true)` if check passes, `Ok(false)` if it fails.
#[cfg(feature = "gpu")]
pub fn bn254_pairing_check_auto(g1_points: &[u8], g2_points: &[u8], n: u32) -> Result<bool> {
    assert_eq!(g1_points.len(), n as usize * 64);
    assert_eq!(g2_points.len(), n as usize * 128);
    let status = unsafe {
        ffi::zkmetal_bn254_pairing_check_auto(g1_points.as_ptr(), g2_points.as_ptr(), n)
    };
    match status {
        0 => Ok(true),
        -2 => Ok(false),
        _ => check_status(status).map(|_| unreachable!()),
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Set the shader directory path. Must be called before any engine creation
/// if running outside the zkMetal source tree.
#[cfg(feature = "gpu")]
pub fn set_shader_dir(path: &str) {
    let c_path = std::ffi::CString::new(path).expect("path contains null byte");
    unsafe { ffi::zkmetal_set_shader_dir(c_path.as_ptr()) }
}

/// Check if a Metal GPU is available.
#[cfg(feature = "gpu")]
pub fn gpu_available() -> bool {
    unsafe { ffi::zkmetal_gpu_available() == 1 }
}

/// Get the zkMetal library version string.
#[cfg(feature = "gpu")]
pub fn version() -> &'static str {
    unsafe {
        let ptr = ffi::zkmetal_version();
        std::ffi::CStr::from_ptr(ptr)
            .to_str()
            .unwrap_or("unknown")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "gpu")]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty(), "version should not be empty");
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_available() {
        let available = gpu_available();
        // On macOS with Metal, this should be true.
        println!("GPU available: {}", available);
    }
}
