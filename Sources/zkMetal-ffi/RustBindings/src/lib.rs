//! # zkmetal
//!
//! Rust bindings for zkMetal — GPU-accelerated ZK primitives on Apple Silicon (Metal).
//!
//! Provides safe wrappers around the C FFI exported by the zkMetal Swift library.
//! All GPU engine initialization is handled internally via lazy singletons
//! (the `_auto` API), or explicitly via engine handles for advanced use.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use zkmetal::{bn254_ntt_auto, bn254_msm_auto};
//!
//! // NTT: pass a mutable slice of BN254 Fr elements (32 bytes each, Montgomery form)
//! let mut data = vec![0u8; 32 * 1024]; // 1024 elements
//! bn254_ntt_auto(&mut data, 10).unwrap(); // log2(1024) = 10
//!
//! // MSM: points (64B each) + scalars (32B each) -> projective result
//! let points = vec![0u8; 64 * 256];
//! let scalars = vec![0u8; 32 * 256];
//! let (x, y, z) = bn254_msm_auto(&points, &scalars, 256).unwrap();
//! ```

use std::fmt;

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

pub type Result<T> = std::result::Result<T, ZkMetalError>;

fn check_status(status: i32) -> Result<()> {
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
// Raw C FFI bindings
// ============================================================================

mod ffi {
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

        // -- MSM --
        pub fn zkmetal_bn254_msm(
            engine: *mut std::ffi::c_void,
            points: *const u8,
            scalars: *const u8,
            n_points: u32,
            result_x: *mut u8,
            result_y: *mut u8,
            result_z: *mut u8,
        ) -> i32;

        pub fn zkmetal_bn254_msm_auto(
            points: *const u8,
            scalars: *const u8,
            n_points: u32,
            result_x: *mut u8,
            result_y: *mut u8,
            result_z: *mut u8,
        ) -> i32;

        // -- NTT --
        pub fn zkmetal_bn254_ntt(engine: *mut std::ffi::c_void, data: *mut u8, log_n: u32) -> i32;
        pub fn zkmetal_bn254_intt(engine: *mut std::ffi::c_void, data: *mut u8, log_n: u32) -> i32;
        pub fn zkmetal_bn254_ntt_auto(data: *mut u8, log_n: u32) -> i32;
        pub fn zkmetal_bn254_intt_auto(data: *mut u8, log_n: u32) -> i32;

        // -- Poseidon2 --
        pub fn zkmetal_bn254_poseidon2_hash_pairs(
            engine: *mut std::ffi::c_void,
            input: *const u8,
            n_pairs: u32,
            output: *mut u8,
        ) -> i32;

        pub fn zkmetal_bn254_poseidon2_hash_pairs_auto(
            input: *const u8,
            n_pairs: u32,
            output: *mut u8,
        ) -> i32;

        // -- Keccak-256 --
        pub fn zkmetal_keccak256_hash(
            engine: *mut std::ffi::c_void,
            input: *const u8,
            n_inputs: u32,
            output: *mut u8,
        ) -> i32;

        pub fn zkmetal_keccak256_hash_auto(
            input: *const u8,
            n_inputs: u32,
            output: *mut u8,
        ) -> i32;

        // -- FRI --
        pub fn zkmetal_fri_fold(
            engine: *mut std::ffi::c_void,
            evals: *const u8,
            log_n: u32,
            beta: *const u8,
            result: *mut u8,
        ) -> i32;

        pub fn zkmetal_fri_fold_auto(
            evals: *const u8,
            log_n: u32,
            beta: *const u8,
            result: *mut u8,
        ) -> i32;

        // -- Utility --
        pub fn zkmetal_set_shader_dir(path: *const c_char);
        pub fn zkmetal_gpu_available() -> i32;
        pub fn zkmetal_version() -> *const c_char;
    }
}

// ============================================================================
// Engine handles (RAII wrappers)
// ============================================================================

/// BN254 MSM engine handle. Dropped automatically.
pub struct MsmEngine {
    raw: *mut std::ffi::c_void,
}

unsafe impl Send for MsmEngine {}

impl MsmEngine {
    /// Create a new MSM engine (compiles GPU shaders, allocates resources).
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check_status(unsafe { ffi::zkmetal_msm_engine_create(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Compute MSM: result = sum(scalars[i] * points[i]).
    ///
    /// - `points`: `n * 64` bytes — affine points in Montgomery form
    /// - `scalars`: `n * 32` bytes — scalars in standard form (8x u32 LE)
    ///
    /// Returns `(x, y, z)` projective coordinates, each 32 bytes in Montgomery form.
    pub fn msm(&self, points: &[u8], scalars: &[u8], n: u32) -> Result<([u8; 32], [u8; 32], [u8; 32])> {
        assert_eq!(points.len(), n as usize * 64, "points must be n*64 bytes");
        assert_eq!(scalars.len(), n as usize * 32, "scalars must be n*32 bytes");
        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        let mut z = [0u8; 32];
        check_status(unsafe {
            ffi::zkmetal_bn254_msm(
                self.raw,
                points.as_ptr(),
                scalars.as_ptr(),
                n,
                x.as_mut_ptr(),
                y.as_mut_ptr(),
                z.as_mut_ptr(),
            )
        })?;
        Ok((x, y, z))
    }
}

impl Drop for MsmEngine {
    fn drop(&mut self) {
        unsafe { ffi::zkmetal_msm_engine_destroy(self.raw) }
    }
}

/// BN254 NTT engine handle.
pub struct NttEngine {
    raw: *mut std::ffi::c_void,
}

unsafe impl Send for NttEngine {}

impl NttEngine {
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check_status(unsafe { ffi::zkmetal_ntt_engine_create(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Forward NTT in-place. `data` must be `2^log_n * 32` bytes.
    pub fn ntt(&self, data: &mut [u8], log_n: u32) -> Result<()> {
        assert_eq!(data.len(), (1usize << log_n) * 32);
        check_status(unsafe { ffi::zkmetal_bn254_ntt(self.raw, data.as_mut_ptr(), log_n) })
    }

    /// Inverse NTT in-place. `data` must be `2^log_n * 32` bytes.
    pub fn intt(&self, data: &mut [u8], log_n: u32) -> Result<()> {
        assert_eq!(data.len(), (1usize << log_n) * 32);
        check_status(unsafe { ffi::zkmetal_bn254_intt(self.raw, data.as_mut_ptr(), log_n) })
    }
}

impl Drop for NttEngine {
    fn drop(&mut self) {
        unsafe { ffi::zkmetal_ntt_engine_destroy(self.raw) }
    }
}

/// BN254 Poseidon2 hash engine handle.
pub struct Poseidon2Engine {
    raw: *mut std::ffi::c_void,
}

unsafe impl Send for Poseidon2Engine {}

impl Poseidon2Engine {
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
                self.raw,
                input.as_ptr(),
                n_pairs,
                output.as_mut_ptr(),
            )
        })
    }
}

impl Drop for Poseidon2Engine {
    fn drop(&mut self) {
        unsafe { ffi::zkmetal_poseidon2_engine_destroy(self.raw) }
    }
}

/// Keccak-256 GPU engine handle.
pub struct KeccakEngine {
    raw: *mut std::ffi::c_void,
}

unsafe impl Send for KeccakEngine {}

impl KeccakEngine {
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check_status(unsafe { ffi::zkmetal_keccak_engine_create(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Batch hash: each input is 64 bytes, each output is 32 bytes.
    /// `input` must be `n * 64` bytes, `output` must be `n * 32` bytes.
    pub fn hash(&self, input: &[u8], n: u32, output: &mut [u8]) -> Result<()> {
        assert_eq!(input.len(), n as usize * 64);
        assert_eq!(output.len(), n as usize * 32);
        check_status(unsafe {
            ffi::zkmetal_keccak256_hash(self.raw, input.as_ptr(), n, output.as_mut_ptr())
        })
    }
}

impl Drop for KeccakEngine {
    fn drop(&mut self) {
        unsafe { ffi::zkmetal_keccak_engine_destroy(self.raw) }
    }
}

/// FRI fold engine handle.
pub struct FriEngine {
    raw: *mut std::ffi::c_void,
}

unsafe impl Send for FriEngine {}

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
            ffi::zkmetal_fri_fold(
                self.raw,
                evals.as_ptr(),
                log_n,
                beta.as_ptr(),
                result.as_mut_ptr(),
            )
        })
    }
}

impl Drop for FriEngine {
    fn drop(&mut self) {
        unsafe { ffi::zkmetal_fri_engine_destroy(self.raw) }
    }
}

// ============================================================================
// Convenience (auto) API — uses lazy singleton engines
// ============================================================================

/// Convenience MSM using a lazy singleton engine.
///
/// - `points`: `n * 64` bytes (affine, Montgomery form)
/// - `scalars`: `n * 32` bytes (standard form, 8x u32 LE)
///
/// Returns `(x, y, z)` projective coordinates, each 32 bytes.
pub fn bn254_msm_auto(
    points: &[u8],
    scalars: &[u8],
    n: u32,
) -> Result<([u8; 32], [u8; 32], [u8; 32])> {
    assert_eq!(points.len(), n as usize * 64);
    assert_eq!(scalars.len(), n as usize * 32);
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    let mut z = [0u8; 32];
    check_status(unsafe {
        ffi::zkmetal_bn254_msm_auto(
            points.as_ptr(),
            scalars.as_ptr(),
            n,
            x.as_mut_ptr(),
            y.as_mut_ptr(),
            z.as_mut_ptr(),
        )
    })?;
    Ok((x, y, z))
}

/// Convenience forward NTT in-place. `data` must be `2^log_n * 32` bytes.
pub fn bn254_ntt_auto(data: &mut [u8], log_n: u32) -> Result<()> {
    assert_eq!(data.len(), (1usize << log_n) * 32);
    check_status(unsafe { ffi::zkmetal_bn254_ntt_auto(data.as_mut_ptr(), log_n) })
}

/// Convenience inverse NTT in-place. `data` must be `2^log_n * 32` bytes.
pub fn bn254_intt_auto(data: &mut [u8], log_n: u32) -> Result<()> {
    assert_eq!(data.len(), (1usize << log_n) * 32);
    check_status(unsafe { ffi::zkmetal_bn254_intt_auto(data.as_mut_ptr(), log_n) })
}

/// Convenience Poseidon2 batch hash pairs.
pub fn bn254_poseidon2_hash_pairs_auto(input: &[u8], n_pairs: u32, output: &mut [u8]) -> Result<()> {
    assert_eq!(input.len(), n_pairs as usize * 64);
    assert_eq!(output.len(), n_pairs as usize * 32);
    check_status(unsafe {
        ffi::zkmetal_bn254_poseidon2_hash_pairs_auto(input.as_ptr(), n_pairs, output.as_mut_ptr())
    })
}

/// Convenience Keccak-256 batch hash (64-byte inputs -> 32-byte outputs).
pub fn keccak256_hash_auto(input: &[u8], n: u32, output: &mut [u8]) -> Result<()> {
    assert_eq!(input.len(), n as usize * 64);
    assert_eq!(output.len(), n as usize * 32);
    check_status(unsafe {
        ffi::zkmetal_keccak256_hash_auto(input.as_ptr(), n, output.as_mut_ptr())
    })
}

/// Convenience FRI fold.
pub fn fri_fold_auto(evals: &[u8], log_n: u32, beta: &[u8; 32], result: &mut [u8]) -> Result<()> {
    let n = 1usize << log_n;
    assert_eq!(evals.len(), n * 32);
    assert_eq!(result.len(), (n / 2) * 32);
    check_status(unsafe {
        ffi::zkmetal_fri_fold_auto(evals.as_ptr(), log_n, beta.as_ptr(), result.as_mut_ptr())
    })
}

// ============================================================================
// Utility
// ============================================================================

/// Set the shader directory path. Must be called before any engine creation
/// if running outside the zkMetal source tree.
pub fn set_shader_dir(path: &str) {
    let c_path = std::ffi::CString::new(path).expect("path contains null byte");
    unsafe { ffi::zkmetal_set_shader_dir(c_path.as_ptr()) }
}

/// Check if a Metal GPU is available.
pub fn gpu_available() -> bool {
    unsafe { ffi::zkmetal_gpu_available() == 1 }
}

/// Get the zkMetal library version string.
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
    fn test_version() {
        let v = version();
        assert!(v.starts_with("0."), "version should start with 0., got: {}", v);
    }

    #[test]
    fn test_gpu_available() {
        // On macOS with Metal, this should be true
        let available = gpu_available();
        println!("GPU available: {}", available);
    }
}
