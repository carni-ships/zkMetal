//! Safe Rust bindings to zkMetal GPU-accelerated ZK primitives.
//!
//! Provides GPU-accelerated MSM, NTT, and Poseidon2 hashing for BN254
//! on Apple Silicon (Metal GPU backend).
//!
//! # Example
//! ```no_run
//! use zkmetal::{Fr, MsmEngine, NttEngine, Poseidon2Engine};
//!
//! // Check GPU availability
//! assert!(zkmetal::gpu_available());
//!
//! // NTT: forward + inverse roundtrip
//! let ntt = NttEngine::new().unwrap();
//! let mut data = vec![Fr::ZERO; 1024]; // 2^10 field elements
//! ntt.ntt_inplace(&mut data).unwrap();  // forward NTT
//! ntt.intt_inplace(&mut data).unwrap(); // inverse NTT (recovers original)
//! ```

use std::ffi::{CStr, CString};
use std::fmt;
use std::sync::Once;

use zkmetal_sys as ffi;

static INIT_SHADER_DIR: Once = Once::new();

/// Initialize the shader directory from the build-time path.
/// Called automatically by engine constructors.
fn ensure_shader_dir() {
    INIT_SHADER_DIR.call_once(|| {
        // The shader dir is baked in at build time via DEP_ZKMETAL_FFI_SHADER_DIR
        // from the zkmetal-sys build script.
        let shader_dir = env!("DEP_ZKMETAL_FFI_SHADER_DIR", "zkmetal-sys must set DEP_ZKMETAL_FFI_SHADER_DIR");
        if !shader_dir.is_empty() {
            if let Ok(cstr) = CString::new(shader_dir) {
                unsafe { ffi::zkmetal_set_shader_dir(cstr.as_ptr()) };
            }
        }
    });
}

// ============================================================================
// Error type
// ============================================================================

/// Errors returned by zkMetal operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// No Metal GPU device found.
    NoGpu,
    /// Invalid input (wrong size, null pointer, etc).
    InvalidInput,
    /// GPU execution error.
    GpuError,
    /// Memory allocation failed.
    AllocFailed,
    /// Unknown error code from the C library.
    Unknown(i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoGpu => write!(f, "No Metal GPU available"),
            Error::InvalidInput => write!(f, "Invalid input"),
            Error::GpuError => write!(f, "GPU execution error"),
            Error::AllocFailed => write!(f, "Memory allocation failed"),
            Error::Unknown(code) => write!(f, "Unknown zkMetal error (code {})", code),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn check(status: i32) -> Result<()> {
    match status {
        ffi::ZKMETAL_SUCCESS => Ok(()),
        ffi::ZKMETAL_ERR_NO_GPU => Err(Error::NoGpu),
        ffi::ZKMETAL_ERR_INVALID_INPUT => Err(Error::InvalidInput),
        ffi::ZKMETAL_ERR_GPU_ERROR => Err(Error::GpuError),
        ffi::ZKMETAL_ERR_ALLOC_FAILED => Err(Error::AllocFailed),
        other => Err(Error::Unknown(other)),
    }
}

// ============================================================================
// Utility
// ============================================================================

/// Returns true if a Metal GPU is available on this system.
pub fn gpu_available() -> bool {
    unsafe { ffi::zkmetal_gpu_available() == 1 }
}

/// Returns the zkMetal library version string.
pub fn version() -> &'static str {
    unsafe {
        let ptr = ffi::zkmetal_version();
        CStr::from_ptr(ptr).to_str().unwrap_or("unknown")
    }
}

// ============================================================================
// Field element type (32 bytes, opaque)
// ============================================================================

/// A BN254 Fr field element (32 bytes, 8x u32 little-endian limbs, Montgomery form).
///
/// This is a transparent wrapper around `[u8; 32]` for type safety.
/// The internal representation matches zkMetal's Montgomery form.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Fr(pub [u8; 32]);

impl Fr {
    /// The zero element.
    pub const ZERO: Self = Self([0u8; 32]);

    /// Create from raw 32 bytes (must be valid Montgomery-form Fr).
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get the raw bytes.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0
    }

    /// Create from 4x u64 limbs (little-endian).
    pub fn from_u64_limbs(limbs: [u64; 4]) -> Self {
        let mut bytes = [0u8; 32];
        for (i, limb) in limbs.iter().enumerate() {
            bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        Self(bytes)
    }

    /// Convert to 4x u64 limbs (little-endian).
    pub fn to_u64_limbs(&self) -> [u64; 4] {
        let mut limbs = [0u64; 4];
        for (i, limb) in limbs.iter_mut().enumerate() {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&self.0[i * 8..(i + 1) * 8]);
            *limb = u64::from_le_bytes(buf);
        }
        limbs
    }
}

impl fmt::Debug for Fr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let limbs = self.to_u64_limbs();
        write!(
            f,
            "Fr({:#018x}, {:#018x}, {:#018x}, {:#018x})",
            limbs[0], limbs[1], limbs[2], limbs[3]
        )
    }
}

/// A BN254 Fp field element (32 bytes, 8x u32 little-endian limbs, Montgomery form).
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Fp(pub [u8; 32]);

impl Fp {
    pub const ZERO: Self = Self([0u8; 32]);

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Debug for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp({:02x?})", &self.0[..8])
    }
}

/// A BN254 G1 affine point (x, y in Fp, Montgomery form).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct G1Affine {
    pub x: Fp,
    pub y: Fp,
}

/// A BN254 G1 projective point (x, y, z in Fp, Montgomery form).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct G1Projective {
    pub x: Fp,
    pub y: Fp,
    pub z: Fp,
}

/// A 256-bit scalar (standard form, NOT Montgomery).
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Scalar(pub [u8; 32]);

impl Scalar {
    pub const ZERO: Self = Self([0u8; 32]);

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn from_u64_limbs(limbs: [u64; 4]) -> Self {
        let mut bytes = [0u8; 32];
        for (i, limb) in limbs.iter().enumerate() {
            bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
        }
        Self(bytes)
    }
}

impl fmt::Debug for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Scalar({:02x?}...)", &self.0[..8])
    }
}

// ============================================================================
// MSM Engine
// ============================================================================

/// GPU-accelerated multi-scalar multiplication engine (BN254 G1).
///
/// Internally manages Metal GPU state. Thread-safe for sequential calls
/// (not concurrent — the GPU pipeline is serialized).
pub struct MsmEngine {
    handle: ffi::ZkMetalMSMEngine,
}

// The Swift engine manages its own synchronization
unsafe impl Send for MsmEngine {}

impl MsmEngine {
    /// Create a new MSM engine. Initializes the Metal GPU device and compiles shaders.
    pub fn new() -> Result<Self> {
        ensure_shader_dir();
        let mut handle = std::ptr::null_mut();
        check(unsafe { ffi::zkmetal_msm_engine_create(&mut handle) })?;
        Ok(Self { handle })
    }

    /// Compute MSM: result = sum(scalars[i] * points[i]).
    ///
    /// - `points`: Affine points on BN254 G1 (coordinates in Fp Montgomery form)
    /// - `scalars`: 256-bit scalars (standard form, NOT Montgomery)
    ///
    /// Returns the result as a projective point.
    pub fn msm(&self, points: &[G1Affine], scalars: &[Scalar]) -> Result<G1Projective> {
        if points.len() != scalars.len() || points.is_empty() {
            return Err(Error::InvalidInput);
        }

        let n = points.len() as u32;
        let points_ptr = points.as_ptr() as *const u8;
        let scalars_ptr = scalars.as_ptr() as *const u8;

        let mut result = G1Projective {
            x: Fp::ZERO,
            y: Fp::ZERO,
            z: Fp::ZERO,
        };

        check(unsafe {
            ffi::zkmetal_bn254_msm(
                self.handle,
                points_ptr,
                scalars_ptr,
                n,
                result.x.0.as_mut_ptr(),
                result.y.0.as_mut_ptr(),
                result.z.0.as_mut_ptr(),
            )
        })?;

        Ok(result)
    }
}

impl Drop for MsmEngine {
    fn drop(&mut self) {
        unsafe {
            ffi::zkmetal_msm_engine_destroy(self.handle);
        }
    }
}

// ============================================================================
// NTT Engine
// ============================================================================

/// GPU-accelerated NTT engine (BN254 Fr).
///
/// Performs forward and inverse Number Theoretic Transforms on arrays of
/// BN254 Fr field elements. Input size must be a power of 2.
pub struct NttEngine {
    handle: ffi::ZkMetalNTTEngine,
}

unsafe impl Send for NttEngine {}

impl NttEngine {
    /// Create a new NTT engine.
    pub fn new() -> Result<Self> {
        ensure_shader_dir();
        let mut handle = std::ptr::null_mut();
        check(unsafe { ffi::zkmetal_ntt_engine_create(&mut handle) })?;
        Ok(Self { handle })
    }

    /// Forward NTT in-place. `data.len()` must be a power of 2.
    /// Elements are BN254 Fr in Montgomery form (32 bytes each).
    pub fn ntt_inplace(&self, data: &mut [Fr]) -> Result<()> {
        let n = data.len();
        if n == 0 || !n.is_power_of_two() {
            return Err(Error::InvalidInput);
        }
        let log_n = n.trailing_zeros();
        check(unsafe {
            ffi::zkmetal_bn254_ntt(self.handle, data.as_mut_ptr() as *mut u8, log_n)
        })
    }

    /// Inverse NTT in-place. `data.len()` must be a power of 2.
    pub fn intt_inplace(&self, data: &mut [Fr]) -> Result<()> {
        let n = data.len();
        if n == 0 || !n.is_power_of_two() {
            return Err(Error::InvalidInput);
        }
        let log_n = n.trailing_zeros();
        check(unsafe {
            ffi::zkmetal_bn254_intt(self.handle, data.as_mut_ptr() as *mut u8, log_n)
        })
    }

    /// Forward NTT, returning a new vector.
    pub fn ntt(&self, data: &[Fr]) -> Result<Vec<Fr>> {
        let mut out = data.to_vec();
        self.ntt_inplace(&mut out)?;
        Ok(out)
    }

    /// Inverse NTT, returning a new vector.
    pub fn intt(&self, data: &[Fr]) -> Result<Vec<Fr>> {
        let mut out = data.to_vec();
        self.intt_inplace(&mut out)?;
        Ok(out)
    }
}

impl Drop for NttEngine {
    fn drop(&mut self) {
        unsafe {
            ffi::zkmetal_ntt_engine_destroy(self.handle);
        }
    }
}

// ============================================================================
// Poseidon2 Engine
// ============================================================================

/// GPU-accelerated Poseidon2 hash engine (BN254 Fr).
///
/// Batch-hashes pairs of field elements using the Poseidon2 permutation.
pub struct Poseidon2Engine {
    handle: ffi::ZkMetalPoseidon2Engine,
}

unsafe impl Send for Poseidon2Engine {}

impl Poseidon2Engine {
    /// Create a new Poseidon2 engine.
    pub fn new() -> Result<Self> {
        ensure_shader_dir();
        let mut handle = std::ptr::null_mut();
        check(unsafe { ffi::zkmetal_poseidon2_engine_create(&mut handle) })?;
        Ok(Self { handle })
    }

    /// Hash pairs of field elements: input[2*i] and input[2*i+1] produce output[i].
    ///
    /// `input.len()` must be even and non-zero.
    pub fn hash_pairs(&self, input: &[Fr]) -> Result<Vec<Fr>> {
        if input.is_empty() || input.len() % 2 != 0 {
            return Err(Error::InvalidInput);
        }
        let n_pairs = (input.len() / 2) as u32;
        let mut output = vec![Fr::ZERO; n_pairs as usize];

        check(unsafe {
            ffi::zkmetal_bn254_poseidon2_hash_pairs(
                self.handle,
                input.as_ptr() as *const u8,
                n_pairs,
                output.as_mut_ptr() as *mut u8,
            )
        })?;

        Ok(output)
    }
}

impl Drop for Poseidon2Engine {
    fn drop(&mut self) {
        unsafe {
            ffi::zkmetal_poseidon2_engine_destroy(self.handle);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_available() {
        let avail = gpu_available();
        println!("GPU available: {}", avail);
        // On Apple Silicon Mac, this should be true
        assert!(avail, "Expected Metal GPU to be available");
    }

    #[test]
    fn test_version() {
        let v = version();
        assert_eq!(v, "0.1.0");
    }

    #[test]
    fn test_ntt_roundtrip() {
        let ntt = NttEngine::new().expect("NTT engine creation failed");
        let n = 1024; // 2^10

        // Create test data: all zeros (valid Montgomery Fr elements)
        let original = vec![Fr::ZERO; n];
        let mut data = original.clone();

        // Forward NTT
        ntt.ntt_inplace(&mut data).expect("NTT forward failed");

        // Inverse NTT should recover original
        ntt.intt_inplace(&mut data).expect("NTT inverse failed");

        assert_eq!(data, original, "NTT roundtrip should recover original data");
    }

    #[test]
    fn test_ntt_nonzero_roundtrip() {
        let ntt = NttEngine::new().expect("NTT engine creation failed");
        let n = 256; // 2^8

        // Create data with Montgomery form of 1 (R mod r)
        // R mod r = [0xac96341c4ffffffb, 0x36fc76959f60cd29, 0x666ea36f7879462e, 0x0e0a77c19a07df2f]
        let one_mont = Fr::from_u64_limbs([
            0xac96341c4ffffffb,
            0x36fc76959f60cd29,
            0x666ea36f7879462e,
            0x0e0a77c19a07df2f,
        ]);

        let original = vec![one_mont; n];
        let mut data = original.clone();

        ntt.ntt_inplace(&mut data).expect("NTT forward failed");

        // After NTT, data should differ from original (except for constant polynomial
        // where all values are the same — NTT of constant is n*value at index 0, 0 elsewhere)
        // Actually for a constant polynomial, NTT produces [n*c, 0, 0, ...] which IS different.

        ntt.intt_inplace(&mut data).expect("NTT inverse failed");
        assert_eq!(
            data, original,
            "NTT roundtrip with nonzero data should recover original"
        );
    }

    #[test]
    fn test_poseidon2_basic() {
        let p2 = Poseidon2Engine::new().expect("Poseidon2 engine creation failed");

        // Hash a single pair of zero elements
        let input = vec![Fr::ZERO; 2];
        let output = p2.hash_pairs(&input).expect("Poseidon2 hash failed");

        assert_eq!(output.len(), 1);
        // The hash of (0, 0) should be deterministic and non-zero
        println!("Poseidon2(0, 0) = {:?}", output[0]);
    }

    #[test]
    fn test_poseidon2_batch() {
        let p2 = Poseidon2Engine::new().expect("Poseidon2 engine creation failed");

        // Hash 1024 pairs
        let input = vec![Fr::ZERO; 2048];
        let output = p2.hash_pairs(&input).expect("Poseidon2 batch hash failed");
        assert_eq!(output.len(), 1024);

        // All pairs are identical, so all outputs should be the same
        for h in &output[1..] {
            assert_eq!(h, &output[0], "Identical inputs should produce identical hashes");
        }
    }

    #[test]
    fn test_msm_engine_create() {
        let _msm = MsmEngine::new().expect("MSM engine creation failed");
    }

    #[test]
    fn test_msm_empty_input() {
        let msm = MsmEngine::new().expect("MSM engine creation failed");
        let result = msm.msm(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_fr_conversions() {
        let limbs = [0x1234u64, 0x5678u64, 0x9abcu64, 0xdef0u64];
        let fr = Fr::from_u64_limbs(limbs);
        let recovered = fr.to_u64_limbs();
        assert_eq!(limbs, recovered);
    }

    #[test]
    fn test_error_display() {
        assert_eq!(format!("{}", Error::NoGpu), "No Metal GPU available");
        assert_eq!(format!("{}", Error::InvalidInput), "Invalid input");
    }
}
