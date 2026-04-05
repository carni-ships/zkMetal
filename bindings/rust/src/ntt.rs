//! Number Theoretic Transform (NTT) safe wrappers.
//!
//! ## GPU NTT (zkmetal.h, `gpu` feature)
//!
//! Operates on byte slices: `n * 32` bytes (n = 2^log_n), each element is
//! 32 bytes in Montgomery form (8 x u32, little-endian).
//!
//! ## CPU NTT (NeonFieldOps, `neon` feature)
//!
//! BN254 Fr: `n * 4` u64 values (4 limbs per element, Montgomery form).
//! BabyBear: `n` u32 values in [0, p).
//! Goldilocks: `n` u64 values in [0, p).

use crate::{check_status, Result};

// ============================================================================
// GPU NTT Engine (RAII)
// ============================================================================

/// BN254 NTT GPU engine handle. Dropped automatically.
#[cfg(feature = "gpu")]
pub struct NttEngine {
    raw: *mut std::ffi::c_void,
}

#[cfg(feature = "gpu")]
unsafe impl Send for NttEngine {}

#[cfg(feature = "gpu")]
impl NttEngine {
    /// Create a new NTT engine (compiles GPU shaders, allocates resources).
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check_status(unsafe { crate::ffi::zkmetal_ntt_engine_create(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Forward NTT in-place. `data` must be `2^log_n * 32` bytes.
    pub fn ntt(&self, data: &mut [u8], log_n: u32) -> Result<()> {
        assert_eq!(data.len(), (1usize << log_n) * 32);
        check_status(unsafe { crate::ffi::zkmetal_bn254_ntt(self.raw, data.as_mut_ptr(), log_n) })
    }

    /// Inverse NTT in-place. `data` must be `2^log_n * 32` bytes.
    pub fn intt(&self, data: &mut [u8], log_n: u32) -> Result<()> {
        assert_eq!(data.len(), (1usize << log_n) * 32);
        check_status(unsafe { crate::ffi::zkmetal_bn254_intt(self.raw, data.as_mut_ptr(), log_n) })
    }
}

#[cfg(feature = "gpu")]
impl Drop for NttEngine {
    fn drop(&mut self) {
        unsafe { crate::ffi::zkmetal_ntt_engine_destroy(self.raw) }
    }
}

// ============================================================================
// Convenience GPU NTT (lazy singleton)
// ============================================================================

/// Convenience forward NTT in-place (GPU, lazy singleton).
/// `data` must be `2^log_n * 32` bytes (BN254 Fr elements in Montgomery form).
#[cfg(feature = "gpu")]
pub fn bn254_ntt_auto(data: &mut [u8], log_n: u32) -> Result<()> {
    assert_eq!(data.len(), (1usize << log_n) * 32);
    check_status(unsafe { crate::ffi::zkmetal_bn254_ntt_auto(data.as_mut_ptr(), log_n) })
}

/// Convenience inverse NTT in-place (GPU, lazy singleton).
/// `data` must be `2^log_n * 32` bytes.
#[cfg(feature = "gpu")]
pub fn bn254_intt_auto(data: &mut [u8], log_n: u32) -> Result<()> {
    assert_eq!(data.len(), (1usize << log_n) * 32);
    check_status(unsafe { crate::ffi::zkmetal_bn254_intt_auto(data.as_mut_ptr(), log_n) })
}

// ============================================================================
// CPU NTT via NeonFieldOps
// ============================================================================

#[cfg(feature = "neon")]
mod neon_ffi {
    extern "C" {
        // BN254 Fr NTT (4 x u64 per element, Montgomery form)
        pub fn bn254_fr_ntt(data: *mut u64, log_n: libc::c_int);
        pub fn bn254_fr_intt(data: *mut u64, log_n: libc::c_int);

        // BabyBear NTT (u32 elements)
        pub fn babybear_ntt_neon(data: *mut u32, log_n: libc::c_int);
        pub fn babybear_intt_neon(data: *mut u32, log_n: libc::c_int);

        // Goldilocks NTT (u64 elements)
        pub fn goldilocks_ntt(data: *mut u64, log_n: libc::c_int);
        pub fn goldilocks_intt(data: *mut u64, log_n: libc::c_int);
        pub fn goldilocks_ntt_neon(data: *mut u64, log_n: libc::c_int);
        pub fn goldilocks_intt_neon(data: *mut u64, log_n: libc::c_int);

        // BLS12-377 Fr NTT
        pub fn bls12_377_fr_ntt(data: *mut u64, log_n: libc::c_int);
        pub fn bls12_377_fr_intt(data: *mut u64, log_n: libc::c_int);

        // Stark252 NTT
        pub fn stark252_ntt(data: *mut u64, log_n: libc::c_int);
        pub fn stark252_intt(data: *mut u64, log_n: libc::c_int);
    }
}

/// CPU forward NTT for BN254 Fr.
/// `data` must be `2^log_n` elements, each 4 x u64 (Montgomery form).
/// i.e., `data.len() == 2^log_n * 4`.
#[cfg(feature = "neon")]
pub fn bn254_fr_ntt_cpu(data: &mut [u64], log_n: u32) {
    assert_eq!(data.len(), (1usize << log_n) * 4);
    unsafe { neon_ffi::bn254_fr_ntt(data.as_mut_ptr(), log_n as libc::c_int) };
}

/// CPU inverse NTT for BN254 Fr.
#[cfg(feature = "neon")]
pub fn bn254_fr_intt_cpu(data: &mut [u64], log_n: u32) {
    assert_eq!(data.len(), (1usize << log_n) * 4);
    unsafe { neon_ffi::bn254_fr_intt(data.as_mut_ptr(), log_n as libc::c_int) };
}

/// CPU forward NTT for BabyBear field (p = 0x78000001). NEON-accelerated.
/// `data` must have `2^log_n` u32 elements in [0, p).
#[cfg(feature = "neon")]
pub fn babybear_ntt(data: &mut [u32], log_n: u32) {
    assert_eq!(data.len(), 1usize << log_n);
    unsafe { neon_ffi::babybear_ntt_neon(data.as_mut_ptr(), log_n as libc::c_int) };
}

/// CPU inverse NTT for BabyBear field. NEON-accelerated.
#[cfg(feature = "neon")]
pub fn babybear_intt(data: &mut [u32], log_n: u32) {
    assert_eq!(data.len(), 1usize << log_n);
    unsafe { neon_ffi::babybear_intt_neon(data.as_mut_ptr(), log_n as libc::c_int) };
}

/// CPU forward NTT for Goldilocks field (p = 2^64 - 2^32 + 1). Scalar version.
/// `data` must have `2^log_n` u64 elements in [0, p).
#[cfg(feature = "neon")]
pub fn goldilocks_ntt_scalar(data: &mut [u64], log_n: u32) {
    assert_eq!(data.len(), 1usize << log_n);
    unsafe { neon_ffi::goldilocks_ntt(data.as_mut_ptr(), log_n as libc::c_int) };
}

/// CPU inverse NTT for Goldilocks field. Scalar version.
#[cfg(feature = "neon")]
pub fn goldilocks_intt_scalar(data: &mut [u64], log_n: u32) {
    assert_eq!(data.len(), 1usize << log_n);
    unsafe { neon_ffi::goldilocks_intt(data.as_mut_ptr(), log_n as libc::c_int) };
}

/// CPU forward NTT for Goldilocks field. NEON-vectorized version.
#[cfg(feature = "neon")]
pub fn goldilocks_ntt_neon(data: &mut [u64], log_n: u32) {
    assert_eq!(data.len(), 1usize << log_n);
    unsafe { neon_ffi::goldilocks_ntt_neon(data.as_mut_ptr(), log_n as libc::c_int) };
}

/// CPU inverse NTT for Goldilocks field. NEON-vectorized version.
#[cfg(feature = "neon")]
pub fn goldilocks_intt_neon(data: &mut [u64], log_n: u32) {
    assert_eq!(data.len(), 1usize << log_n);
    unsafe { neon_ffi::goldilocks_intt_neon(data.as_mut_ptr(), log_n as libc::c_int) };
}

/// CPU forward NTT for BLS12-377 Fr. 4 x u64 per element, Montgomery form.
#[cfg(feature = "neon")]
pub fn bls12_377_fr_ntt(data: &mut [u64], log_n: u32) {
    assert_eq!(data.len(), (1usize << log_n) * 4);
    unsafe { neon_ffi::bls12_377_fr_ntt(data.as_mut_ptr(), log_n as libc::c_int) };
}

/// CPU inverse NTT for BLS12-377 Fr.
#[cfg(feature = "neon")]
pub fn bls12_377_fr_intt(data: &mut [u64], log_n: u32) {
    assert_eq!(data.len(), (1usize << log_n) * 4);
    unsafe { neon_ffi::bls12_377_fr_intt(data.as_mut_ptr(), log_n as libc::c_int) };
}

/// CPU forward NTT for Stark252 field (p = 2^251 + 17*2^192 + 1).
/// 4 x u64 per element, Montgomery form.
#[cfg(feature = "neon")]
pub fn stark252_ntt(data: &mut [u64], log_n: u32) {
    assert_eq!(data.len(), (1usize << log_n) * 4);
    unsafe { neon_ffi::stark252_ntt(data.as_mut_ptr(), log_n as libc::c_int) };
}

/// CPU inverse NTT for Stark252 field.
#[cfg(feature = "neon")]
pub fn stark252_intt(data: &mut [u64], log_n: u32) {
    assert_eq!(data.len(), (1usize << log_n) * 4);
    unsafe { neon_ffi::stark252_intt(data.as_mut_ptr(), log_n as libc::c_int) };
}
