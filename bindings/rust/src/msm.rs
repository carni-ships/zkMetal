//! Multi-Scalar Multiplication (MSM) safe wrappers.
//!
//! Provides both GPU-accelerated MSM (via Metal, `gpu` feature) and
//! CPU Pippenger MSM (via NeonFieldOps, `neon` feature).
//!
//! ## GPU MSM (zkmetal.h)
//!
//! Points are affine: 64 bytes (x: 32B, y: 32B) in Montgomery form.
//! Scalars are 32 bytes (8 x u32 limbs, standard form, NOT Montgomery).
//! Results are projective: 3 x 32 bytes (X, Y, Z) in Montgomery form.
//!
//! ## CPU MSM (NeonFieldOps)
//!
//! Points are affine: 8 x u64 (x[4], y[4]) in Montgomery form.
//! Scalars: 8 x u32 limbs (little-endian integer form).
//! Results are projective: 12 x u64 (x[4], y[4], z[4]).

use crate::{check_status, Result};

// ============================================================================
// GPU types and engine (zkmetal.h)
// ============================================================================

/// BN254 G1 affine point for GPU MSM. 64 bytes: x (32B) || y (32B), Montgomery form.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct G1Affine {
    pub x: [u8; 32],
    pub y: [u8; 32],
}

/// BN254 G1 projective point result from GPU MSM. 3 x 32 bytes, Montgomery form.
#[derive(Clone, Copy, Debug)]
pub struct G1Projective {
    pub x: [u8; 32],
    pub y: [u8; 32],
    pub z: [u8; 32],
}

/// BN254 scalar for GPU MSM. 32 bytes (8 x u32, little-endian, standard form).
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Scalar256(pub [u8; 32]);

// ============================================================================
// GPU MSM Engine (RAII)
// ============================================================================

/// BN254 MSM GPU engine handle. Dropped automatically.
#[cfg(feature = "gpu")]
pub struct MsmEngine {
    raw: *mut std::ffi::c_void,
}

#[cfg(feature = "gpu")]
unsafe impl Send for MsmEngine {}

#[cfg(feature = "gpu")]
impl MsmEngine {
    /// Create a new MSM engine (compiles GPU shaders, allocates resources).
    pub fn new() -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        check_status(unsafe { crate::ffi::zkmetal_msm_engine_create(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Compute MSM: result = sum(scalars[i] * points[i]).
    ///
    /// - `points`: `n * 64` bytes -- affine points in Montgomery form
    /// - `scalars`: `n * 32` bytes -- scalars in standard form (8 x u32 LE)
    ///
    /// Returns `(x, y, z)` projective coordinates, each 32 bytes in Montgomery form.
    pub fn msm(&self, points: &[u8], scalars: &[u8], n: u32) -> Result<G1Projective> {
        assert_eq!(points.len(), n as usize * 64, "points must be n*64 bytes");
        assert_eq!(scalars.len(), n as usize * 32, "scalars must be n*32 bytes");
        let mut result = G1Projective { x: [0; 32], y: [0; 32], z: [0; 32] };
        check_status(unsafe {
            crate::ffi::zkmetal_bn254_msm(
                self.raw, points.as_ptr(), scalars.as_ptr(), n,
                result.x.as_mut_ptr(), result.y.as_mut_ptr(), result.z.as_mut_ptr(),
            )
        })?;
        Ok(result)
    }

    /// MSM with 1-byte scalars (u8). More efficient for small scalar ranges.
    pub fn msm_u8(&self, points: &[u8], scalars: &[u8], n: u32) -> Result<G1Projective> {
        assert_eq!(points.len(), n as usize * 64);
        assert_eq!(scalars.len(), n as usize);
        let mut result = G1Projective { x: [0; 32], y: [0; 32], z: [0; 32] };
        check_status(unsafe {
            crate::ffi::zkmetal_bn254_msm_u8(
                self.raw, points.as_ptr(), scalars.as_ptr(), n,
                result.x.as_mut_ptr(), result.y.as_mut_ptr(), result.z.as_mut_ptr(),
            )
        })?;
        Ok(result)
    }

    /// MSM with 2-byte scalars (u16, little-endian).
    pub fn msm_u16(&self, points: &[u8], scalars: &[u8], n: u32) -> Result<G1Projective> {
        assert_eq!(points.len(), n as usize * 64);
        assert_eq!(scalars.len(), n as usize * 2);
        let mut result = G1Projective { x: [0; 32], y: [0; 32], z: [0; 32] };
        check_status(unsafe {
            crate::ffi::zkmetal_bn254_msm_u16(
                self.raw, points.as_ptr(), scalars.as_ptr(), n,
                result.x.as_mut_ptr(), result.y.as_mut_ptr(), result.z.as_mut_ptr(),
            )
        })?;
        Ok(result)
    }

    /// MSM with 4-byte scalars (u32, little-endian).
    pub fn msm_u32(&self, points: &[u8], scalars: &[u8], n: u32) -> Result<G1Projective> {
        assert_eq!(points.len(), n as usize * 64);
        assert_eq!(scalars.len(), n as usize * 4);
        let mut result = G1Projective { x: [0; 32], y: [0; 32], z: [0; 32] };
        check_status(unsafe {
            crate::ffi::zkmetal_bn254_msm_u32(
                self.raw, points.as_ptr(), scalars.as_ptr(), n,
                result.x.as_mut_ptr(), result.y.as_mut_ptr(), result.z.as_mut_ptr(),
            )
        })?;
        Ok(result)
    }

    /// Batch MSM: compute multiple independent MSMs in one call.
    /// Amortizes Swift/GPU dispatch overhead for many small MSMs.
    ///
    /// - `all_points`: concatenated affine points for all MSMs (64 bytes each)
    /// - `all_scalars`: concatenated 32-byte scalars for all MSMs
    /// - `counts`: number of points in each MSM
    ///
    /// Returns `n_msms * 96` bytes -- projective (X,Y,Z) per MSM, 32 bytes each.
    pub fn msm_batch(
        &self,
        all_points: &[u8],
        all_scalars: &[u8],
        counts: &[u32],
    ) -> Result<Vec<u8>> {
        let n_msms = counts.len() as u32;
        let total_points: u32 = counts.iter().sum();
        assert_eq!(all_points.len(), total_points as usize * 64);
        assert_eq!(all_scalars.len(), total_points as usize * 32);
        let mut results = vec![0u8; counts.len() * 96];
        check_status(unsafe {
            crate::ffi::zkmetal_bn254_msm_batch(
                self.raw,
                all_points.as_ptr(), all_scalars.as_ptr(),
                counts.as_ptr(), n_msms, results.as_mut_ptr(),
            )
        })?;
        Ok(results)
    }
}

#[cfg(feature = "gpu")]
impl Drop for MsmEngine {
    fn drop(&mut self) {
        unsafe { crate::ffi::zkmetal_msm_engine_destroy(self.raw) }
    }
}

// ============================================================================
// Convenience (auto) GPU MSM API
// ============================================================================

/// Convenience MSM using a lazy singleton engine (GPU).
///
/// - `points`: `n * 64` bytes (affine, Montgomery form)
/// - `scalars`: `n * 32` bytes (standard form, 8 x u32 LE)
#[cfg(feature = "gpu")]
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
        crate::ffi::zkmetal_bn254_msm_auto(
            points.as_ptr(), scalars.as_ptr(), n,
            x.as_mut_ptr(), y.as_mut_ptr(), z.as_mut_ptr(),
        )
    })?;
    Ok((x, y, z))
}

/// Convenience MSM with u8 scalars (GPU, lazy singleton).
#[cfg(feature = "gpu")]
pub fn bn254_msm_u8_auto(
    points: &[u8], scalars: &[u8], n: u32,
) -> Result<([u8; 32], [u8; 32], [u8; 32])> {
    assert_eq!(points.len(), n as usize * 64);
    assert_eq!(scalars.len(), n as usize);
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    let mut z = [0u8; 32];
    check_status(unsafe {
        crate::ffi::zkmetal_bn254_msm_u8_auto(
            points.as_ptr(), scalars.as_ptr(), n,
            x.as_mut_ptr(), y.as_mut_ptr(), z.as_mut_ptr(),
        )
    })?;
    Ok((x, y, z))
}

/// Convenience MSM with u16 scalars (GPU, lazy singleton).
#[cfg(feature = "gpu")]
pub fn bn254_msm_u16_auto(
    points: &[u8], scalars: &[u8], n: u32,
) -> Result<([u8; 32], [u8; 32], [u8; 32])> {
    assert_eq!(points.len(), n as usize * 64);
    assert_eq!(scalars.len(), n as usize * 2);
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    let mut z = [0u8; 32];
    check_status(unsafe {
        crate::ffi::zkmetal_bn254_msm_u16_auto(
            points.as_ptr(), scalars.as_ptr(), n,
            x.as_mut_ptr(), y.as_mut_ptr(), z.as_mut_ptr(),
        )
    })?;
    Ok((x, y, z))
}

/// Convenience MSM with u32 scalars (GPU, lazy singleton).
#[cfg(feature = "gpu")]
pub fn bn254_msm_u32_auto(
    points: &[u8], scalars: &[u8], n: u32,
) -> Result<([u8; 32], [u8; 32], [u8; 32])> {
    assert_eq!(points.len(), n as usize * 64);
    assert_eq!(scalars.len(), n as usize * 4);
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    let mut z = [0u8; 32];
    check_status(unsafe {
        crate::ffi::zkmetal_bn254_msm_u32_auto(
            points.as_ptr(), scalars.as_ptr(), n,
            x.as_mut_ptr(), y.as_mut_ptr(), z.as_mut_ptr(),
        )
    })?;
    Ok((x, y, z))
}

/// Convenience batch MSM (GPU, lazy singleton).
#[cfg(feature = "gpu")]
pub fn bn254_msm_batch_auto(
    all_points: &[u8],
    all_scalars: &[u8],
    counts: &[u32],
) -> Result<Vec<u8>> {
    let n_msms = counts.len() as u32;
    let total_points: u32 = counts.iter().sum();
    assert_eq!(all_points.len(), total_points as usize * 64);
    assert_eq!(all_scalars.len(), total_points as usize * 32);
    let mut results = vec![0u8; counts.len() * 96];
    check_status(unsafe {
        crate::ffi::zkmetal_bn254_msm_batch_auto(
            all_points.as_ptr(), all_scalars.as_ptr(),
            counts.as_ptr(), n_msms, results.as_mut_ptr(),
        )
    })?;
    Ok(results)
}

// ============================================================================
// CPU MSM via NeonFieldOps
// ============================================================================

/// CPU projective point: 12 x u64 (x[4], y[4], z[4]) in Montgomery form.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ProjectivePoint {
    pub coords: [u64; 12],
}

impl ProjectivePoint {
    pub const ZERO: Self = Self { coords: [0; 12] };
}

#[cfg(feature = "neon")]
mod neon_ffi {
    extern "C" {
        pub fn bn254_pippenger_msm(
            points: *const u64, scalars: *const u32, n: libc::c_int, result: *mut u64,
        );
        pub fn bn254_point_add(p: *const u64, q: *const u64, r: *mut u64);
        pub fn bn254_point_add_mixed(p: *const u64, q_aff: *const u64, r: *mut u64);
        pub fn bn254_point_scalar_mul(p: *const u64, scalar: *const u32, r: *mut u64);
        pub fn bn254_projective_to_affine(p: *const u64, affine: *mut u64);
        pub fn bn254_batch_to_affine(proj: *const u64, aff: *mut u64, n: libc::c_int);
        pub fn bn254_msm_projective(
            points: *const u64, scalars: *const u32, n: libc::c_int, result: *mut u64,
        );
    }
}

/// CPU Pippenger MSM for BN254 G1.
///
/// - `points`: n affine points as `n * 8` u64 values (x[4], y[4] per point, Montgomery form)
/// - `scalars`: n scalars as `n * 8` u32 values (little-endian limbs)
///
/// Returns a projective point (12 x u64).
#[cfg(feature = "neon")]
pub fn bn254_pippenger_msm_cpu(points: &[u64], scalars: &[u32]) -> ProjectivePoint {
    let n_points = points.len() / 8;
    assert_eq!(points.len(), n_points * 8, "points must be n*8 u64");
    assert_eq!(scalars.len(), n_points * 8, "scalars must be n*8 u32");
    let mut result = ProjectivePoint::ZERO;
    unsafe {
        neon_ffi::bn254_pippenger_msm(
            points.as_ptr(), scalars.as_ptr(), n_points as libc::c_int,
            result.coords.as_mut_ptr(),
        );
    }
    result
}

/// CPU point addition (projective).
#[cfg(feature = "neon")]
pub fn bn254_point_add_cpu(p: &ProjectivePoint, q: &ProjectivePoint) -> ProjectivePoint {
    let mut r = ProjectivePoint::ZERO;
    unsafe { neon_ffi::bn254_point_add(p.coords.as_ptr(), q.coords.as_ptr(), r.coords.as_mut_ptr()) };
    r
}

/// CPU mixed addition: projective P + affine Q.
#[cfg(feature = "neon")]
pub fn bn254_point_add_mixed_cpu(p: &ProjectivePoint, q_aff: &[u64; 8]) -> ProjectivePoint {
    let mut r = ProjectivePoint::ZERO;
    unsafe { neon_ffi::bn254_point_add_mixed(p.coords.as_ptr(), q_aff.as_ptr(), r.coords.as_mut_ptr()) };
    r
}

/// CPU scalar multiplication.
#[cfg(feature = "neon")]
pub fn bn254_scalar_mul_cpu(p: &ProjectivePoint, scalar: &[u32; 8]) -> ProjectivePoint {
    let mut r = ProjectivePoint::ZERO;
    unsafe { neon_ffi::bn254_point_scalar_mul(p.coords.as_ptr(), scalar.as_ptr(), r.coords.as_mut_ptr()) };
    r
}

/// CPU projective to affine conversion.
#[cfg(feature = "neon")]
pub fn bn254_to_affine_cpu(p: &ProjectivePoint) -> [u64; 8] {
    let mut aff = [0u64; 8];
    unsafe { neon_ffi::bn254_projective_to_affine(p.coords.as_ptr(), aff.as_mut_ptr()) };
    aff
}

/// CPU batch projective to affine conversion (Montgomery's trick).
#[cfg(feature = "neon")]
pub fn bn254_batch_to_affine_cpu(proj: &[ProjectivePoint]) -> Vec<[u64; 8]> {
    let n = proj.len();
    let mut aff = vec![[0u64; 8]; n];
    unsafe {
        neon_ffi::bn254_batch_to_affine(
            proj.as_ptr() as *const u64,
            aff.as_mut_ptr() as *mut u64,
            n as libc::c_int,
        );
    }
    aff
}
