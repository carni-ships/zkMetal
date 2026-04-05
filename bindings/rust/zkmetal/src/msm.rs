//! Multi-scalar multiplication (Pippenger) for supported curves.
//!
//! All MSM functions use the Pippenger algorithm with multi-threaded bucket
//! accumulation and Montgomery's batch-to-affine trick.

use crate::bn254;
use crate::bls12_381;

/// BN254 G1 Pippenger MSM.
///
/// Computes sum(scalars[i] * points[i]) for i in 0..n.
///
/// - `points`: n affine points, each 8 x u64 (x[4], y[4]) in Montgomery Fq.
/// - `scalars`: n scalars, each 8 x u32 in non-Montgomery integer form.
/// - Returns: projective result point.
///
/// # Panics
/// Panics if `scalars.len() != points.len() * 2` (each scalar is 8 u32 = 2 * 4 u64).
pub fn bn254_g1_msm(
    points: &[bn254::G1Affine],
    scalars: &[[u32; 8]],
) -> bn254::G1Projective {
    assert_eq!(points.len(), scalars.len(), "msm: points/scalars length mismatch");
    let mut result = bn254::G1Projective::IDENTITY;
    unsafe {
        zkmetal_sys::bn254_pippenger_msm(
            points.as_ptr() as *const u64,
            scalars.as_ptr() as *const u32,
            points.len() as core::ffi::c_int,
            &mut result as *mut _ as *mut u64,
        );
    }
    result
}

/// BN254 G1 MSM from raw slices (for interop with other libraries).
///
/// - `points_raw`: flat array of n * 8 u64 values.
/// - `scalars_raw`: flat array of n * 8 u32 values.
/// - `n`: number of points.
pub fn bn254_g1_msm_raw(
    points_raw: &[u64],
    scalars_raw: &[u32],
    n: usize,
) -> bn254::G1Projective {
    assert!(points_raw.len() >= n * 8, "msm: points buffer too small");
    assert!(scalars_raw.len() >= n * 8, "msm: scalars buffer too small");
    let mut result = bn254::G1Projective::IDENTITY;
    unsafe {
        zkmetal_sys::bn254_pippenger_msm(
            points_raw.as_ptr(),
            scalars_raw.as_ptr(),
            n as core::ffi::c_int,
            &mut result as *mut _ as *mut u64,
        );
    }
    result
}

/// BLS12-381 G1 Pippenger MSM.
///
/// - `points`: flat array of n * 12 u64 (affine, x[6] y[6] per point).
/// - `scalars`: flat array of n * 8 u32 (integer form).
/// - `n`: number of points.
pub fn bls12_381_g1_msm_raw(
    points: &[u64],
    scalars: &[u32],
    n: usize,
) -> bls12_381::G1Projective {
    assert!(points.len() >= n * 12, "msm: points buffer too small");
    assert!(scalars.len() >= n * 8, "msm: scalars buffer too small");
    let mut result = bls12_381::G1Projective::IDENTITY;
    unsafe {
        zkmetal_sys::bls12_381_g1_pippenger_msm(
            points.as_ptr(),
            scalars.as_ptr(),
            n as core::ffi::c_int,
            &mut result as *mut _ as *mut u64,
        );
    }
    result
}

/// Grumpkin Pippenger MSM (raw interface).
pub fn grumpkin_msm_raw(
    points: &[u64],
    scalars: &[u32],
    n: usize,
) -> [u64; 12] {
    assert!(points.len() >= n * 8);
    assert!(scalars.len() >= n * 8);
    let mut result = [0u64; 12];
    unsafe {
        zkmetal_sys::grumpkin_pippenger_msm(
            points.as_ptr(),
            scalars.as_ptr(),
            n as core::ffi::c_int,
            result.as_mut_ptr(),
        );
    }
    result
}

/// Pallas Pippenger MSM (raw interface).
pub fn pallas_msm_raw(
    points: &[u64],
    scalars: &[u32],
    n: usize,
) -> [u64; 12] {
    assert!(points.len() >= n * 8);
    assert!(scalars.len() >= n * 8);
    let mut result = [0u64; 12];
    unsafe {
        zkmetal_sys::pallas_pippenger_msm(
            points.as_ptr(),
            scalars.as_ptr(),
            n as core::ffi::c_int,
            result.as_mut_ptr(),
        );
    }
    result
}

/// Vesta Pippenger MSM (raw interface).
pub fn vesta_msm_raw(
    points: &[u64],
    scalars: &[u32],
    n: usize,
) -> [u64; 12] {
    assert!(points.len() >= n * 8);
    assert!(scalars.len() >= n * 8);
    let mut result = [0u64; 12];
    unsafe {
        zkmetal_sys::vesta_pippenger_msm(
            points.as_ptr(),
            scalars.as_ptr(),
            n as core::ffi::c_int,
            result.as_mut_ptr(),
        );
    }
    result
}

/// secp256k1 Pippenger MSM (raw interface).
pub fn secp256k1_msm_raw(
    points: &[u64],
    scalars: &[u32],
    n: usize,
) -> [u64; 12] {
    assert!(points.len() >= n * 8);
    assert!(scalars.len() >= n * 8);
    let mut result = [0u64; 12];
    unsafe {
        zkmetal_sys::secp256k1_pippenger_msm(
            points.as_ptr(),
            scalars.as_ptr(),
            n as core::ffi::c_int,
            result.as_mut_ptr(),
        );
    }
    result
}
