//! Additional NeonFieldOps bindings for curves beyond BN254.
//!
//! This module exposes safe wrappers for CPU-accelerated operations on:
//! - BLS12-381 (Fr, Fp, G1, G2, Fp2, pairing)
//! - BLS12-377 (Fq, Fr, G1)
//! - Pallas / Vesta (Fp, point ops, MSM)
//! - Grumpkin (point ops, MSM)
//! - secp256k1 (point ops, MSM, ECDSA batch verify)
//! - Ed25519 (Fp, Fq, point ops, MSM, EdDSA)
//! - Goldilocks batch ops
//! - BabyJubjub / Jubjub
//! - Stark252
//! - Binary tower fields (GF(2^k) for Binius)
//!
//! Requires the `neon` feature.

// ============================================================================
// 4-limb field element (256-bit: BN254, BLS12-381 Fr, Pallas, Vesta, etc.)
// ============================================================================

/// Generic 4-limb field element (256-bit, Montgomery form).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Fp4(pub [u64; 4]);

/// Generic 6-limb field element (384-bit, Montgomery form: BLS12-381 Fp, BLS12-377 Fq).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Fp6(pub [u64; 6]);

/// Projective point with 4-limb coordinates (12 x u64: x[4], y[4], z[4]).
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Proj4(pub [u64; 12]);

/// Projective point with 6-limb coordinates (18 x u64: x[6], y[6], z[6]).
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Proj6(pub [u64; 18]);

/// Extended point (4 coordinates, 4 limbs each = 16 x u64: X,Y,Z,T).
/// Used for twisted Edwards curves (Ed25519, BabyJubjub, Jubjub).
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct ExtendedPoint(pub [u64; 16]);

// ============================================================================
// Raw FFI
// ============================================================================

extern "C" {
    // -- BLS12-381 Fr --
    fn bls12_381_fr_add(a: *const u64, b: *const u64, r: *mut u64);
    fn bls12_381_fr_sub(a: *const u64, b: *const u64, r: *mut u64);
    fn bls12_381_fr_mul(a: *const u64, b: *const u64, r: *mut u64);
    fn bls12_381_fr_neg(a: *const u64, r: *mut u64);
    fn bls12_381_fr_sqr(a: *const u64, r: *mut u64);

    // -- BLS12-381 Fp (6-limb) --
    fn bls12_381_fp_add(a: *const u64, b: *const u64, r: *mut u64);
    fn bls12_381_fp_sub(a: *const u64, b: *const u64, r: *mut u64);
    fn bls12_381_fp_mul(a: *const u64, b: *const u64, r: *mut u64);
    fn bls12_381_fp_neg(a: *const u64, r: *mut u64);
    fn bls12_381_fp_sqr(a: *const u64, r: *mut u64);
    fn bls12_381_fp_inv_ext(a: *const u64, r: *mut u64);
    fn bls12_381_fp_sqrt(a: *const u64, r: *mut u64) -> libc::c_int;

    // -- BLS12-381 G1 --
    fn bls12_381_g1_point_add(p: *const u64, q: *const u64, r: *mut u64);
    fn bls12_381_g1_point_double(p: *const u64, r: *mut u64);
    fn bls12_381_g1_point_add_mixed(p: *const u64, q_aff: *const u64, r: *mut u64);
    fn bls12_381_g1_scalar_mul(p: *const u64, scalar: *const u64, r: *mut u64);
    fn bls12_381_g1_pippenger_msm(points: *const u64, scalars: *const u32, n: libc::c_int, result: *mut u64);

    // -- BLS12-381 Pairing --
    fn bls12_381_miller_loop(p_aff: *const u64, q_aff: *const u64, result: *mut u64);
    fn bls12_381_final_exp(f: *const u64, result: *mut u64);
    fn bls12_381_pairing(p_aff: *const u64, q_aff: *const u64, result: *mut u64);
    fn bls12_381_pairing_check(pairs: *const u64, n: libc::c_int) -> libc::c_int;

    // -- Pallas --
    fn pallas_fp_add(a: *const u64, b: *const u64, r: *mut u64);
    fn pallas_fp_sub(a: *const u64, b: *const u64, r: *mut u64);
    fn pallas_fp_mul(a: *const u64, b: *const u64, r: *mut u64);
    fn pallas_fp_neg(a: *const u64, r: *mut u64);
    fn pallas_pippenger_msm(points: *const u64, scalars: *const u32, n: libc::c_int, result: *mut u64);

    // -- Vesta --
    fn vesta_fp_add(a: *const u64, b: *const u64, r: *mut u64);
    fn vesta_fp_sub(a: *const u64, b: *const u64, r: *mut u64);
    fn vesta_fp_mul(a: *const u64, b: *const u64, r: *mut u64);
    fn vesta_fp_neg(a: *const u64, r: *mut u64);
    fn vesta_pippenger_msm(points: *const u64, scalars: *const u32, n: libc::c_int, result: *mut u64);

    // -- Grumpkin --
    fn grumpkin_point_add(p: *const u64, q: *const u64, r: *mut u64);
    fn grumpkin_point_double(p: *const u64, r: *mut u64);
    fn grumpkin_scalar_mul(p: *const u64, scalar: *const u64, r: *mut u64);
    fn grumpkin_pippenger_msm(points: *const u64, scalars: *const u32, n: libc::c_int, result: *mut u64);

    // -- secp256k1 --
    fn secp256k1_point_add(p: *const u64, q: *const u64, r: *mut u64);
    fn secp256k1_point_scalar_mul(p: *const u64, scalar: *const u64, r: *mut u64);
    fn secp256k1_pippenger_msm(points: *const u64, scalars: *const u32, n: libc::c_int, result: *mut u64);

    // -- Ed25519 --
    fn ed25519_fp_add(a: *const u64, b: *const u64, r: *mut u64);
    fn ed25519_fp_sub(a: *const u64, b: *const u64, r: *mut u64);
    fn ed25519_fp_mul(a: *const u64, b: *const u64, r: *mut u64);
    fn ed25519_fp_neg(a: *const u64, r: *mut u64);
    fn ed25519_fp_inverse(a: *const u64, r: *mut u64);
    fn ed25519_scalar_mul(p: *const u64, scalar: *const u64, r: *mut u64);
    fn ed25519_pippenger_msm(points: *const u64, scalars: *const u32, n: libc::c_int, result: *mut u64);
    fn ed25519_eddsa_verify(
        gen: *const u64, s_raw: *const u64, r_point: *const u64,
        h_raw: *const u64, pub_key: *const u64,
    ) -> libc::c_int;

    // -- Goldilocks batch ops --
    fn gl_batch_add_neon(a: *const u64, b: *const u64, out: *mut u64, n: libc::c_int);
    fn gl_batch_sub_neon(a: *const u64, b: *const u64, out: *mut u64, n: libc::c_int);
    fn gl_batch_mul_neon(a: *const u64, b: *const u64, out: *mut u64, n: libc::c_int);
}

// ============================================================================
// Macro for field op wrappers (reduces boilerplate)
// ============================================================================

macro_rules! field_ops_4 {
    ($prefix:ident, $add_fn:ident, $sub_fn:ident, $mul_fn:ident, $neg_fn:ident) => {
        pub mod $prefix {
            use super::*;

            pub fn add(a: &Fp4, b: &Fp4) -> Fp4 {
                let mut r = Fp4::default();
                unsafe { $add_fn(a.0.as_ptr(), b.0.as_ptr(), r.0.as_mut_ptr()) };
                r
            }
            pub fn sub(a: &Fp4, b: &Fp4) -> Fp4 {
                let mut r = Fp4::default();
                unsafe { $sub_fn(a.0.as_ptr(), b.0.as_ptr(), r.0.as_mut_ptr()) };
                r
            }
            pub fn mul(a: &Fp4, b: &Fp4) -> Fp4 {
                let mut r = Fp4::default();
                unsafe { $mul_fn(a.0.as_ptr(), b.0.as_ptr(), r.0.as_mut_ptr()) };
                r
            }
            pub fn neg(a: &Fp4) -> Fp4 {
                let mut r = Fp4::default();
                unsafe { $neg_fn(a.0.as_ptr(), r.0.as_mut_ptr()) };
                r
            }
        }
    };
}

macro_rules! msm_4 {
    ($fn_name:ident, $ffi_fn:ident) => {
        pub fn $fn_name(points: &[u64], scalars: &[u32]) -> Proj4 {
            let n = points.len() / 8;
            assert_eq!(points.len(), n * 8);
            assert_eq!(scalars.len(), n * 8);
            let mut r = Proj4::default();
            unsafe { $ffi_fn(points.as_ptr(), scalars.as_ptr(), n as libc::c_int, r.0.as_mut_ptr()) };
            r
        }
    };
}

macro_rules! msm_6 {
    ($fn_name:ident, $ffi_fn:ident) => {
        pub fn $fn_name(points: &[u64], scalars: &[u32]) -> Proj6 {
            let n = points.len() / 12;
            assert_eq!(points.len(), n * 12);
            assert_eq!(scalars.len(), n * 8);
            let mut r = Proj6::default();
            unsafe { $ffi_fn(points.as_ptr(), scalars.as_ptr(), n as libc::c_int, r.0.as_mut_ptr()) };
            r
        }
    };
}

// ============================================================================
// BLS12-381 safe wrappers
// ============================================================================

field_ops_4!(bls12_381_fr, bls12_381_fr_add, bls12_381_fr_sub, bls12_381_fr_mul, bls12_381_fr_neg);

pub mod bls12_381_fp {
    use super::*;

    pub fn add(a: &Fp6, b: &Fp6) -> Fp6 {
        let mut r = Fp6::default();
        unsafe { bls12_381_fp_add(a.0.as_ptr(), b.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
    pub fn sub(a: &Fp6, b: &Fp6) -> Fp6 {
        let mut r = Fp6::default();
        unsafe { bls12_381_fp_sub(a.0.as_ptr(), b.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
    pub fn mul(a: &Fp6, b: &Fp6) -> Fp6 {
        let mut r = Fp6::default();
        unsafe { bls12_381_fp_mul(a.0.as_ptr(), b.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
    pub fn neg(a: &Fp6) -> Fp6 {
        let mut r = Fp6::default();
        unsafe { bls12_381_fp_neg(a.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
    pub fn inv(a: &Fp6) -> Fp6 {
        let mut r = Fp6::default();
        unsafe { bls12_381_fp_inv_ext(a.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
    /// Returns `Some(sqrt)` if a is a QR, `None` otherwise.
    pub fn sqrt(a: &Fp6) -> Option<Fp6> {
        let mut r = Fp6::default();
        let ok = unsafe { bls12_381_fp_sqrt(a.0.as_ptr(), r.0.as_mut_ptr()) };
        if ok == 1 { Some(r) } else { None }
    }
}

msm_6!(bls12_381_g1_msm, bls12_381_g1_pippenger_msm);

/// BLS12-381 pairing: e(P, Q) -> Fp12 (72 x u64).
pub fn bls12_381_pairing_compute(p_aff: &[u64; 12], q_aff: &[u64; 24]) -> [u64; 72] {
    let mut r = [0u64; 72];
    unsafe { bls12_381_pairing(p_aff.as_ptr(), q_aff.as_ptr(), r.as_mut_ptr()) };
    r
}

/// BLS12-381 pairing check: verify product of pairings == 1.
/// `pairs` is a flat array of (G1_affine[12], G2_affine[24]) tuples.
pub fn bls12_381_pairing_check_multi(pairs: &[u64], n: usize) -> bool {
    assert_eq!(pairs.len(), n * 36); // 12 + 24 per pair
    unsafe { bls12_381_pairing_check(pairs.as_ptr(), n as libc::c_int) == 1 }
}

// ============================================================================
// Pallas / Vesta safe wrappers
// ============================================================================

field_ops_4!(pallas, pallas_fp_add, pallas_fp_sub, pallas_fp_mul, pallas_fp_neg);
field_ops_4!(vesta, vesta_fp_add, vesta_fp_sub, vesta_fp_mul, vesta_fp_neg);

msm_4!(pallas_msm, pallas_pippenger_msm);
msm_4!(vesta_msm, vesta_pippenger_msm);

// ============================================================================
// Grumpkin safe wrappers
// ============================================================================

msm_4!(grumpkin_msm, grumpkin_pippenger_msm);

pub fn grumpkin_point_add_proj(p: &Proj4, q: &Proj4) -> Proj4 {
    let mut r = Proj4::default();
    unsafe { grumpkin_point_add(p.0.as_ptr(), q.0.as_ptr(), r.0.as_mut_ptr()) };
    r
}

pub fn grumpkin_point_double_proj(p: &Proj4) -> Proj4 {
    let mut r = Proj4::default();
    unsafe { grumpkin_point_double(p.0.as_ptr(), r.0.as_mut_ptr()) };
    r
}

pub fn grumpkin_scalar_mul_proj(p: &Proj4, scalar: &[u64; 4]) -> Proj4 {
    let mut r = Proj4::default();
    unsafe { grumpkin_scalar_mul(p.0.as_ptr(), scalar.as_ptr(), r.0.as_mut_ptr()) };
    r
}

// ============================================================================
// secp256k1 safe wrappers
// ============================================================================

msm_4!(secp256k1_msm, secp256k1_pippenger_msm);

pub fn secp256k1_point_add_proj(p: &Proj4, q: &Proj4) -> Proj4 {
    let mut r = Proj4::default();
    unsafe { secp256k1_point_add(p.0.as_ptr(), q.0.as_ptr(), r.0.as_mut_ptr()) };
    r
}

// ============================================================================
// Ed25519 safe wrappers
// ============================================================================

field_ops_4!(ed25519_fp, ed25519_fp_add, ed25519_fp_sub, ed25519_fp_mul, ed25519_fp_neg);

pub fn ed25519_msm(points: &[u64], scalars: &[u32]) -> ExtendedPoint {
    let n = points.len() / 8;
    assert_eq!(points.len(), n * 8);
    assert_eq!(scalars.len(), n * 8);
    let mut r = ExtendedPoint::default();
    unsafe { ed25519_pippenger_msm(points.as_ptr(), scalars.as_ptr(), n as libc::c_int, r.0.as_mut_ptr()) };
    r
}

/// Verify an Ed25519 EdDSA signature.
/// Returns `true` if the signature is valid.
pub fn ed25519_verify(
    generator: &ExtendedPoint,
    s_raw: &[u64; 4],
    r_point: &ExtendedPoint,
    h_raw: &[u64; 4],
    pub_key: &ExtendedPoint,
) -> bool {
    unsafe {
        ed25519_eddsa_verify(
            generator.0.as_ptr(), s_raw.as_ptr(), r_point.0.as_ptr(),
            h_raw.as_ptr(), pub_key.0.as_ptr(),
        ) == 1
    }
}

// ============================================================================
// Goldilocks batch ops
// ============================================================================

pub mod goldilocks {
    use super::*;

    /// Batch add: out[i] = a[i] + b[i] mod p.
    pub fn batch_add(a: &[u64], b: &[u64], out: &mut [u64]) {
        let n = a.len();
        assert_eq!(b.len(), n);
        assert_eq!(out.len(), n);
        unsafe { gl_batch_add_neon(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n as libc::c_int) };
    }

    /// Batch sub: out[i] = a[i] - b[i] mod p.
    pub fn batch_sub(a: &[u64], b: &[u64], out: &mut [u64]) {
        let n = a.len();
        assert_eq!(b.len(), n);
        assert_eq!(out.len(), n);
        unsafe { gl_batch_sub_neon(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n as libc::c_int) };
    }

    /// Batch mul: out[i] = a[i] * b[i] mod p.
    pub fn batch_mul(a: &[u64], b: &[u64], out: &mut [u64]) {
        let n = a.len();
        assert_eq!(b.len(), n);
        assert_eq!(out.len(), n);
        unsafe { gl_batch_mul_neon(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n as libc::c_int) };
    }
}
