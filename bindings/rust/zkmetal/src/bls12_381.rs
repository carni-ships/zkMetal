//! BLS12-381 field elements and curve operations.
//!
//! Fr (scalar field): 4 x u64 limbs in Montgomery form.
//! Fp (base field): 6 x u64 limbs in Montgomery form.
//! Fp2: 12 x u64 (two Fp elements: c0 + c1*u).

use core::ops::{Add, Sub, Mul, Neg};

/// BLS12-381 scalar field element (Fr). 4 x u64, Montgomery form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct Fr(pub [u64; 4]);

/// BLS12-381 base field element (Fp). 6 x u64, Montgomery form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct Fp(pub [u64; 6]);

/// BLS12-381 Fp2 element: c0 + c1*u where u^2 = -1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Fp2 {
    pub c0: Fp,
    pub c1: Fp,
}

/// BLS12-381 G1 affine point.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct G1Affine {
    pub x: Fp,
    pub y: Fp,
}

/// BLS12-381 G1 projective (Jacobian) point.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct G1Projective {
    pub x: Fp,
    pub y: Fp,
    pub z: Fp,
}

/// BLS12-381 G2 affine point (Fp2 coordinates).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct G2Affine {
    pub x: Fp2,
    pub y: Fp2,
}

/// BLS12-381 G2 projective point (Fp2 coordinates).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct G2Projective {
    pub x: Fp2,
    pub y: Fp2,
    pub z: Fp2,
}

// ---------------------------------------------------------------------------
// Fr arithmetic
// ---------------------------------------------------------------------------

impl Fr {
    pub const ZERO: Self = Fr([0; 4]);

    #[inline]
    pub fn square(&self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fr_sqr(self.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
}

impl Add for Fr {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fr_add(self.0.as_ptr(), rhs.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
}

impl Sub for Fr {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fr_sub(self.0.as_ptr(), rhs.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
}

impl Mul for Fr {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fr_mul(self.0.as_ptr(), rhs.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
}

impl Neg for Fr {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fr_neg(self.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
}

// ---------------------------------------------------------------------------
// Fp arithmetic
// ---------------------------------------------------------------------------

impl Fp {
    pub const ZERO: Self = Fp([0; 6]);

    #[inline]
    pub fn square(&self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fp_sqr(self.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }

    #[inline]
    pub fn inverse(&self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fp_inv_ext(self.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }

    /// Returns (true, sqrt) if square root exists, (false, _) otherwise.
    #[inline]
    pub fn sqrt(&self) -> Option<Self> {
        let mut r = Self::ZERO;
        let exists = unsafe { zkmetal_sys::bls12_381_fp_sqrt(self.0.as_ptr(), r.0.as_mut_ptr()) };
        if exists == 1 { Some(r) } else { None }
    }
}

impl Add for Fp {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fp_add(self.0.as_ptr(), rhs.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
}

impl Sub for Fp {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fp_sub(self.0.as_ptr(), rhs.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
}

impl Mul for Fp {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fp_mul(self.0.as_ptr(), rhs.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
}

impl Neg for Fp {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fp_neg(self.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
}

// ---------------------------------------------------------------------------
// Fp2 arithmetic
// ---------------------------------------------------------------------------

impl Fp2 {
    pub const ZERO: Self = Fp2 { c0: Fp::ZERO, c1: Fp::ZERO };

    #[inline]
    pub fn square(&self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fp2_sqr(self as *const _ as *const u64, &mut r as *mut _ as *mut u64) };
        r
    }

    #[inline]
    pub fn conjugate(&self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fp2_conj(self as *const _ as *const u64, &mut r as *mut _ as *mut u64) };
        r
    }

    #[inline]
    pub fn inverse(&self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fp2_inv(self as *const _ as *const u64, &mut r as *mut _ as *mut u64) };
        r
    }
}

impl Add for Fp2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fp2_add(&self as *const _ as *const u64, &rhs as *const _ as *const u64, &mut r as *mut _ as *mut u64) };
        r
    }
}

impl Sub for Fp2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fp2_sub(&self as *const _ as *const u64, &rhs as *const _ as *const u64, &mut r as *mut _ as *mut u64) };
        r
    }
}

impl Mul for Fp2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fp2_mul(&self as *const _ as *const u64, &rhs as *const _ as *const u64, &mut r as *mut _ as *mut u64) };
        r
    }
}

impl Neg for Fp2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut r = Self::ZERO;
        unsafe { zkmetal_sys::bls12_381_fp2_neg(&self as *const _ as *const u64, &mut r as *mut _ as *mut u64) };
        r
    }
}

// ---------------------------------------------------------------------------
// G1 curve operations
// ---------------------------------------------------------------------------

impl G1Projective {
    pub const IDENTITY: Self = G1Projective {
        x: Fp::ZERO,
        y: Fp::ZERO,
        z: Fp::ZERO,
    };

    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut r = Self::IDENTITY;
        unsafe {
            zkmetal_sys::bls12_381_g1_point_add(
                self as *const _ as *const u64,
                other as *const _ as *const u64,
                &mut r as *mut _ as *mut u64,
            );
        }
        r
    }

    #[inline]
    pub fn double(&self) -> Self {
        let mut r = Self::IDENTITY;
        unsafe {
            zkmetal_sys::bls12_381_g1_point_double(
                self as *const _ as *const u64,
                &mut r as *mut _ as *mut u64,
            );
        }
        r
    }

    #[inline]
    pub fn add_mixed(&self, other: &G1Affine) -> Self {
        let mut r = Self::IDENTITY;
        unsafe {
            zkmetal_sys::bls12_381_g1_point_add_mixed(
                self as *const _ as *const u64,
                other as *const _ as *const u64,
                &mut r as *mut _ as *mut u64,
            );
        }
        r
    }

    /// Scalar multiplication. Scalar is 4 x u64 (non-Montgomery integer form).
    #[inline]
    pub fn scalar_mul(&self, scalar: &[u64; 4]) -> Self {
        let mut r = Self::IDENTITY;
        unsafe {
            zkmetal_sys::bls12_381_g1_scalar_mul(
                self as *const _ as *const u64,
                scalar.as_ptr(),
                &mut r as *mut _ as *mut u64,
            );
        }
        r
    }
}

// ---------------------------------------------------------------------------
// G2 curve operations
// ---------------------------------------------------------------------------

impl G2Projective {
    pub const IDENTITY: Self = G2Projective {
        x: Fp2::ZERO,
        y: Fp2::ZERO,
        z: Fp2::ZERO,
    };

    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut r = Self::IDENTITY;
        unsafe {
            zkmetal_sys::bls12_381_g2_point_add(
                self as *const _ as *const u64,
                other as *const _ as *const u64,
                &mut r as *mut _ as *mut u64,
            );
        }
        r
    }

    #[inline]
    pub fn double(&self) -> Self {
        let mut r = Self::IDENTITY;
        unsafe {
            zkmetal_sys::bls12_381_g2_point_double(
                self as *const _ as *const u64,
                &mut r as *mut _ as *mut u64,
            );
        }
        r
    }

    #[inline]
    pub fn scalar_mul(&self, scalar: &[u64; 4]) -> Self {
        let mut r = Self::IDENTITY;
        unsafe {
            zkmetal_sys::bls12_381_g2_scalar_mul(
                self as *const _ as *const u64,
                scalar.as_ptr(),
                &mut r as *mut _ as *mut u64,
            );
        }
        r
    }
}
