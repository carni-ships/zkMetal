//! BN254 field elements and curve operations.
//!
//! All field elements are stored in Montgomery form as 4 x u64 limbs (little-endian).

use core::ops::{Add, Sub, Mul, Neg};

/// BN254 scalar field element (Fr). 4 x u64 limbs in Montgomery form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct Fr(pub [u64; 4]);

/// BN254 base field element (Fq). 4 x u64 limbs in Montgomery form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct Fq(pub [u64; 4]);

/// BN254 G1 affine point: (x, y) where x, y are Fq elements.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct G1Affine {
    pub x: Fq,
    pub y: Fq,
}

/// BN254 G1 projective (Jacobian) point: (X, Y, Z) where x = X/Z^2, y = Y/Z^3.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct G1Projective {
    pub x: Fq,
    pub y: Fq,
    pub z: Fq,
}

// ---------------------------------------------------------------------------
// Fr arithmetic
// ---------------------------------------------------------------------------

impl Fr {
    /// The zero element.
    pub const ZERO: Self = Fr([0; 4]);

    /// Multiplicative inverse via Fermat's little theorem.
    #[inline]
    pub fn inverse(&self) -> Self {
        let mut r = Fr::ZERO;
        unsafe { zkmetal_sys::bn254_fr_inverse(self.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }

    /// Squaring (faster than general multiply).
    #[inline]
    pub fn square(&self) -> Self {
        let mut r = Fr::ZERO;
        unsafe { zkmetal_sys::bn254_fr_sqr(self.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }

    /// Exponentiation: self^exp.
    #[inline]
    pub fn pow(&self, exp: u64) -> Self {
        let mut r = Fr::ZERO;
        unsafe { zkmetal_sys::bn254_fr_pow(self.0.as_ptr(), exp, r.0.as_mut_ptr()) };
        r
    }

    /// Equality check (constant-time in the C implementation).
    #[inline]
    pub fn ct_eq(&self, other: &Self) -> bool {
        unsafe { zkmetal_sys::bn254_fr_eq(self.0.as_ptr(), other.0.as_ptr()) == 1 }
    }

    /// Inner product: sum(a[i] * b[i]).
    pub fn inner_product(a: &[Fr], b: &[Fr]) -> Fr {
        assert_eq!(a.len(), b.len(), "inner_product: length mismatch");
        let mut r = Fr::ZERO;
        unsafe {
            zkmetal_sys::bn254_fr_inner_product(
                a.as_ptr() as *const u64,
                b.as_ptr() as *const u64,
                a.len() as core::ffi::c_int,
                r.0.as_mut_ptr(),
            );
        }
        r
    }

    /// Batch inverse via Montgomery's trick. O(3n) muls + 1 inversion.
    pub fn batch_inverse(elements: &[Fr]) -> Vec<Fr> {
        let mut out = vec![Fr::ZERO; elements.len()];
        unsafe {
            zkmetal_sys::bn254_fr_batch_inverse(
                elements.as_ptr() as *const u64,
                elements.len() as core::ffi::c_int,
                out.as_mut_ptr() as *mut u64,
            );
        }
        out
    }

    /// Horner polynomial evaluation: coeffs[0] + coeffs[1]*z + ... + coeffs[n-1]*z^(n-1).
    pub fn horner_eval(coeffs: &[Fr], z: &Fr) -> Fr {
        let mut r = Fr::ZERO;
        unsafe {
            zkmetal_sys::bn254_fr_horner_eval(
                coeffs.as_ptr() as *const u64,
                coeffs.len() as core::ffi::c_int,
                z.0.as_ptr(),
                r.0.as_mut_ptr(),
            );
        }
        r
    }
}

impl Add for Fr {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut r = Fr::ZERO;
        unsafe { zkmetal_sys::bn254_fr_add(self.0.as_ptr(), rhs.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
}

impl Sub for Fr {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut r = Fr::ZERO;
        unsafe { zkmetal_sys::bn254_fr_sub(self.0.as_ptr(), rhs.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
}

impl Mul for Fr {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut r = Fr::ZERO;
        unsafe { zkmetal_sys::bn254_fr_mul(self.0.as_ptr(), rhs.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
}

impl Neg for Fr {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut r = Fr::ZERO;
        unsafe { zkmetal_sys::bn254_fr_neg(self.0.as_ptr(), r.0.as_mut_ptr()) };
        r
    }
}

// ---------------------------------------------------------------------------
// G1 curve operations
// ---------------------------------------------------------------------------

impl G1Projective {
    /// The point at infinity (identity).
    pub const IDENTITY: Self = G1Projective {
        x: Fq([0; 4]),
        y: Fq([0; 4]),
        z: Fq([0; 4]),
    };

    /// Projective point addition.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut r = Self::IDENTITY;
        unsafe {
            zkmetal_sys::bn254_point_add(
                self as *const _ as *const u64,
                other as *const _ as *const u64,
                &mut r as *mut _ as *mut u64,
            );
        }
        r
    }

    /// Mixed addition: projective + affine (saves field operations).
    #[inline]
    pub fn add_mixed(&self, other: &G1Affine) -> Self {
        let mut r = Self::IDENTITY;
        unsafe {
            zkmetal_sys::bn254_point_add_mixed(
                self as *const _ as *const u64,
                other as *const _ as *const u64,
                &mut r as *mut _ as *mut u64,
            );
        }
        r
    }

    /// Scalar multiplication. Scalar is 8 x u32 in non-Montgomery integer form.
    #[inline]
    pub fn scalar_mul(&self, scalar: &[u32; 8]) -> Self {
        let mut r = Self::IDENTITY;
        unsafe {
            zkmetal_sys::bn254_point_scalar_mul(
                self as *const _ as *const u64,
                scalar.as_ptr(),
                &mut r as *mut _ as *mut u64,
            );
        }
        r
    }

    /// Convert to affine coordinates.
    #[inline]
    pub fn to_affine(&self) -> G1Affine {
        let mut aff = G1Affine {
            x: Fq([0; 4]),
            y: Fq([0; 4]),
        };
        unsafe {
            zkmetal_sys::bn254_projective_to_affine(
                self as *const _ as *const u64,
                &mut aff as *mut _ as *mut u64,
            );
        }
        aff
    }

    /// Batch convert projective points to affine (Montgomery's trick).
    pub fn batch_to_affine(points: &[G1Projective]) -> Vec<G1Affine> {
        let mut aff = vec![
            G1Affine {
                x: Fq([0; 4]),
                y: Fq([0; 4]),
            };
            points.len()
        ];
        unsafe {
            zkmetal_sys::bn254_batch_to_affine(
                points.as_ptr() as *const u64,
                aff.as_mut_ptr() as *mut u64,
                points.len() as core::ffi::c_int,
            );
        }
        aff
    }
}
