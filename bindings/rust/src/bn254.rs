//! BN254 Fr (scalar field) safe wrappers.
//!
//! Field elements are 32 bytes = 4 x u64 limbs in little-endian Montgomery form.
//! The `Fr` type wraps `[u64; 4]` with `repr(C)` for direct FFI interop.

/// BN254 Fr field element: 4 x u64 limbs in little-endian Montgomery form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Fr(pub [u64; 4]);

impl Fr {
    /// The zero element.
    pub const ZERO: Self = Self([0, 0, 0, 0]);

    /// Create from raw Montgomery limbs.
    pub const fn from_raw(limbs: [u64; 4]) -> Self {
        Self(limbs)
    }

    /// View as a byte slice (32 bytes, little-endian).
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const u8, 32) }
    }

    /// Construct from a 32-byte little-endian slice (Montgomery form).
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            limbs[i] = u64::from_le_bytes(bytes[i * 8..(i + 1) * 8].try_into().unwrap());
        }
        Self(limbs)
    }
}

impl Default for Fr {
    fn default() -> Self {
        Self::ZERO
    }
}

// ============================================================================
// FFI declarations for NeonFieldOps BN254 Fr operations
// ============================================================================

#[cfg(feature = "neon")]
mod ffi {
    extern "C" {
        pub fn bn254_fr_add(a: *const u64, b: *const u64, r: *mut u64);
        pub fn bn254_fr_sub(a: *const u64, b: *const u64, r: *mut u64);
        pub fn bn254_fr_mul(a: *const u64, b: *const u64, r: *mut u64);
        pub fn bn254_fr_neg(a: *const u64, r: *mut u64);
        pub fn bn254_fr_inverse(a: *const u64, r: *mut u64);
        pub fn bn254_fr_sqr(a: *const u64, r: *mut u64);
        pub fn bn254_fr_pow(a: *const u64, exp: u64, r: *mut u64);
        pub fn bn254_fr_eq(a: *const u64, b: *const u64) -> libc::c_int;

        pub fn bn254_fr_batch_add_neon(
            result: *mut u64, a: *const u64, b: *const u64, n: libc::c_int,
        );
        pub fn bn254_fr_batch_sub_neon(
            result: *mut u64, a: *const u64, b: *const u64, n: libc::c_int,
        );
        pub fn bn254_fr_batch_neg_neon(result: *mut u64, a: *const u64, n: libc::c_int);
        pub fn bn254_fr_batch_mul_scalar_neon(
            result: *mut u64, a: *const u64, scalar: *const u64, n: libc::c_int,
        );
        pub fn bn254_fr_batch_inverse(a: *const u64, n: libc::c_int, out: *mut u64);

        pub fn bn254_fr_inner_product(
            a: *const u64, b: *const u64, n: libc::c_int, result: *mut u64,
        );
        pub fn bn254_fr_vector_sum(a: *const u64, n: libc::c_int, result: *mut u64);
        pub fn bn254_fr_horner_eval(
            coeffs: *const u64, n: libc::c_int, z: *const u64, result: *mut u64,
        );

        // Montgomery mul (ASM + C variants)
        pub fn mont_mul_asm(result: *mut u64, a: *const u64, b: *const u64);
        pub fn mont_mul_c(result: *mut u64, a: *const u64, b: *const u64);
    }
}

// ============================================================================
// Safe wrappers (require "neon" feature)
// ============================================================================

/// Modular addition: r = (a + b) mod p.
#[cfg(feature = "neon")]
pub fn fr_add(a: &Fr, b: &Fr) -> Fr {
    let mut r = Fr::ZERO;
    unsafe { ffi::bn254_fr_add(a.0.as_ptr(), b.0.as_ptr(), r.0.as_mut_ptr()) };
    r
}

/// Modular subtraction: r = (a - b) mod p.
#[cfg(feature = "neon")]
pub fn fr_sub(a: &Fr, b: &Fr) -> Fr {
    let mut r = Fr::ZERO;
    unsafe { ffi::bn254_fr_sub(a.0.as_ptr(), b.0.as_ptr(), r.0.as_mut_ptr()) };
    r
}

/// Montgomery multiplication: r = a * b * R^{-1} mod p.
#[cfg(feature = "neon")]
pub fn fr_mul(a: &Fr, b: &Fr) -> Fr {
    let mut r = Fr::ZERO;
    unsafe { ffi::bn254_fr_mul(a.0.as_ptr(), b.0.as_ptr(), r.0.as_mut_ptr()) };
    r
}

/// Modular negation: r = (-a) mod p.
#[cfg(feature = "neon")]
pub fn fr_neg(a: &Fr) -> Fr {
    let mut r = Fr::ZERO;
    unsafe { ffi::bn254_fr_neg(a.0.as_ptr(), r.0.as_mut_ptr()) };
    r
}

/// Modular inverse via Fermat's little theorem: r = a^(p-2) mod p.
#[cfg(feature = "neon")]
pub fn fr_inverse(a: &Fr) -> Fr {
    let mut r = Fr::ZERO;
    unsafe { ffi::bn254_fr_inverse(a.0.as_ptr(), r.0.as_mut_ptr()) };
    r
}

/// Optimized squaring: r = a^2 mod p.
#[cfg(feature = "neon")]
pub fn fr_sqr(a: &Fr) -> Fr {
    let mut r = Fr::ZERO;
    unsafe { ffi::bn254_fr_sqr(a.0.as_ptr(), r.0.as_mut_ptr()) };
    r
}

/// Exponentiation: r = a^exp mod p (64-bit exponent).
#[cfg(feature = "neon")]
pub fn fr_pow(a: &Fr, exp: u64) -> Fr {
    let mut r = Fr::ZERO;
    unsafe { ffi::bn254_fr_pow(a.0.as_ptr(), exp, r.0.as_mut_ptr()) };
    r
}

/// Equality check.
#[cfg(feature = "neon")]
pub fn fr_eq(a: &Fr, b: &Fr) -> bool {
    unsafe { ffi::bn254_fr_eq(a.0.as_ptr(), b.0.as_ptr()) == 1 }
}

/// Batch addition: result[i] = a[i] + b[i] mod p. NEON-accelerated.
#[cfg(feature = "neon")]
pub fn fr_batch_add(a: &[Fr], b: &[Fr], result: &mut [Fr]) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert_eq!(result.len(), n);
    unsafe {
        ffi::bn254_fr_batch_add_neon(
            result.as_mut_ptr() as *mut u64,
            a.as_ptr() as *const u64,
            b.as_ptr() as *const u64,
            n as libc::c_int,
        );
    }
}

/// Batch subtraction: result[i] = a[i] - b[i] mod p. NEON-accelerated.
#[cfg(feature = "neon")]
pub fn fr_batch_sub(a: &[Fr], b: &[Fr], result: &mut [Fr]) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert_eq!(result.len(), n);
    unsafe {
        ffi::bn254_fr_batch_sub_neon(
            result.as_mut_ptr() as *mut u64,
            a.as_ptr() as *const u64,
            b.as_ptr() as *const u64,
            n as libc::c_int,
        );
    }
}

/// Batch negation: result[i] = -a[i] mod p. NEON-accelerated.
#[cfg(feature = "neon")]
pub fn fr_batch_neg(a: &[Fr], result: &mut [Fr]) {
    let n = a.len();
    assert_eq!(result.len(), n);
    unsafe {
        ffi::bn254_fr_batch_neg_neon(
            result.as_mut_ptr() as *mut u64,
            a.as_ptr() as *const u64,
            n as libc::c_int,
        );
    }
}

/// Batch scalar multiply: result[i] = a[i] * scalar mod p. NEON-accelerated.
#[cfg(feature = "neon")]
pub fn fr_batch_mul_scalar(a: &[Fr], scalar: &Fr, result: &mut [Fr]) {
    let n = a.len();
    assert_eq!(result.len(), n);
    unsafe {
        ffi::bn254_fr_batch_mul_scalar_neon(
            result.as_mut_ptr() as *mut u64,
            a.as_ptr() as *const u64,
            scalar.0.as_ptr(),
            n as libc::c_int,
        );
    }
}

/// Batch inverse via Montgomery's trick: result[i] = a[i]^(-1).
/// O(3n) muls + 1 Fermat inversion.
#[cfg(feature = "neon")]
pub fn fr_batch_inverse(a: &[Fr], result: &mut [Fr]) {
    let n = a.len();
    assert_eq!(result.len(), n);
    unsafe {
        ffi::bn254_fr_batch_inverse(
            a.as_ptr() as *const u64,
            n as libc::c_int,
            result.as_mut_ptr() as *mut u64,
        );
    }
}

/// Inner product: result = sum(a[i] * b[i]).
#[cfg(feature = "neon")]
pub fn fr_inner_product(a: &[Fr], b: &[Fr]) -> Fr {
    let n = a.len();
    assert_eq!(b.len(), n);
    let mut r = Fr::ZERO;
    unsafe {
        ffi::bn254_fr_inner_product(
            a.as_ptr() as *const u64,
            b.as_ptr() as *const u64,
            n as libc::c_int,
            r.0.as_mut_ptr(),
        );
    }
    r
}

/// Vector sum: result = sum(a[i]).
#[cfg(feature = "neon")]
pub fn fr_vector_sum(a: &[Fr]) -> Fr {
    let mut r = Fr::ZERO;
    unsafe {
        ffi::bn254_fr_vector_sum(
            a.as_ptr() as *const u64,
            a.len() as libc::c_int,
            r.0.as_mut_ptr(),
        );
    }
    r
}

/// Horner polynomial evaluation: result = coeffs[0] + coeffs[1]*z + ... + coeffs[n-1]*z^(n-1).
#[cfg(feature = "neon")]
pub fn fr_horner_eval(coeffs: &[Fr], z: &Fr) -> Fr {
    let mut r = Fr::ZERO;
    unsafe {
        ffi::bn254_fr_horner_eval(
            coeffs.as_ptr() as *const u64,
            coeffs.len() as libc::c_int,
            z.0.as_ptr(),
            r.0.as_mut_ptr(),
        );
    }
    r
}

#[cfg(all(test, feature = "neon"))]
mod tests {
    use super::*;

    #[test]
    fn test_fr_add_sub_roundtrip() {
        let a = Fr([1, 0, 0, 0]);
        let b = Fr([2, 0, 0, 0]);
        let c = fr_add(&a, &b);
        let d = fr_sub(&c, &b);
        assert!(fr_eq(&a, &d));
    }

    #[test]
    fn test_fr_neg_double() {
        let a = Fr([42, 0, 0, 0]);
        let neg_a = fr_neg(&a);
        let zero = fr_add(&a, &neg_a);
        assert_eq!(zero, Fr::ZERO);
    }
}
