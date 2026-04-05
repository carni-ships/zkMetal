//! Bilinear pairing operations for BN254 and BLS12-381.

/// BN254 optimal Ate pairing: e(P, Q) -> Gt.
///
/// - `p_affine`: G1 affine point (8 x u64: x[4], y[4]).
/// - `q_affine`: G2 affine point (16 x u64: x[4][2], y[4][2]).
/// - Returns: Fp12 element (48 x u64).
pub fn bn254_pairing(p_affine: &[u64; 8], q_affine: &[u64; 16]) -> [u64; 48] {
    let mut result = [0u64; 48];
    unsafe {
        zkmetal_sys::bn254_pairing(p_affine.as_ptr(), q_affine.as_ptr(), result.as_mut_ptr());
    }
    result
}

/// BN254 Miller loop only (without final exponentiation).
pub fn bn254_miller_loop(p_affine: &[u64; 8], q_affine: &[u64; 16]) -> [u64; 48] {
    let mut result = [0u64; 48];
    unsafe {
        zkmetal_sys::bn254_miller_loop(p_affine.as_ptr(), q_affine.as_ptr(), result.as_mut_ptr());
    }
    result
}

/// BN254 final exponentiation.
pub fn bn254_final_exp(f: &[u64; 48]) -> [u64; 48] {
    let mut result = [0u64; 48];
    unsafe {
        zkmetal_sys::bn254_final_exp(f.as_ptr(), result.as_mut_ptr());
    }
    result
}

/// BN254 pairing check: verifies that the product of pairings equals 1.
///
/// `pairs`: flat array of n pairs, each consisting of a G1Affine (8 u64) + G2Affine (16 u64) = 24 u64.
/// Returns true if the check passes.
pub fn bn254_pairing_check(pairs: &[u64], n: usize) -> bool {
    assert!(pairs.len() >= n * 24, "pairing_check: pairs buffer too small");
    unsafe { zkmetal_sys::bn254_pairing_check(pairs.as_ptr(), n as core::ffi::c_int) == 1 }
}

/// BLS12-381 optimal Ate pairing: e(P, Q) -> Gt.
///
/// - `p_affine`: G1 affine point (12 x u64: x[6], y[6]).
/// - `q_affine`: G2 affine point (24 x u64: x[12], y[12]).
/// - Returns: Fp12 element (72 x u64).
pub fn bls12_381_pairing(p_affine: &[u64; 12], q_affine: &[u64; 24]) -> [u64; 72] {
    let mut result = [0u64; 72];
    unsafe {
        zkmetal_sys::bls12_381_pairing(p_affine.as_ptr(), q_affine.as_ptr(), result.as_mut_ptr());
    }
    result
}

/// BLS12-381 Miller loop only.
pub fn bls12_381_miller_loop(p_affine: &[u64; 12], q_affine: &[u64; 24]) -> [u64; 72] {
    let mut result = [0u64; 72];
    unsafe {
        zkmetal_sys::bls12_381_miller_loop(p_affine.as_ptr(), q_affine.as_ptr(), result.as_mut_ptr());
    }
    result
}

/// BLS12-381 final exponentiation.
pub fn bls12_381_final_exp(f: &[u64; 72]) -> [u64; 72] {
    let mut result = [0u64; 72];
    unsafe {
        zkmetal_sys::bls12_381_final_exp(f.as_ptr(), result.as_mut_ptr());
    }
    result
}

/// BLS12-381 pairing check.
///
/// `pairs`: flat array of n pairs, each G1Affine (12 u64) + G2Affine (24 u64) = 36 u64.
pub fn bls12_381_pairing_check(pairs: &[u64], n: usize) -> bool {
    assert!(pairs.len() >= n * 36, "pairing_check: pairs buffer too small");
    unsafe { zkmetal_sys::bls12_381_pairing_check(pairs.as_ptr(), n as core::ffi::c_int) == 1 }
}

/// BLS12-381 hash-to-curve G2 (RFC 9380).
///
/// Maps a message to a G2 point using SSWU + 3-isogeny map.
/// Returns a projective G2 point (36 x u64).
pub fn bls12_381_hash_to_g2(msg: &[u8], dst: &[u8]) -> [u64; 36] {
    let mut result = [0u64; 36];
    unsafe {
        zkmetal_sys::bls12_381_hash_to_g2(
            msg.as_ptr(),
            msg.len(),
            dst.as_ptr(),
            dst.len(),
            result.as_mut_ptr(),
        );
    }
    result
}

/// BLS12-381 hash-to-curve G2 with default DST.
pub fn bls12_381_hash_to_g2_default(msg: &[u8]) -> [u64; 36] {
    let mut result = [0u64; 36];
    unsafe {
        zkmetal_sys::bls12_381_hash_to_g2_default(
            msg.as_ptr(),
            msg.len(),
            result.as_mut_ptr(),
        );
    }
    result
}
