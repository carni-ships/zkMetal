//! Number Theoretic Transform (NTT) for supported fields.
//!
//! All NTT functions operate in-place on the data array.

/// BN254 Fr forward NTT (Cooley-Tukey DIT).
///
/// - `data`: mutable slice of n * 4 u64 values (n elements in Montgomery form).
/// - `log_n`: log2 of the transform size. n = 2^log_n.
///
/// # Panics
/// Panics if `data.len() < (1 << log_n) * 4`.
pub fn bn254_fr_ntt(data: &mut [u64], log_n: u32) {
    let n = 1usize << log_n;
    assert!(data.len() >= n * 4, "ntt: data buffer too small");
    unsafe { zkmetal_sys::bn254_fr_ntt(data.as_mut_ptr(), log_n as core::ffi::c_int) };
}

/// BN254 Fr inverse NTT (Gentleman-Sande DIF + bit-reversal + 1/n scaling).
pub fn bn254_fr_intt(data: &mut [u64], log_n: u32) {
    let n = 1usize << log_n;
    assert!(data.len() >= n * 4, "intt: data buffer too small");
    unsafe { zkmetal_sys::bn254_fr_intt(data.as_mut_ptr(), log_n as core::ffi::c_int) };
}

/// BLS12-377 Fr forward NTT.
pub fn bls12_377_fr_ntt(data: &mut [u64], log_n: u32) {
    let n = 1usize << log_n;
    assert!(data.len() >= n * 4, "ntt: data buffer too small");
    unsafe { zkmetal_sys::bls12_377_fr_ntt(data.as_mut_ptr(), log_n as core::ffi::c_int) };
}

/// BLS12-377 Fr inverse NTT.
pub fn bls12_377_fr_intt(data: &mut [u64], log_n: u32) {
    let n = 1usize << log_n;
    assert!(data.len() >= n * 4, "intt: data buffer too small");
    unsafe { zkmetal_sys::bls12_377_fr_intt(data.as_mut_ptr(), log_n as core::ffi::c_int) };
}

/// Goldilocks forward NTT (scalar, optimized ARM64).
pub fn goldilocks_ntt(data: &mut [u64], log_n: u32) {
    let n = 1usize << log_n;
    assert!(data.len() >= n, "ntt: data buffer too small");
    unsafe { zkmetal_sys::goldilocks_ntt(data.as_mut_ptr(), log_n as core::ffi::c_int) };
}

/// Goldilocks inverse NTT.
pub fn goldilocks_intt(data: &mut [u64], log_n: u32) {
    let n = 1usize << log_n;
    assert!(data.len() >= n, "intt: data buffer too small");
    unsafe { zkmetal_sys::goldilocks_intt(data.as_mut_ptr(), log_n as core::ffi::c_int) };
}

/// Goldilocks forward NTT (NEON-vectorized).
pub fn goldilocks_ntt_neon(data: &mut [u64], log_n: u32) {
    let n = 1usize << log_n;
    assert!(data.len() >= n, "ntt: data buffer too small");
    unsafe { zkmetal_sys::goldilocks_ntt_neon(data.as_mut_ptr(), log_n as core::ffi::c_int) };
}

/// Goldilocks inverse NTT (NEON-vectorized).
pub fn goldilocks_intt_neon(data: &mut [u64], log_n: u32) {
    let n = 1usize << log_n;
    assert!(data.len() >= n, "intt: data buffer too small");
    unsafe { zkmetal_sys::goldilocks_intt_neon(data.as_mut_ptr(), log_n as core::ffi::c_int) };
}

/// BabyBear forward NTT (NEON).
pub fn babybear_ntt(data: &mut [u32], log_n: u32) {
    let n = 1usize << log_n;
    assert!(data.len() >= n, "ntt: data buffer too small");
    unsafe { zkmetal_sys::babybear_ntt_neon(data.as_mut_ptr(), log_n as core::ffi::c_int) };
}

/// BabyBear inverse NTT (NEON).
pub fn babybear_intt(data: &mut [u32], log_n: u32) {
    let n = 1usize << log_n;
    assert!(data.len() >= n, "intt: data buffer too small");
    unsafe { zkmetal_sys::babybear_intt_neon(data.as_mut_ptr(), log_n as core::ffi::c_int) };
}

/// Stark252 forward NTT.
pub fn stark252_ntt(data: &mut [u64], log_n: u32) {
    let n = 1usize << log_n;
    assert!(data.len() >= n * 4, "ntt: data buffer too small");
    unsafe { zkmetal_sys::stark252_ntt(data.as_mut_ptr(), log_n as core::ffi::c_int) };
}

/// Stark252 inverse NTT.
pub fn stark252_intt(data: &mut [u64], log_n: u32) {
    let n = 1usize << log_n;
    assert!(data.len() >= n * 4, "intt: data buffer too small");
    unsafe { zkmetal_sys::stark252_intt(data.as_mut_ptr(), log_n as core::ffi::c_int) };
}
