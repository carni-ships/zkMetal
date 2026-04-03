//! Raw FFI bindings to zkMetal GPU-accelerated ZK primitives.
//!
//! This crate provides unsafe C bindings to zkMetal's Metal GPU engines.
//! For a safe API, use the `zkmetal` crate instead.
//!
//! **Platform:** macOS (Apple Silicon) only.

#![allow(non_camel_case_types)]

use std::os::raw::c_char;

// Error codes
pub const ZKMETAL_SUCCESS: i32 = 0;
pub const ZKMETAL_ERR_NO_GPU: i32 = -1;
pub const ZKMETAL_ERR_INVALID_INPUT: i32 = -2;
pub const ZKMETAL_ERR_GPU_ERROR: i32 = -3;
pub const ZKMETAL_ERR_ALLOC_FAILED: i32 = -4;

// Opaque engine handles
pub type ZkMetalMSMEngine = *mut std::ffi::c_void;
pub type ZkMetalNTTEngine = *mut std::ffi::c_void;
pub type ZkMetalPoseidon2Engine = *mut std::ffi::c_void;

extern "C" {
    // Engine lifecycle
    pub fn zkmetal_msm_engine_create(out: *mut ZkMetalMSMEngine) -> i32;
    pub fn zkmetal_msm_engine_destroy(engine: ZkMetalMSMEngine);
    pub fn zkmetal_ntt_engine_create(out: *mut ZkMetalNTTEngine) -> i32;
    pub fn zkmetal_ntt_engine_destroy(engine: ZkMetalNTTEngine);
    pub fn zkmetal_poseidon2_engine_create(out: *mut ZkMetalPoseidon2Engine) -> i32;
    pub fn zkmetal_poseidon2_engine_destroy(engine: ZkMetalPoseidon2Engine);

    // MSM
    pub fn zkmetal_bn254_msm(
        engine: ZkMetalMSMEngine,
        points: *const u8,
        scalars: *const u8,
        n_points: u32,
        result_x: *mut u8,
        result_y: *mut u8,
        result_z: *mut u8,
    ) -> i32;

    // NTT
    pub fn zkmetal_bn254_ntt(engine: ZkMetalNTTEngine, data: *mut u8, log_n: u32) -> i32;
    pub fn zkmetal_bn254_intt(engine: ZkMetalNTTEngine, data: *mut u8, log_n: u32) -> i32;

    // Poseidon2
    pub fn zkmetal_bn254_poseidon2_hash_pairs(
        engine: ZkMetalPoseidon2Engine,
        input: *const u8,
        n_pairs: u32,
        output: *mut u8,
    ) -> i32;

    // Utility
    pub fn zkmetal_set_shader_dir(path: *const c_char);
    pub fn zkmetal_gpu_available() -> i32;
    pub fn zkmetal_version() -> *const c_char;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_available() {
        unsafe {
            let avail = zkmetal_gpu_available();
            assert!(avail == 0 || avail == 1);
            println!("GPU available: {}", avail == 1);
        }
    }

    #[test]
    fn test_version() {
        unsafe {
            let v = zkmetal_version();
            assert!(!v.is_null());
            let s = std::ffi::CStr::from_ptr(v).to_str().unwrap();
            assert_eq!(s, "0.1.0");
            println!("zkMetal version: {}", s);
        }
    }
}
