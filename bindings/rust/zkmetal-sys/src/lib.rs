//! Raw FFI bindings to the NeonFieldOps C library.
//!
//! This crate provides unsafe `extern "C"` function declarations matching
//! `Sources/NeonFieldOps/include/NeonFieldOps.h`. All functions require
//! `target_arch = "aarch64"` (Apple Silicon / ARM64).
//!
//! Field elements are represented as arrays of `u64` limbs in little-endian
//! Montgomery form unless otherwise noted. Scalars for curve operations
//! use `u32` limbs in non-Montgomery integer form.

#![no_std]
#![allow(non_camel_case_types)]

use core::ffi::c_int;

// =============================================================================
// BabyBear NTT (p = 0x78000001, 32-bit field)
// =============================================================================

extern "C" {
    pub fn babybear_ntt_neon(data: *mut u32, logN: c_int);
    pub fn babybear_intt_neon(data: *mut u32, logN: c_int);
}

// =============================================================================
// Goldilocks NTT (p = 2^64 - 2^32 + 1)
// =============================================================================

extern "C" {
    pub fn goldilocks_ntt(data: *mut u64, logN: c_int);
    pub fn goldilocks_intt(data: *mut u64, logN: c_int);
    pub fn goldilocks_ntt_neon(data: *mut u64, logN: c_int);
    pub fn goldilocks_intt_neon(data: *mut u64, logN: c_int);

    pub fn gl_batch_add_neon(a: *const u64, b: *const u64, out: *mut u64, n: c_int);
    pub fn gl_batch_sub_neon(a: *const u64, b: *const u64, out: *mut u64, n: c_int);
    pub fn gl_batch_mul_neon(a: *const u64, b: *const u64, out: *mut u64, n: c_int);
    pub fn gl_batch_butterfly_neon(data: *mut u64, twiddles: *const u64, halfBlock: c_int, nBlocks: c_int);
    pub fn gl_batch_butterfly_dif_neon(data: *mut u64, twiddles: *const u64, halfBlock: c_int, nBlocks: c_int);
}

// =============================================================================
// BN254 Fr field arithmetic (4 x u64 limbs, Montgomery form)
// =============================================================================

extern "C" {
    pub fn bn254_fr_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bn254_fr_sqr(a: *const u64, r: *mut u64);
    pub fn bn254_fr_add(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bn254_fr_sub(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bn254_fr_neg(a: *const u64, r: *mut u64);
    pub fn bn254_fr_inverse(a: *const u64, r: *mut u64);
    pub fn bn254_fr_pow(a: *const u64, exp: u64, r: *mut u64);
    pub fn bn254_fr_eq(a: *const u64, b: *const u64) -> c_int;
}

// =============================================================================
// BN254 Fr NTT
// =============================================================================

extern "C" {
    pub fn bn254_fr_ntt(data: *mut u64, logN: c_int);
    pub fn bn254_fr_intt(data: *mut u64, logN: c_int);
}

// =============================================================================
// BN254 Fr Montgomery mul (ASM and C variants)
// =============================================================================

extern "C" {
    pub fn mont_mul_asm(result: *mut u64, a: *const u64, b: *const u64);
    pub fn mont_mul_c(result: *mut u64, a: *const u64, b: *const u64);
    pub fn mont_mul_batch_asm(data: *mut u64, multiplier: *const u64, n: c_int);
    pub fn mont_mul_pair_batch_asm(result: *mut u64, a: *const u64, b: *const u64, n: c_int);
    pub fn mont_mul_batch_c(data: *mut u64, multiplier: *const u64, n: c_int);
    pub fn mont_mul_pair_batch_c(result: *mut u64, a: *const u64, b: *const u64, n: c_int);
    pub fn mont_mul_asm_test() -> c_int;
}

// =============================================================================
// BN254 Fr batch operations (NEON-optimized)
// =============================================================================

extern "C" {
    pub fn bn254_fr_batch_add_neon(result: *mut u64, a: *const u64, b: *const u64, n: c_int);
    pub fn bn254_fr_batch_sub_neon(result: *mut u64, a: *const u64, b: *const u64, n: c_int);
    pub fn bn254_fr_batch_neg_neon(result: *mut u64, a: *const u64, n: c_int);
    pub fn bn254_fr_batch_mul_scalar_neon(result: *mut u64, a: *const u64, scalar: *const u64, n: c_int);
    pub fn bn254_fr_batch_add_parallel(result: *mut u64, a: *const u64, b: *const u64, n: c_int);
    pub fn bn254_fr_batch_sub_parallel(result: *mut u64, a: *const u64, b: *const u64, n: c_int);
    pub fn bn254_fr_batch_neg_parallel(result: *mut u64, a: *const u64, n: c_int);
    pub fn bn254_fr_batch_mul_scalar_parallel(result: *mut u64, a: *const u64, scalar: *const u64, n: c_int);
}

// =============================================================================
// BN254 Fr vector/polynomial operations
// =============================================================================

extern "C" {
    pub fn bn254_fr_vector_sum(a: *const u64, n: c_int, result: *mut u64);
    pub fn bn254_fr_inner_product(a: *const u64, b: *const u64, n: c_int, result: *mut u64);
    pub fn bn254_fr_vector_fold(a: *const u64, b: *const u64, x: *const u64, x_inv: *const u64, n: c_int, out: *mut u64);
    pub fn bn254_fr_batch_to_limbs(mont: *const u64, limbs: *mut u32, n: c_int);
    pub fn bn254_fr_synthetic_div(coeffs: *const u64, z: *const u64, n: c_int, quotient: *mut u64);
    pub fn bn254_fr_horner_eval(coeffs: *const u64, n: c_int, z: *const u64, result: *mut u64);
    pub fn bn254_fr_eval_and_div(coeffs: *const u64, n: c_int, z: *const u64, eval_out: *mut u64, quotient: *mut u64);
    pub fn bn254_fr_batch_inverse(a: *const u64, n: c_int, out: *mut u64);
    pub fn bn254_fr_batch_beta_add(beta: *const u64, values: *const u64, indices: *const c_int, m: c_int, result: *mut u64);
    pub fn bn254_fr_batch_decompose(lookups: *const u64, m: c_int, num_chunks: c_int, bits_per_chunk: c_int, indices: *mut c_int);
}

// =============================================================================
// BN254 Fr fused ops
// =============================================================================

extern "C" {
    pub fn bn254_fr_pointwise_mul_sub(a: *const u64, b: *const u64, c: *const u64, result: *mut u64, n: c_int);
    pub fn bn254_fr_coeff_div_vanishing(p_coeffs: *const u64, domain_n: c_int, h_coeffs: *mut u64);
    pub fn bn254_fr_linear_combine(running: *const u64, new_vals: *const u64, rho: *const u64, result: *mut u64, count: c_int);
    pub fn bn254_fr_basefold_fold(evals: *const u64, result: *mut u64, alpha: *const u64, half_n: u32);
    pub fn bn254_fr_basefold_fold_all(evals: *const u64, num_vars: c_int, point: *const u64, out_layers: *mut u64);
    pub fn bn254_fr_whir_fold(evals: *const u64, n: c_int, beta: *const u64, reduction_factor: c_int, result: *mut u64);
}

// =============================================================================
// BN254 Fp operations
// =============================================================================

extern "C" {
    pub fn bn254_fp_sqr(a: *const u64, r: *mut u64);
    pub fn bn254_fp_inv(a: *const u64, r: *mut u64);
    pub fn bn254_fp_sqrt(a: *const u64, r: *mut u64) -> c_int;
}

// =============================================================================
// BN254 G1 curve operations (Jacobian projective, 12 x u64)
// =============================================================================

extern "C" {
    pub fn bn254_point_add(p: *const u64, q: *const u64, r: *mut u64);
    pub fn bn254_point_add_mixed(p: *const u64, q_aff: *const u64, r: *mut u64);
    pub fn bn254_point_scalar_mul(p: *const u64, scalar: *const u32, r: *mut u64);
    pub fn bn254_projective_to_affine(p: *const u64, affine: *mut u64);
    pub fn bn254_batch_to_affine(proj: *const u64, aff: *mut u64, n: c_int);
}

// =============================================================================
// BN254 MSM (Pippenger)
// =============================================================================

extern "C" {
    pub fn bn254_pippenger_msm(points: *const u64, scalars: *const u32, n: c_int, result: *mut u64);
    pub fn bn254_msm_projective(points: *const u64, scalars: *const u32, n: c_int, result: *mut u64);
    pub fn bn254_dual_msm_projective(
        points1: *const u64, scalars1: *const u32, n1: c_int,
        points2: *const u64, scalars2: *const u32, n2: c_int,
        result1: *mut u64, result2: *mut u64,
    );
    pub fn bn254_fold_generators(gl: *const u64, gr: *const u64, x: *const u32, x_inv: *const u32, half_len: c_int, result: *mut u64);
}

// =============================================================================
// BGMW fixed-base MSM
// =============================================================================

extern "C" {
    pub fn bgmw_precompute(generators_affine: *const u64, n: c_int, window_bits: c_int, table_out: *mut u64);
    pub fn bgmw_msm(table: *const u64, n: c_int, window_bits: c_int, scalars: *const u32, result: *mut u64);
}

// =============================================================================
// BN254 Pairing
// =============================================================================

extern "C" {
    pub fn bn254_miller_loop(p_aff: *const u64, q_aff: *const u64, result: *mut u64);
    pub fn bn254_final_exp(f: *const u64, result: *mut u64);
    pub fn bn254_pairing(p_aff: *const u64, q_aff: *const u64, result: *mut u64);
    pub fn bn254_pairing_check(pairs: *const u64, n: c_int) -> c_int;
}

// =============================================================================
// BN254 KZG batch verification
// =============================================================================

extern "C" {
    pub fn bn254_batch_kzg_verify(
        srs_g1: *const u64, srs_secret: *const u64,
        commitments: *const u64, points: *const u64,
        evaluations: *const u64, witnesses: *const u64,
        batch_challenge: *const u64, n: c_int,
    ) -> c_int;
}

// =============================================================================
// BN254 sumcheck and MLE
// =============================================================================

extern "C" {
    pub fn bn254_fr_full_sumcheck(evals: *const u64, num_vars: c_int, challenges: *const u64, rounds: *mut u64, final_eval: *mut u64);
    pub fn bn254_fr_mle_eval(evals: *const u64, num_vars: c_int, point: *const u64, result: *mut u64);
    pub fn bn254_fr_inverse_evals(beta: *const u64, values: *const u64, n: c_int, out: *mut u64);
    pub fn bn254_fr_inverse_evals_indexed(beta: *const u64, subtable: *const u64, indices: *const c_int, n: c_int, out: *mut u64);
    pub fn bn254_fr_weighted_inverse_evals(beta: *const u64, values: *const u64, weights: *const u64, n: c_int, out: *mut u64);
    pub fn bn254_fr_inverse_mle_eval(beta: *const u64, subtable: *const u64, indices: *const c_int, n: c_int, num_vars: c_int, point: *const u64, result: *mut u64);
}

// =============================================================================
// GKR operations
// =============================================================================

extern "C" {
    pub fn gkr_eq_poly(point: *const u64, n: c_int, eq: *mut u64);
    pub fn gkr_accumulate_wiring(
        gates: *const i32, num_gates: c_int,
        eq_vals: *const u64, weight: *const u64, in_size: c_int,
        accum: *mut u64, accum_capacity: c_int,
        nonzero_indices: *mut i32, num_nonzero: *mut c_int,
    );
    pub fn gkr_mle_fold(v: *mut u64, half: c_int, challenge: *const u64);
    pub fn gkr_sumcheck_round_x(
        wiring: *const u64, num_entries: c_int,
        vx: *const u64, vx_size: c_int, vy: *const u64, vy_size: c_int,
        n_in: c_int, half_size: c_int,
        s0: *mut u64, s1: *mut u64, s2: *mut u64,
    );
    pub fn gkr_sumcheck_round_y(
        wiring: *const u64, num_entries: c_int, vx_scalar: *const u64,
        vy: *const u64, vy_size: c_int, half_size: c_int,
        s0: *mut u64, s1: *mut u64, s2: *mut u64,
    );
    pub fn gkr_wiring_reduce(wiring: *const u64, num_entries: c_int, challenge: *const u64, half_size: c_int, out_wiring: *mut u64) -> c_int;
    pub fn gkr_sumcheck_step(
        wiring: *const u64, num_entries: c_int,
        cur_vx: *const u64, vx_size: c_int, cur_vy: *const u64, vy_size: c_int,
        round: c_int, n_in: c_int, current_table_size: c_int,
        s0: *mut u64, s1: *mut u64, s2: *mut u64,
    );
}

// =============================================================================
// Spartan operations
// =============================================================================

extern "C" {
    pub fn spartan_sparse_matvec(entries: *const u64, num_entries: c_int, z: *const u64, z_len: c_int, result: *mut u64, num_rows: c_int);
    pub fn spartan_sc1_round(eq_c: *mut u64, az_c: *mut u64, bz_c: *mut u64, cz_c: *mut u64, half_size: c_int, s0: *mut u64, s1: *mut u64, s2: *mut u64, s3: *mut u64);
    pub fn spartan_fold_array(arr: *mut u64, half_size: c_int, ri: *const u64);
    pub fn spartan_sc2_round(w_c: *mut u64, z_c: *mut u64, half_size: c_int, s0: *mut u64, s1: *mut u64, s2: *mut u64);
    pub fn spartan_build_weight_vec(entries: *const u64, num_entries: c_int, eq_rx: *const u64, eq_rx_len: c_int, weight: *const u64, w_vec: *mut u64, padded_n: c_int);
    pub fn spartan_eq_poly(point: *const u64, n: c_int, eq: *mut u64);
    pub fn spartan_mle_eval(evals: *const u64, num_vars: c_int, point: *const u64, result: *mut u64);
}

// =============================================================================
// CCS (Customizable Constraint System)
// =============================================================================

extern "C" {
    pub fn ccs_sparse_matvec(result: *mut u64, row_ptr: *const c_int, col_idx: *const c_int, values: *const u64, z: *const u64, n_rows: c_int);
    pub fn ccs_hadamard_accumulate(
        acc: *mut u64, mat_result_ptrs: *const *const u64, n_matrices_per_term: *const c_int,
        coefficients: *const u64, n_terms: c_int, max_degree: c_int, m: c_int,
    );
    pub fn ccs_compute_term(result: *mut u64, mat_vec_results: *const *const u64, n_matrices: c_int, coeff: *const u64, m: c_int);
}

// =============================================================================
// Tensor proof compression
// =============================================================================

extern "C" {
    pub fn tensor_mat_vec_mul(matrix: *const u64, vec: *const u64, rows: c_int, cols: c_int, result: *mut u64);
    pub fn tensor_inner_product_sumcheck(evals_a: *const u64, evals_b: *const u64, num_vars: c_int, challenges: *const u64, rounds: *mut u64, final_eval: *mut u64);
    pub fn tensor_eq_weighted_row(matrix: *const u64, row_point: *const u64, rows: c_int, cols: c_int, result: *mut u64);
}

// =============================================================================
// Sparse matvec + MLE
// =============================================================================

extern "C" {
    pub fn bn254_sparse_matvec_mle(row_ptr: *const c_int, col_idx: *const c_int, values: *const u64, rows: c_int, z: *const u64, point: *const u64, num_vars: c_int, pad_m: c_int, result: *mut u64);
}

// =============================================================================
// Plonk permutation Z accumulator
// =============================================================================

extern "C" {
    pub fn plonk_compute_z_accumulator(
        a_evals: *const u64, b_evals: *const u64, c_evals: *const u64,
        sigma1: *const u64, sigma2: *const u64, sigma3: *const u64,
        domain: *const u64, beta: *const u64, gamma: *const u64,
        k1: *const u64, k2: *const u64, n: c_int, z_evals: *mut u64,
    );
}

// =============================================================================
// Hashing: Keccak-256, Blake3
// =============================================================================

extern "C" {
    pub fn keccak_f1600_neon(state: *mut u64);
    pub fn keccak256_hash_neon(input: *const u8, len: usize, output: *mut u8);
    pub fn keccak256_hash_pair_neon(a: *const u8, b: *const u8, output: *mut u8);
    pub fn keccak256_batch_hash_pairs_neon(inputs: *const u8, outputs: *mut u8, n: usize);
    pub fn blake3_hash_neon(input: *const u8, len: usize, output: *mut u8);
    pub fn blake3_hash_pair_neon(left: *const u8, right: *const u8, output: *mut u8);
    pub fn blake3_batch_hash_pairs_neon(inputs: *const u8, outputs: *mut u8, n: usize);
}

// =============================================================================
// Poseidon2 (BN254 Fr, CPU)
// =============================================================================

extern "C" {
    pub fn poseidon2_permutation_cpu(state: *const u64, result: *mut u64);
    pub fn poseidon2_hash_cpu(a: *const u64, b: *const u64, out: *mut u64);
    pub fn poseidon2_hash_batch_cpu(input: *const u64, count: c_int, output: *mut u64);
    pub fn poseidon2_merkle_tree_cpu(leaves: *const u64, n: c_int, tree: *mut u64);
}

// =============================================================================
// secp256k1 curve operations
// =============================================================================

extern "C" {
    pub fn secp256k1_point_scalar_mul(p: *const u64, scalar: *const u64, r: *mut u64);
    pub fn secp256k1_point_add(p: *const u64, q: *const u64, r: *mut u64);
    pub fn secp256k1_point_to_affine(p: *const u64, ax: *mut u64, ay: *mut u64);
    pub fn secp256k1_pippenger_msm(points: *const u64, scalars: *const u32, n: c_int, result: *mut u64);
    pub fn secp256k1_fr_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn secp256k1_fr_inverse(a: *const u64, r: *mut u64);
    pub fn secp256k1_fr_batch_inverse(a: *const u64, n: c_int, out: *mut u64);
    pub fn secp256k1_shamir_double_mul(p1: *const u64, s1: *const u64, p2: *const u64, s2: *const u64, r: *mut u64);
    pub fn secp256k1_ecdsa_batch_prepare(sigs: *const u64, pubkeys: *const u64, recov: *const u8, n: c_int, out_points: *mut u64, out_scalars: *mut u32) -> c_int;
}

// =============================================================================
// Grumpkin curve (y^2 = x^3 - 17 over BN254 Fr)
// =============================================================================

extern "C" {
    pub fn grumpkin_point_add(p: *const u64, q: *const u64, r: *mut u64);
    pub fn grumpkin_point_double(p: *const u64, r: *mut u64);
    pub fn grumpkin_scalar_mul(p: *const u64, scalar: *const u64, r: *mut u64);
    pub fn grumpkin_point_add_mixed(p: *const u64, q_aff: *const u64, r: *mut u64);
    pub fn grumpkin_pippenger_msm(points: *const u64, scalars: *const u32, n: c_int, result: *mut u64);
}

// =============================================================================
// Ed25519
// =============================================================================

extern "C" {
    pub fn ed25519_fp_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn ed25519_fp_sqr(a: *const u64, r: *mut u64);
    pub fn ed25519_fp_add(a: *const u64, b: *const u64, r: *mut u64);
    pub fn ed25519_fp_sub(a: *const u64, b: *const u64, r: *mut u64);
    pub fn ed25519_fp_neg(a: *const u64, r: *mut u64);
    pub fn ed25519_fp_inverse(a: *const u64, r: *mut u64);
    pub fn ed25519_scalar_mul(p: *const u64, scalar: *const u64, r: *mut u64);
    pub fn ed25519_point_add_c(p: *const u64, q: *const u64, r: *mut u64);
    pub fn ed25519_point_double_c(p: *const u64, r: *mut u64);
    pub fn ed25519_point_to_affine(p: *const u64, aff: *mut u64);
    pub fn ed25519_mont_to_direct(mont: *const u64, direct: *mut u64);
    pub fn ed25519_direct_to_mont(direct: *const u64, mont: *mut u64);
    pub fn ed25519_pippenger_msm(points: *const u64, scalars: *const u32, n: c_int, result: *mut u64);
    pub fn ed25519_fq_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn ed25519_fq_add(a: *const u64, b: *const u64, r: *mut u64);
    pub fn ed25519_fq_sub(a: *const u64, b: *const u64, r: *mut u64);
    pub fn ed25519_fq_from_raw(raw: *const u64, mont: *mut u64);
    pub fn ed25519_fq_to_raw(mont: *const u64, raw: *mut u64);
    pub fn ed25519_fq_from_bytes64(bytes: *const u8, mont: *mut u64);
    pub fn ed25519_fq_to_bytes(mont: *const u64, bytes: *mut u8);
    pub fn ed25519_shamir_double_mul(g: *const u64, s: *const u64, a: *const u64, h: *const u64, result: *mut u64);
    pub fn ed25519_eddsa_sign_compute_r(gen: *const u64, r_scalar: *const u64, r_point: *mut u64);
    pub fn ed25519_eddsa_sign_compute_s(r_mont: *const u64, k_mont: *const u64, a_mont: *const u64, s_mont: *mut u64);
    pub fn ed25519_eddsa_verify(gen: *const u64, s_raw: *const u64, r_point: *const u64, h_raw: *const u64, pub_key: *const u64) -> c_int;
}

// =============================================================================
// BabyJubjub twisted Edwards curve (over BN254 Fr)
// =============================================================================

extern "C" {
    pub fn babyjubjub_point_add(p: *const u64, q: *const u64, r: *mut u64);
    pub fn babyjubjub_point_double(p: *const u64, r: *mut u64);
    pub fn babyjubjub_scalar_mul(p: *const u64, scalar: *const u64, r: *mut u64);
}

// =============================================================================
// Pallas curve (y^2 = x^3 + 5)
// =============================================================================

extern "C" {
    pub fn pallas_fp_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn pallas_fp_sqr(a: *const u64, r: *mut u64);
    pub fn pallas_fp_add(a: *const u64, b: *const u64, r: *mut u64);
    pub fn pallas_fp_sub(a: *const u64, b: *const u64, r: *mut u64);
    pub fn pallas_fp_neg(a: *const u64, r: *mut u64);
    pub fn pallas_point_add(p: *const u64, q: *const u64, r: *mut u64);
    pub fn pallas_point_double(p: *const u64, r: *mut u64);
    pub fn pallas_point_add_mixed(p: *const u64, q_aff: *const u64, r: *mut u64);
    pub fn pallas_scalar_mul(p: *const u64, scalar: *const u64, r: *mut u64);
    pub fn pallas_pippenger_msm(points: *const u64, scalars: *const u32, n: c_int, result: *mut u64);
}

// =============================================================================
// Vesta curve (y^2 = x^3 + 5)
// =============================================================================

extern "C" {
    pub fn vesta_fp_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn vesta_fp_sqr(a: *const u64, r: *mut u64);
    pub fn vesta_fp_add(a: *const u64, b: *const u64, r: *mut u64);
    pub fn vesta_fp_sub(a: *const u64, b: *const u64, r: *mut u64);
    pub fn vesta_fp_neg(a: *const u64, r: *mut u64);
    pub fn vesta_point_add(p: *const u64, q: *const u64, r: *mut u64);
    pub fn vesta_point_double(p: *const u64, r: *mut u64);
    pub fn vesta_point_add_mixed(p: *const u64, q_aff: *const u64, r: *mut u64);
    pub fn vesta_scalar_mul(p: *const u64, scalar: *const u64, r: *mut u64);
    pub fn vesta_pippenger_msm(points: *const u64, scalars: *const u32, n: c_int, result: *mut u64);
}

// =============================================================================
// BLS12-381 Fr (scalar field, 4 x u64)
// =============================================================================

extern "C" {
    pub fn bls12_381_fr_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_381_fr_sqr(a: *const u64, r: *mut u64);
    pub fn bls12_381_fr_add(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_381_fr_sub(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_381_fr_neg(a: *const u64, r: *mut u64);
}

// =============================================================================
// BLS12-381 Fp (base field, 6 x u64)
// =============================================================================

extern "C" {
    pub fn bls12_381_fp_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_381_fp_sqr(a: *const u64, r: *mut u64);
    pub fn bls12_381_fp_add(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_381_fp_sub(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_381_fp_neg(a: *const u64, r: *mut u64);
    pub fn bls12_381_fp_inv_ext(a: *const u64, r: *mut u64);
    pub fn bls12_381_fp_sqrt(a: *const u64, r: *mut u64) -> c_int;
}

// =============================================================================
// BLS12-381 Fp2 tower (12 x u64 per element)
// =============================================================================

extern "C" {
    pub fn bls12_381_fp2_add(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_381_fp2_sub(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_381_fp2_neg(a: *const u64, r: *mut u64);
    pub fn bls12_381_fp2_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_381_fp2_sqr(a: *const u64, r: *mut u64);
    pub fn bls12_381_fp2_conj(a: *const u64, r: *mut u64);
    pub fn bls12_381_fp2_mul_by_nonresidue(a: *const u64, r: *mut u64);
    pub fn bls12_381_fp2_inv(a: *const u64, r: *mut u64);
}

// =============================================================================
// BLS12-381 higher tower (Fp6, Fp12)
// =============================================================================

extern "C" {
    pub fn bls12_381_fp6_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_381_fp6_sqr(a: *const u64, r: *mut u64);
    pub fn bls12_381_fp12_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_381_fp12_sqr(a: *const u64, r: *mut u64);
    pub fn bls12_381_fp12_inv(a: *const u64, r: *mut u64);
    pub fn bls12_381_fp12_conj(a: *const u64, r: *mut u64);
}

// =============================================================================
// BLS12-381 G1 curve ops (Jacobian projective, 18 x u64)
// =============================================================================

extern "C" {
    pub fn bls12_381_g1_point_add(p: *const u64, q: *const u64, r: *mut u64);
    pub fn bls12_381_g1_point_double(p: *const u64, r: *mut u64);
    pub fn bls12_381_g1_point_add_mixed(p: *const u64, q_aff: *const u64, r: *mut u64);
    pub fn bls12_381_g1_scalar_mul(p: *const u64, scalar: *const u64, r: *mut u64);
    pub fn bls12_381_g1_pippenger_msm(points: *const u64, scalars: *const u32, n: c_int, result: *mut u64);
}

// =============================================================================
// BLS12-381 G2 curve ops (Fp2 coords, 36 x u64 projective)
// =============================================================================

extern "C" {
    pub fn bls12_381_g2_point_add(p: *const u64, q: *const u64, r: *mut u64);
    pub fn bls12_381_g2_point_double(p: *const u64, r: *mut u64);
    pub fn bls12_381_g2_point_add_mixed(p: *const u64, q_aff: *const u64, r: *mut u64);
    pub fn bls12_381_g2_scalar_mul(p: *const u64, scalar: *const u64, r: *mut u64);
    pub fn bls12_381_g2_scalar_mul_wide(p: *const u64, scalar: *const u64, n_limbs: c_int, r: *mut u64);
}

// =============================================================================
// BLS12-381 Pairing
// =============================================================================

extern "C" {
    pub fn bls12_381_miller_loop(p_aff: *const u64, q_aff: *const u64, result: *mut u64);
    pub fn bls12_381_final_exp(f: *const u64, result: *mut u64);
    pub fn bls12_381_pairing(p_aff: *const u64, q_aff: *const u64, result: *mut u64);
    pub fn bls12_381_pairing_check(pairs: *const u64, n: c_int) -> c_int;
}

// =============================================================================
// BLS12-381 hash-to-curve G2
// =============================================================================

extern "C" {
    pub fn bls12_381_hash_to_g2(msg: *const u8, msg_len: usize, dst: *const u8, dst_len: usize, result: *mut u64);
    pub fn bls12_381_hash_to_g2_default(msg: *const u8, msg_len: usize, result: *mut u64);
    pub fn bls12_381_g2_clear_cofactor(p: *const u64, r: *mut u64);
}

// =============================================================================
// Jubjub twisted Edwards curve (over BLS12-381 Fr)
// =============================================================================

extern "C" {
    pub fn jubjub_point_add(p: *const u64, q: *const u64, r: *mut u64);
    pub fn jubjub_point_double(p: *const u64, r: *mut u64);
    pub fn jubjub_scalar_mul(p: *const u64, scalar: *const u64, r: *mut u64);
}

// =============================================================================
// BLS12-377 Fq (base field, 6 x u64)
// =============================================================================

extern "C" {
    pub fn bls12_377_fq_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_377_fq_sqr(a: *const u64, r: *mut u64);
    pub fn bls12_377_fq_add(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_377_fq_sub(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_377_fq_neg(a: *const u64, r: *mut u64);
    pub fn bls12_377_fq_inverse(a: *const u64, r: *mut u64);
}

// =============================================================================
// BLS12-377 Fr (scalar field, 4 x u64)
// =============================================================================

extern "C" {
    pub fn bls12_377_fr_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_377_fr_sqr(a: *const u64, r: *mut u64);
    pub fn bls12_377_fr_add(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_377_fr_sub(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bls12_377_fr_neg(a: *const u64, r: *mut u64);
    pub fn bls12_377_fr_ntt(data: *mut u64, logN: c_int);
    pub fn bls12_377_fr_intt(data: *mut u64, logN: c_int);
}

// =============================================================================
// BLS12-377 G1 curve ops
// =============================================================================

extern "C" {
    pub fn bls12_377_g1_point_add(p: *const u64, q: *const u64, r: *mut u64);
    pub fn bls12_377_g1_point_double(p: *const u64, r: *mut u64);
    pub fn bls12_377_g1_point_add_mixed(p: *const u64, q_aff: *const u64, r: *mut u64);
    pub fn bls12_377_g1_scalar_mul(p: *const u64, scalar: *const u64, r: *mut u64);
    pub fn bls12_377_g1_to_affine(p: *const u64, aff: *mut u64);
    pub fn bls12_377_g1_pippenger_msm(points: *const u64, scalars: *const u32, n: c_int, result: *mut u64);
}

// =============================================================================
// Stark252 field (p = 2^251 + 17*2^192 + 1, 4 x u64)
// =============================================================================

extern "C" {
    pub fn stark252_fp_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn stark252_fp_sqr(a: *const u64, r: *mut u64);
    pub fn stark252_fp_add(a: *const u64, b: *const u64, r: *mut u64);
    pub fn stark252_fp_sub(a: *const u64, b: *const u64, r: *mut u64);
    pub fn stark252_fp_neg(a: *const u64, r: *mut u64);
    pub fn stark252_ntt(data: *mut u64, logN: c_int);
    pub fn stark252_intt(data: *mut u64, logN: c_int);
}

// =============================================================================
// Binary tower field arithmetic (GF(2) towers for Binius STARKs)
// =============================================================================

extern "C" {
    pub fn bt_gf8_init();
    pub fn bt_gf8_mul(a: u8, b: u8) -> u8;
    pub fn bt_gf8_sqr(a: u8) -> u8;
    pub fn bt_gf8_inv(a: u8) -> u8;
    pub fn bt_gf16_mul(a: u16, b: u16) -> u16;
    pub fn bt_gf16_sqr(a: u16) -> u16;
    pub fn bt_gf32_mul(a: u32, b: u32) -> u32;
    pub fn bt_gf32_sqr(a: u32) -> u32;
    pub fn bt_gf64_mul(a: u64, b: u64) -> u64;
    pub fn bt_gf64_sqr(a: u64) -> u64;
    pub fn bt_gf64_inv(a: u64) -> u64;
    pub fn bt_gf128_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bt_gf128_sqr(a: *const u64, r: *mut u64);
    pub fn bt_gf128_add(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bt_gf128_inv(a: *const u64, r: *mut u64);
    pub fn bt_tower128_mul(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bt_tower128_sqr(a: *const u64, r: *mut u64);
    pub fn bt_tower128_add(a: *const u64, b: *const u64, r: *mut u64);
    pub fn bt_tower128_inv(a: *const u64, r: *mut u64);
    pub fn bt_gf64_batch_mul(a: *const u64, b: *const u64, out: *mut u64, n: c_int);
    pub fn bt_gf64_batch_add(a: *const u64, b: *const u64, out: *mut u64, n: c_int);
    pub fn bt_gf64_batch_sqr(a: *const u64, out: *mut u64, n: c_int);
    pub fn bt_gf128_batch_mul(a: *const u64, b: *const u64, out: *mut u64, n: c_int);
    pub fn bt_gf128_batch_add(a: *const u64, b: *const u64, out: *mut u64, n: c_int);
    pub fn bt_tower128_batch_mul(a: *const u64, b: *const u64, out: *mut u64, n: c_int);
}
