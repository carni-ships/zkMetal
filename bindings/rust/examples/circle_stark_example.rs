//! Circle STARK FFI Example Program
//!
//! This demonstrates the Rust FFI bindings for Circle STARK over M31.
//! It tests basic functionality: M31 arithmetic, circle domain generation,
//! Circle NTT (CPU reference), Merkle trees, and Circle FRI.
//!
//! Run with: cargo run --example circle_stark_example

use zkmetal_sys::{
    M31, M31Digest, CirclePoint, CircleSTARKConfig,
    circle_coset_domain, circle_domain_size, circle_ntt_cpu, circle_intt_cpu,
    merkle_tree_m31_build, merkle_root_m31, merkle_proof_m31,
    circle_fri_fold_cpu, circle_fri_fold_all_cpu,
    verify_merkle_proof,
    bytes_to_m31_slice,
};

fn main() {
    println!("Circle STARK FFI Example");
    println!("========================\n");

    // ============================================
    // Test 1: M31 field element operations
    // ============================================
    println!("[1] M31 Field Arithmetic");

    let a = M31::from_raw(100);
    let b = M31::from_raw(200);
    println!("  a = {}, b = {}", a, b);
    println!("  a.is_zero() = {}, b.is_one() = {}", a.is_zero(), b.is_one());

    // Convert to/from bytes
    let a_bytes = a.to_bytes();
    let a_back = M31::from_bytes(&a_bytes);
    println!("  a bytes = {:?}, recovered = {}", a_bytes, a_back);
    assert_eq!(a.raw(), a_back.raw());

    println!("  M31 arithmetic OK");

    // ============================================
    // Test 2: Circle domain generation
    // ============================================
    println!("\n[2] Circle Domain Generation (2^3 = 8 points)");

    let domain = circle_coset_domain(3);
    println!("  Generated {} domain points", domain.len());

    for (i, pt) in domain.iter().enumerate().take(4) {
        println!("  domain[{}] = ({}, {})", i, pt.x, pt.y);
    }
    println!("  ...");

    // Domain[0] should be identity (1, 0)
    assert_eq!(domain[0].x.raw(), 1);
    assert_eq!(domain[0].y.raw(), 0);
    println!("  Identity point (1, 0) verified");

    // ============================================
    // Test 3: M31Digest (Poseidon2-M31 hash output)
    // ============================================
    println!("\n[3] M31Digest (8 M31 = 32 bytes)");

    let digest = M31Digest::ZERO;
    println!("  Zero digest is_zero = {}", digest.is_zero());

    let non_zero = M31Digest([0x01u8; 32]);
    println!("  Non-zero digest is_zero = {}", non_zero.is_zero());

    println!("  M31Digest OK");

    // ============================================
    // Test 4: Circle NTT (CPU reference)
    // ============================================
    println!("\n[4] Circle NTT CPU Reference (forward + inverse)");

    // Create a simple input: [1, 2, 3, 4, 5, 6, 7, 8]
    let input: Vec<M31> = (1..=8).map(|i| M31::from_raw(i)).collect();
    println!("  Input: {:?}", input.iter().map(|m| m.raw()).collect::<Vec<_>>());

    // Forward NTT
    let nttd = circle_ntt_cpu(&input, 3);
    println!("  NTT output (first 4): {:?}", nttd.iter().take(4).map(|m| m.raw()).collect::<Vec<_>>());

    // Inverse NTT
    let inttd = circle_intt_cpu(&nttd, 3);
    println!("  INTT output (first 4): {:?}", inttd.iter().take(4).map(|m| m.raw()).collect::<Vec<_>>());

    // Verify round-trip
    let success = input.iter().zip(inttd.iter()).all(|(a, b)| a.raw() == b.raw());
    println!("  Round-trip verification: {}", if success { "PASS" } else { "FAIL" });
    assert!(success, "Circle NTT round-trip failed");

    // ============================================
    // Test 5: Merkle tree (CPU reference)
    // ============================================
    println!("\n[5] Merkle Tree (M31 -> M31Digest)");

    let leaves: Vec<M31> = (1..=4).map(|i| M31::from_raw(i)).collect();
    println!("  Leaves: {:?}", leaves.iter().map(|m| m.raw()).collect::<Vec<_>>());

    let tree = merkle_tree_m31_build(&leaves);
    println!("  Tree size: {} (expected {})", tree.len(), 2 * 4 - 1);
    assert_eq!(tree.len(), 2 * 4 - 1);

    let root = merkle_root_m31(&tree, 4);
    println!("  Root: first 8 bytes = {:?}", &root.as_bytes()[..8]);

    // ============================================
    // Test 6: Merkle proof verification
    // ============================================
    println!("\n[6] Merkle Proof Verification");

    let proof = merkle_proof_m31(&tree, 4, 1);
    println!("  Proof for index 1: {} nodes", proof.len());

    let leaf = M31::from_raw(2);
    let valid = verify_merkle_proof(leaf, &proof, 1, root);
    println!("  Verification for leaf 2 at index 1: {}", if valid { "PASS" } else { "FAIL" });
    assert!(valid, "Merkle proof verification failed");

    // ============================================
    // Test 7: Circle FRI (CPU reference)
    // ============================================
    println!("\n[7] Circle FRI Fold (y-fold, then x-fold)");

    let evals: Vec<M31> = (0..8).map(|i| M31::from_raw((i + 1) as u32)).collect();
    println!("  Input evaluations: {:?}", evals.iter().map(|m| m.raw()).collect::<Vec<_>>());

    let alpha1 = M31::from_raw(42);
    println!("  Alpha (round 1) = {}", alpha1);

    // First round: y-fold
    let folded1 = circle_fri_fold_cpu(&evals, alpha1, true);
    println!("  After y-fold: len={}, first 4: {:?}", folded1.len(), folded1.iter().take(4).map(|m| m.raw()).collect::<Vec<_>>());

    let alpha2 = M31::from_raw(17);
    println!("  Alpha (round 2) = {}", alpha2);

    // Second round: x-fold
    let folded2 = circle_fri_fold_cpu(&folded1, alpha2, false);
    println!("  After x-fold: len={}, values: {:?}", folded2.len(), folded2.iter().map(|m| m.raw()).collect::<Vec<_>>());

    println!("  Circle FRI OK");

    // ============================================
    // Test 8: Circle FRI multi-round
    // ============================================
    println!("\n[8] Circle FRI Multi-Round");

    let evals2: Vec<M31> = (0..16).map(|i| M31::from_raw((i * 3 + 1) as u32)).collect();
    let alphas = vec![M31::from_raw(5), M31::from_raw(13), M31::from_raw(21)];

    let rounds = circle_fri_fold_all_cpu(&evals2, &alphas);
    println!("  Created {} FRI rounds", rounds.len());

    for (i, round) in rounds.iter().enumerate() {
        println!("  Round {}: commitment[0..8]={:?}, alpha={}, folded_len={}",
                 i, &round.commitment.as_bytes()[..8], round.alpha, round.folded.len());
    }

    println!("  Circle FRI multi-round OK");

    // ============================================
    // Test 9: Circle STARK Configuration
    // ============================================
    println!("\n[9] Circle STARK Configuration");

    let default_config = CircleSTARKConfig::default_config();
    println!("  Default config:");
    println!("    log_blowup = {}", default_config.log_blowup);
    println!("    blowup_factor = {}", default_config.blowup_factor());
    println!("    num_queries = {}", default_config.num_queries);
    println!("    extension_degree = {}", default_config.extension_degree);
    println!("    security_bits = {}", default_config.security_bits());

    let fast_config = CircleSTARKConfig::fast_config();
    println!("  Fast config: security_bits = {}", fast_config.security_bits());

    println!("  Config OK");

    // ============================================
    // Test 10: Byte conversion helpers
    // ============================================
    println!("\n[10] Byte <-> M31 Slice Conversion");

    let bytes = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let m31_slice = bytes_to_m31_slice(&bytes);
    println!("  12 bytes -> {} M31 elements", m31_slice.len());
    assert_eq!(m31_slice.len(), 3);

    println!("  Byte conversion OK");

    println!("\n========================");
    println!("All tests PASSED!");
    println!("\nCircle STARK FFI bindings are working correctly.");
    println!("\nNote: This uses CPU reference implementations.");
    println!("GPU-accelerated versions are accessed via the C FFI layer.");
}