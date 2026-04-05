//! Cryptographic hash functions: Poseidon2, Keccak-256, Blake3.

/// Poseidon2 2-to-1 hash compression over BN254 Fr.
///
/// Hashes two Fr elements (each 4 x u64, Montgomery form) into one Fr element.
pub fn poseidon2_hash(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    unsafe { zkmetal_sys::poseidon2_hash_cpu(a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) };
    out
}

/// Poseidon2 full permutation on a 3-element state (12 x u64).
pub fn poseidon2_permutation(state: &[u64; 12]) -> [u64; 12] {
    let mut result = [0u64; 12];
    unsafe { zkmetal_sys::poseidon2_permutation_cpu(state.as_ptr(), result.as_mut_ptr()) };
    result
}

/// Batch Poseidon2 hash of pairs. Input: pairs of Fr elements (8 u64 per pair).
/// Output: one Fr element (4 u64) per pair.
pub fn poseidon2_hash_batch(input: &[u64], count: usize) -> Vec<u64> {
    assert!(input.len() >= count * 8, "poseidon2 batch: input too small");
    let mut output = vec![0u64; count * 4];
    unsafe {
        zkmetal_sys::poseidon2_hash_batch_cpu(
            input.as_ptr(),
            count as core::ffi::c_int,
            output.as_mut_ptr(),
        );
    }
    output
}

/// Poseidon2 Merkle tree builder.
///
/// Builds a complete binary Merkle tree from `n` leaves (each 4 x u64 Fr).
/// Returns a buffer of (2n - 1) Fr elements:
///   tree[0..n] = leaves, tree[n..2n-1] = internal nodes, tree[2n-2] = root.
pub fn poseidon2_merkle_tree(leaves: &[u64], n: usize) -> Vec<u64> {
    assert!(n.is_power_of_two(), "merkle tree: n must be power of 2");
    assert!(leaves.len() >= n * 4, "merkle tree: leaves buffer too small");
    let mut tree = vec![0u64; (2 * n - 1) * 4];
    unsafe {
        zkmetal_sys::poseidon2_merkle_tree_cpu(
            leaves.as_ptr(),
            n as core::ffi::c_int,
            tree.as_mut_ptr(),
        );
    }
    tree
}

/// Keccak-256 hash of arbitrary input.
pub fn keccak256(input: &[u8]) -> [u8; 32] {
    let mut output = [0u8; 32];
    unsafe {
        zkmetal_sys::keccak256_hash_neon(input.as_ptr(), input.len(), output.as_mut_ptr());
    }
    output
}

/// Keccak-256 hash of two concatenated 32-byte inputs (optimized Merkle node).
pub fn keccak256_pair(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut output = [0u8; 32];
    unsafe {
        zkmetal_sys::keccak256_hash_pair_neon(a.as_ptr(), b.as_ptr(), output.as_mut_ptr());
    }
    output
}

/// Batch Keccak-256 of n pairs of 32-byte inputs.
pub fn keccak256_batch_pairs(inputs: &[u8], n: usize) -> Vec<u8> {
    assert!(inputs.len() >= n * 64, "keccak batch: input too small");
    let mut outputs = vec![0u8; n * 32];
    unsafe {
        zkmetal_sys::keccak256_batch_hash_pairs_neon(
            inputs.as_ptr(),
            outputs.as_mut_ptr(),
            n,
        );
    }
    outputs
}

/// Blake3 hash of arbitrary input (up to 64 bytes optimized path).
pub fn blake3(input: &[u8]) -> [u8; 32] {
    let mut output = [0u8; 32];
    unsafe {
        zkmetal_sys::blake3_hash_neon(input.as_ptr(), input.len(), output.as_mut_ptr());
    }
    output
}

/// Blake3 parent node hash: hash(left || right) for Merkle trees.
pub fn blake3_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut output = [0u8; 32];
    unsafe {
        zkmetal_sys::blake3_hash_pair_neon(left.as_ptr(), right.as_ptr(), output.as_mut_ptr());
    }
    output
}

/// Batch Blake3 parent hashing: n pairs -> n parent hashes.
pub fn blake3_batch_pairs(inputs: &[u8], n: usize) -> Vec<u8> {
    assert!(inputs.len() >= n * 64, "blake3 batch: input too small");
    let mut outputs = vec![0u8; n * 32];
    unsafe {
        zkmetal_sys::blake3_batch_hash_pairs_neon(
            inputs.as_ptr(),
            outputs.as_mut_ptr(),
            n,
        );
    }
    outputs
}
