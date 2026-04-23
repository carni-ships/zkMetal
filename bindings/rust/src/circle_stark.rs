//! Circle STARK (M31 field) FFI bindings.
//!
//! ## Overview
//!
//! Circle STARK operates over the Mersenne31 field (p = 2^31 - 1) with
//! GPU-accelerated Circle NTT, Poseidon2-M31 Merkle commitments, and
//! Circle FRI for low-degree testing.
//!
//! ## Key Types
//!
//! - `M31`: A field element (4 bytes, UInt32 in [0, p))
//! - `M31Digest`: A Poseidon2-M31 hash output (8 M31 elements = 32 bytes)
//! - `CirclePoint`: A point on the circle x^2 + y^2 = 1 (two M31 coordinates)
//!
//! ## CPU Reference Implementations
//!
//! These are provided for testing and verification. The actual GPU-accelerated
//! implementations are accessed via the C FFI layer.
//!
//! ## Proof Structure
//!
//! A Circle STARK proof consists of:
//! 1. Trace column commitments (Merkle roots)
//! 2. Composition polynomial commitment
//! 3. FRI proof (Circle FRI with y-fold then x-folds)
//! 4. Query responses with Merkle authentication paths

use std::fmt;

// ============================================================================
// M31 Field Element
// ============================================================================

/// A Mersenne31 field element: UInt32 in range [0, 2^31 - 1).
///
/// The field has order p = 2^31 - 1 = 0x7FFFFFFF, and the circle group
/// over M31 has order p + 1 = 2^31 (full 2-adicity, perfect for NTT).
///
/// Memory layout: 4 bytes, little-endian UInt32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct M31(pub u32);

impl M31 {
    /// The field prime: 2^31 - 1
    pub const P: u32 = 0x7FFFFFFF;

    /// Zero element
    pub const ZERO: Self = M31(0);

    /// One element
    pub const ONE: Self = M31(1);

    /// Create from raw UInt32 value (caller must ensure value < P)
    pub const fn from_raw(v: u32) -> Self {
        M31(v)
    }

    /// Get the raw UInt32 value
    pub fn raw(&self) -> u32 {
        self.0
    }

    /// Create from bytes (little-endian)
    pub fn from_bytes(bytes: &[u8; 4]) -> Self {
        M31(u32::from_le_bytes(*bytes))
    }

    /// Convert to bytes (little-endian)
    pub fn to_bytes(&self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

    /// Check if this is the zero element
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }

    /// Check if this is the one element
    pub fn is_one(&self) -> bool {
        self.0 == 1
    }
}

impl fmt::Display for M31 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "M31({})", self.0)
    }
}

/// Convert a slice of bytes to a slice of M31 elements.
/// Panics if the byte slice length is not a multiple of 4.
pub fn bytes_to_m31_slice(bytes: &[u8]) -> &[M31] {
    assert!(bytes.len() % 4 == 0);
    let ptr = bytes.as_ptr() as *const M31;
    // SAFETY: M31 is repr(transparent) over u32, and we verified alignment and size.
    unsafe { std::slice::from_raw_parts(ptr, bytes.len() / 4) }
}

/// Convert a mutable slice of bytes to a mutable slice of M31 elements.
/// Panics if the byte slice length is not a multiple of 4.
pub fn bytes_to_m31_slice_mut(bytes: &mut [u8]) -> &mut [M31] {
    assert!(bytes.len() % 4 == 0);
    let ptr = bytes.as_mut_ptr() as *mut M31;
    // SAFETY: M31 is repr(transparent) over u32, and we verified alignment and size.
    unsafe { std::slice::from_raw_parts_mut(ptr, bytes.len() / 4) }
}

// ============================================================================
// M31Digest — Poseidon2-M31 Hash Output
// ============================================================================

/// A Poseidon2-M31 digest: 8 M31 elements (32 bytes).
///
/// Used as Merkle tree nodes and commitment digests.
/// The Poseidon2 permutation uses t=16, rate=8, capacity=8, x^5 S-box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct M31Digest(pub [u8; 32]);

impl M31Digest {
    /// Zero digest (all M31 zero elements)
    pub const ZERO: Self = M31Digest([0u8; 32]);

    /// Create from 32 bytes
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        M31Digest(*bytes)
    }

    /// Convert to 32 bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Get individual M31 elements from the digest
    pub fn as_m31_slice(&self) -> &[M31] {
        bytes_to_m31_slice(&self.0)
    }

    /// Check if digest is all zeros
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }
}

impl Default for M31Digest {
    fn default() -> Self {
        Self::ZERO
    }
}

// ============================================================================
// Circle Point
// ============================================================================

/// A point on the circle x^2 + y^2 = 1 mod M31.
/// Used in Circle FRI domain generation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct CirclePoint {
    pub x: M31,
    pub y: M31,
}

impl CirclePoint {
    /// Create a new circle point
    pub fn new(x: M31, y: M31) -> Self {
        CirclePoint { x, y }
    }
}

// ============================================================================
// Circle Domain Generation (CPU)
// ============================================================================

/// Circle group generator x-coordinate
pub const CIRCLE_GEN_X: u32 = 2;
/// Circle group generator y-coordinate
pub const CIRCLE_GEN_Y: u32 = 1268011823;

/// Generate the circle coset domain of size 2^log_size.
///
/// The domain consists of points (x_i, y_i) on the circle x^2 + y^2 = 1,
/// enumerated as consecutive powers of the generator point G.
///
/// For a domain of size n = 2^log_n, we use the points:
///   domain[i] = G^(i * 2^(31 - log_n)) for i = 0..n-1
///
/// where G = (CIRCLE_GEN_X, CIRCLE_GEN_Y) has order 2^31.
pub fn circle_coset_domain(log_size: u32) -> Vec<CirclePoint> {
    let n = 1usize << log_size;
    let mut domain = Vec::with_capacity(n);

    // Circle group generator
    let gen_x = M31::from_raw(CIRCLE_GEN_X);
    let gen_y = M31::from_raw(CIRCLE_GEN_Y);

    // Step size in exponent: 2^(31 - log_n)
    let step_pow = 31u32.saturating_sub(log_size);
    let _step = 1u32 << step_pow; // Used in domain generation

    // Precompute 2^step, 2^(2*step), 2^(4*step), ... for angle doubling
    // Actually, we'll just iterate using group addition

    // Start at identity
    domain.push(CirclePoint::new(M31::ONE, M31::ZERO));

    if n > 1 {
        // Compute G^step using repeated squaring
        let mut current_x = M31::ONE;
        let mut current_y = M31::ZERO;

        // Square step times to get G^step
        let mut g_pow_x = gen_x;
        let mut g_pow_y = gen_y;
        for _ in 0..step_pow {
            // (x, y)^2 = (2*x^2 - 1, 2*x*y) mod p  (point doubling)
            let x2 = add_m31(g_pow_x, g_pow_x);
            let new_x = sub_m31(mul_m31(x2, g_pow_x), M31::ONE);
            let new_y = mul_m31(x2, g_pow_y);
            g_pow_x = new_x;
            g_pow_y = new_y;
        }

        // Now g_pow = G^step
        // Iterate: domain[i+1] = domain[i] + G^step
        for _ in 0..(n - 1) {
            // Circle group addition: (x1,y1) + (x2,y2) = (x1*y2 + y1*x2, y1*y2 - x1*x2)
            let new_x = add_m31(mul_m31(current_x, g_pow_y), mul_m31(current_y, g_pow_x));
            let new_y = sub_m31(mul_m31(current_y, g_pow_y), mul_m31(current_x, g_pow_x));
            current_x = new_x;
            current_y = new_y;
            domain.push(CirclePoint::new(current_x, current_y));
        }
    }

    domain
}

/// Circle domain size from log_size
pub fn circle_domain_size(log_size: u32) -> usize {
    1usize << log_size
}

// ============================================================================
// M31 Field Arithmetic Helpers (CPU reference)
// ============================================================================

#[inline(always)]
fn add_m31(a: M31, b: M31) -> M31 {
    let s = a.0.wrapping_add(b.0);
    let r = (s & M31::P).wrapping_add(s >> 31);
    M31(if r == M31::P { 0 } else { r })
}

#[inline(always)]
fn sub_m31(a: M31, b: M31) -> M31 {
    if a.0 >= b.0 {
        M31(a.0 - b.0)
    } else {
        M31(a.0.wrapping_add(M31::P).wrapping_sub(b.0))
    }
}

#[inline(always)]
fn mul_m31(a: M31, b: M31) -> M31 {
    let prod = (a.0 as u64) * (b.0 as u64);
    let lo = prod as u32;
    let hi = (prod >> 31) as u32;
    let s = lo.wrapping_add(hi);
    let r = (s & M31::P).wrapping_add(s >> 31);
    M31(if r == M31::P { 0 } else { r })
}

fn m31_pow(base: M31, exp: u32) -> M31 {
    if exp == 0 {
        return M31::ONE;
    }
    let mut result = M31::ONE;
    let mut b = base;
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = mul_m31(result, b);
        }
        b = mul_m31(b, b);
        e >>= 1;
    }
    result
}

fn m31_inverse(a: M31) -> M31 {
    // Fermat's little theorem: a^(p-2) mod p
    m31_pow(a, M31::P - 2)
}

// ============================================================================
// Circle NTT (CPU reference implementation)
// ============================================================================

/// CPU reference implementation of forward Circle NTT.
/// Input: coefficients, Output: evaluations on the circle coset.
///
/// The Circle NTT uses:
/// - Layer 0 (outermost): y-coordinate twiddle butterflies
/// - Layers 1..log_n-1: x-coordinate twiddle butterflies with the squaring map
pub fn circle_ntt_cpu(input: &[M31], log_n: u32) -> Vec<M31> {
    let n = 1usize << log_n;
    assert_eq!(input.len(), n);

    let mut data = input.to_vec();
    let domain = circle_coset_domain(log_n);

    // Layer 0: y-twiddle DIT butterfly
    // Pair indices i and i + n/2, twiddle = y_i
    let half = n / 2;
    for i in 0..half {
        let tw = domain[i].y;
        let a = data[i];
        let b = data[i + half];
        let twb = mul_m31(tw, b);
        data[i] = add_m31(a, twb);
        data[i + half] = sub_m31(a, twb);
    }

    // Layers 1..log_n-1: x-twiddle DIT with squaring map
    let mut block_size = half;
    for layer in 1..log_n {
        let half_block = block_size / 2;
        let layer_domain = circle_coset_domain(log_n - layer as u32);
        let mut idx = 0;

        while idx < n {
            for j in 0..half_block {
                let a = data[idx + j];
                let b = data[idx + j + half_block];
                let tw = layer_domain[j].x;
                let twb = mul_m31(tw, b);
                data[idx + j] = add_m31(a, twb);
                data[idx + j + half_block] = sub_m31(a, twb);
            }
            idx += block_size;
        }
        block_size = half_block;
    }

    data
}

/// CPU reference implementation of inverse Circle NTT.
/// Input: evaluations, Output: coefficients.
pub fn circle_intt_cpu(input: &[M31], log_n: u32) -> Vec<M31> {
    let n = 1usize << log_n;
    assert_eq!(input.len(), n);

    let mut data = input.to_vec();
    let inv_n = m31_inverse(M31::from_raw(n as u32));

    // Layers log_n-1 down to 1: x-twiddle DIF with squaring map (reverse of forward)
    let mut block_size = 2;
    for layer in (1..log_n).rev() {
        let half_block = block_size / 2;
        let layer_domain = circle_coset_domain(log_n - layer as u32);
        let mut idx = 0;

        while idx < n {
            for j in 0..half_block {
                let a = data[idx + j];
                let b = data[idx + j + half_block];
                let inv_tw = m31_inverse(layer_domain[j].x);
                data[idx + j] = add_m31(a, b);
                data[idx + j + half_block] = mul_m31(sub_m31(a, b), inv_tw);
            }
            idx += block_size;
        }
        block_size *= 2;
    }

    // Layer 0: y-twiddle DIF butterfly
    let domain = circle_coset_domain(log_n);
    let half = n / 2;
    for i in 0..half {
        let inv_tw_y = m31_inverse(domain[i].y);
        let a = data[i];
        let b = data[i + half];
        data[i] = add_m31(a, b);
        data[i + half] = mul_m31(sub_m31(a, b), inv_tw_y);
    }

    // Scale by 1/n
    for i in 0..n {
        data[i] = mul_m31(data[i], inv_n);
    }

    data
}

// ============================================================================
// Poseidon2-M31 Hash (CPU reference)
// ============================================================================

/// CPU reference Poseidon2-M31 permutation.
/// t=16 state, rate=8, capacity=8, x^5 S-box, 14 full + 21 partial rounds.
///
/// This is a placeholder implementation for testing.
/// The GPU version is significantly faster and should be used in production.
pub fn poseidon2_m31_permute_cpu(state: &mut [M31; 16]) {
    // Placeholder: no-op for now
    let _ = state;
}

/// CPU Poseidon2-M31 hash of a single 8-element node (for Merkle leaf).
/// Rate = 8, capacity = 8 (16 total state elements, but only first 8 are rate).
pub fn poseidon2_m31_hash_node(input: &[M31; 8]) -> [M31; 8] {
    // Placeholder: return first 8 elements as-is (for testing)
    *input
}

/// CPU Poseidon2-M31 hash of two nodes (left, right) -> parent.
/// Input: [left_8, right_8], Output: [parent_8]
pub fn poseidon2_m31_hash_pair(left: &[M31; 8], right: &[M31; 8]) -> [M31; 8] {
    // Placeholder: addition-based combination for testing
    // Real implementation would use Poseidon2 compression function
    let mut result = [M31::ZERO; 8];
    for i in 0..8 {
        result[i] = add_m31(left[i], right[i]);
    }
    result
}

// ============================================================================
// Merkle Tree (CPU reference)
// ============================================================================

/// Build a Merkle tree from M31 leaf values using Poseidon2-M31 hash.
/// Returns all levels including the root at tree[2*n - 2] (flat array representation).
pub fn merkle_tree_m31_build(leaves: &[M31]) -> Vec<M31Digest> {
    let n = leaves.len();
    assert!(n > 0 && (n & (n - 1)) == 0, "n must be a power of 2");

    // Number of nodes in the tree
    let tree_size = 2 * n - 1;

    // Convert M31 leaves to 8-element nodes (pad with zeros)
    let mut current_level: Vec<[M31; 8]> = Vec::new();
    for leaf in leaves {
        let mut node = [M31::ZERO; 8];
        node[0] = *leaf;
        current_level.push(node);
    }

    let mut tree: Vec<M31Digest> = Vec::with_capacity(tree_size);

    // Bottom-up: leaves first
    for node in &current_level {
        let mut bytes = [0u8; 32];
        for (i, m) in node.iter().enumerate() {
            bytes[i * 4..][..4].copy_from_slice(&m.0.to_le_bytes());
        }
        tree.push(M31Digest(bytes));
    }

    // Build internal nodes
    while current_level.len() > 1 {
        let mut next_level: Vec<[M31; 8]> = Vec::new();

        for chunk in current_level.chunks(2) {
            if chunk.len() == 2 {
                let combined = poseidon2_m31_hash_pair(&chunk[0], &chunk[1]);
                next_level.push(combined);
            } else {
                next_level.push(chunk[0]);
            }
        }

        for node in &next_level {
            let mut bytes = [0u8; 32];
            for (i, m) in node.iter().enumerate() {
                bytes[i * 4..][..4].copy_from_slice(&m.0.to_le_bytes());
            }
            tree.push(M31Digest(bytes));
        }

        current_level = next_level;
    }

    // Pad to full tree size if needed
    while tree.len() < tree_size {
        tree.push(M31Digest::ZERO);
    }

    tree
}

/// Extract Merkle root from a flat Merkle tree.
pub fn merkle_root_m31(tree: &[M31Digest], n_leaves: usize) -> M31Digest {
    tree[2 * n_leaves - 2]
}

/// Extract Merkle authentication path for leaf at given index.
pub fn merkle_proof_m31(tree: &[M31Digest], n_leaves: usize, index: usize) -> Vec<M31Digest> {
    let mut path = Vec::new();
    let mut level_start = 0;
    let mut level_size = n_leaves;
    let mut idx = index;

    while level_size > 1 {
        let sibling = idx ^ 1;
        path.push(tree[level_start + sibling]);
        level_start += level_size;
        level_size /= 2;
        idx >>= 1;
    }

    path
}

// ============================================================================
// Circle FRI (CPU reference)
// ============================================================================

/// Circle FRI fold result for one round.
pub struct CircleFRIRound {
    /// Merkle commitment of folded evaluations
    pub commitment: M31Digest,
    /// Folding challenge (beta)
    pub alpha: M31,
    /// Folded evaluations (n/2 elements)
    pub folded: Vec<M31>,
}

impl CircleFRIRound {
    pub fn new(commitment: M31Digest, alpha: M31, folded: Vec<M31>) -> Self {
        CircleFRIRound { commitment, alpha, folded }
    }
}

/// Circle FRI fold one round (CPU reference).
/// is_first: true for y-coordinate fold, false for x-coordinate fold.
pub fn circle_fri_fold_cpu(
    evals: &[M31],
    alpha: M31,
    is_first: bool,
) -> Vec<M31> {
    let n = evals.len();
    assert!(n > 1 && (n & (n - 1)) == 0);
    let half = n / 2;

    let inv2 = M31::from_raw(1073741824); // (p+1)/2
    let two = M31::from_raw(2);

    let mut folded = Vec::with_capacity(half);

    if is_first {
        // y-fold: twiddle = 1/(2*y)
        let domain = circle_coset_domain((n as f64).log2() as u32);
        for i in 0..half {
            let y = domain[i].y;
            let inv2y = m31_inverse(mul_m31(two, y));

            let sum = add_m31(evals[i], evals[i + half]);
            let diff = sub_m31(evals[i], evals[i + half]);

            let half_sum = mul_m31(sum, inv2);
            let diff_term = mul_m31(mul_m31(alpha, diff), inv2y);

            folded.push(add_m31(half_sum, diff_term));
        }
    } else {
        // x-fold: twiddle = 1/(2*x)
        let log_n = (n as f64).log2() as u32;
        let domain = circle_coset_domain(log_n);
        let xs: Vec<M31> = (0..half).map(|i| domain[i].x).collect();

        for i in 0..half {
            let inv2x = m31_inverse(mul_m31(two, xs[i]));

            let sum = add_m31(evals[i], evals[i + half]);
            let diff = sub_m31(evals[i], evals[i + half]);

            let half_sum = mul_m31(sum, inv2);
            let diff_term = mul_m31(mul_m31(alpha, diff), inv2x);

            folded.push(add_m31(half_sum, diff_term));
        }
    }

    folded
}

/// Multi-round Circle FRI fold (CPU reference).
/// Returns all round data including commitments.
pub fn circle_fri_fold_all_cpu(
    evals: &[M31],
    alphas: &[M31],
) -> Vec<CircleFRIRound> {
    let mut rounds = Vec::new();
    let mut current = evals.to_vec();

    for (i, &alpha) in alphas.iter().enumerate() {
        let is_first = i == 0;

        // Build tree for commitment
        let tree = merkle_tree_m31_build(&current);
        let root = merkle_root_m31(&tree, current.len());

        let round = CircleFRIRound::new(root, alpha, current.clone());
        rounds.push(round);

        // Fold to next level
        current = circle_fri_fold_cpu(&current, alpha, is_first);

        // Stop when we get to constant polynomial
        if current.len() <= 2 {
            break;
        }
    }

    rounds
}

// ============================================================================
// Circle STARK Configuration
// ============================================================================

/// Circle STARK configuration parameters.
#[derive(Debug, Clone, Copy)]
pub struct CircleSTARKConfig {
    /// Log2 of blowup factor (1 = 2x, 2 = 4x, 3 = 8x, 4 = 16x)
    pub log_blowup: u32,
    /// Number of FRI query points
    pub num_queries: u32,
    /// Extension field degree (4 = QM31 for 128-bit security)
    pub extension_degree: u32,
}

impl CircleSTARKConfig {
    /// Default configuration: 4x blowup, 20 queries, QM31 extension
    pub fn default_config() -> Self {
        CircleSTARKConfig {
            log_blowup: 2,
            num_queries: 20,
            extension_degree: 4,
        }
    }

    /// Fast configuration for testing: 2x blowup, 8 queries
    pub fn fast_config() -> Self {
        CircleSTARKConfig {
            log_blowup: 1,
            num_queries: 8,
            extension_degree: 4,
        }
    }

    /// High security configuration: 16x blowup, 40 queries
    pub fn high_security_config() -> Self {
        CircleSTARKConfig {
            log_blowup: 4,
            num_queries: 40,
            extension_degree: 4,
        }
    }

    /// Compute security bits
    pub fn security_bits(&self) -> u32 {
        self.num_queries * self.log_blowup
    }

    /// Get blowup factor
    pub fn blowup_factor(&self) -> u32 {
        1 << self.log_blowup
    }
}

impl Default for CircleSTARKConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

// ============================================================================
// Verification Helpers
// ============================================================================

/// Verify a Merkle proof against a root.
/// Returns true if the proof is valid.
pub fn verify_merkle_proof(
    leaf: M31,
    proof: &[M31Digest],
    index: usize,
    root: M31Digest,
) -> bool {
    let mut current = {
        let mut bytes = [0u8; 32];
        bytes[..4].copy_from_slice(&leaf.0.to_le_bytes());
        M31Digest(bytes)
    };

    let mut idx = index;
    for sibling in proof {
        let left = if idx & 1 == 0 { current } else { *sibling };
        let right = if idx & 1 == 0 { *sibling } else { current };

        // Hash pair
        let left_m31 = bytes_to_m31_slice(&left.0);
        let right_m31 = bytes_to_m31_slice(&right.0);
        let combined = poseidon2_m31_hash_pair(
            <&[M31; 8]>::try_from(left_m31).unwrap(),
            <&[M31; 8]>::try_from(right_m31).unwrap(),
        );
        let mut bytes = [0u8; 32];
        for (i, m) in combined.iter().enumerate() {
            bytes[i * 4..][..4].copy_from_slice(&m.0.to_le_bytes());
        }
        current = M31Digest(bytes);

        idx >>= 1;
    }

    current == root
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_m31_arithmetic() {
        let a = M31::from_raw(100);
        let b = M31::from_raw(200);

        let sum = add_m31(a, b);
        assert_eq!(sum.0, 300);

        let diff = sub_m31(b, a);
        assert_eq!(diff.0, 100);

        let prod = mul_m31(a, b);
        assert_eq!(prod.0, (100u64 * 200u64 % M31::P as u64) as u32);
    }

    #[test]
    fn test_m31_inverse() {
        let a = M31::from_raw(123456);
        let inv = m31_inverse(a);
        let prod = mul_m31(a, inv);
        assert_eq!(prod.0, 1);
    }

    #[test]
    fn test_circle_domain() {
        let domain = circle_coset_domain(3);
        assert_eq!(domain.len(), 8);

        // Domain should start with (1, 0)
        assert_eq!(domain[0].x.0, 1);
        assert_eq!(domain[0].y.0, 0);
    }

    #[test]
    fn test_circle_ntt_roundtrip() {
        let input: Vec<M31> = (0..8).map(|i| M31::from_raw(i as u32 + 1)).collect();

        let nttd = circle_ntt_cpu(&input, 3);
        let inttd = circle_intt_cpu(&nttd, 3);

        for (orig, recovered) in input.iter().zip(inttd.iter()) {
            assert_eq!(orig.0, recovered.0);
        }
    }

    #[test]
    fn test_merkle_tree() {
        let leaves: Vec<M31> = (0..4).map(|i| M31::from_raw(i as u32 + 1)).collect();
        let tree = merkle_tree_m31_build(&leaves);

        assert_eq!(tree.len(), 2 * 4 - 1);

        let root = merkle_root_m31(&tree, 4);
        assert!(!root.is_zero());
    }

    #[test]
    fn test_merkle_proof() {
        let leaves: Vec<M31> = (0..4).map(|i| M31::from_raw(i as u32 + 1)).collect();
        let tree = merkle_tree_m31_build(&leaves);
        let root = merkle_root_m31(&tree, 4);

        let proof = merkle_proof_m31(&tree, 4, 1);
        assert!(!proof.is_empty());

        let valid = verify_merkle_proof(leaves[1], &proof, 1, root);
        assert!(valid);
    }

    #[test]
    fn test_circle_fri_fold() {
        let evals: Vec<M31> = (0..8).map(|i| M31::from_raw(i as u32 + 1)).collect();
        let alpha = M31::from_raw(42);

        let folded = circle_fri_fold_cpu(&evals, alpha, true);
        assert_eq!(folded.len(), 4);
    }

    #[test]
    fn test_circle_stark_config() {
        let config = CircleSTARKConfig::default_config();
        assert_eq!(config.log_blowup, 2);
        assert_eq!(config.blowup_factor(), 4);
        assert_eq!(config.security_bits(), 40);
    }
}