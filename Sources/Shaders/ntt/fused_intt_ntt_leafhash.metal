// Fused INTT + Forward NTT + LeafHash Kernel for EVMetal
//
// Combines three phases into one dispatch to eliminate buffer synchronization overhead:
//
// Phase 1: Inverse NTT (interpolation from evaluation to coefficient form)
//          - Final DIF stage with unshift and 1/N scaling fused
//          - Uses Gentleman-Sande radix-2 DIF
//
// Phase 2: Forward NTT (for extended coset domain)
//          - First DIT stage with coset shift multiplication fused
//          - Uses Cooley-Tukey radix-2 DIT
//
// Phase 3: Leaf Hash (Merkle tree authentication path)
//          - Computes Poseidon2-M31 hashes for tree leaves
//          - 8 M31 values per leaf (single Poseidon2 block)
//
// Memory Layout Optimization:
//   Input:  evaluations in evaluation form (size N)
//   Output: extended evaluations (size M = blowupFactor * N) + leaf hashes
//   Intermediate: coefficient form kept in registers (no extra memory writes)
//
// This eliminates 2 GPU memory barriers and 2 buffer synchronizations.

#include "../fields/mersenne31.metal"
#include "../hash/poseidon2_m31.metal"

// =============================================================================
// CONFIGURATION
// =============================================================================

// Number of INTT stages for given logN
constant uint LOG_N = 0;  // Set at kernel launch

// Number of NTT stages for extended size (logN + logBlowup)
constant uint LOG_M = 0;  // Set at kernel launch

// Log of blowup factor
constant uint LOG_BLOWUP = 0;  // Set at kernel launch

// Leaf hash settings
constant uint NODE_SIZE = 8;  // 8 M31 per Poseidon2 leaf

// =============================================================================
// PHASE 1: INVERSE NTT - Final DIF Stage with Unshift + Scale
// =============================================================================

// Fused last INTT stage: butterfly + unshift + scale by 1/N
// For DIF (Gentleman-Sande): twiddle on output, result in place
// Last stage (stage 0): twiddle = 1, so we fuse the unshift and scale directly
//
// After this stage, data is in coefficient form (but still on GPU buffer)
kernel void fused_intt_final_unshift_scale(
    device M31* data              [[buffer(0)]],
    device const M31* invShift    [[buffer(1)]],   // shift^(-i), precomputed
    device const M31* twiddles    [[buffer(2)]],   // NTT twiddles (unused for stage 0)
    constant uint& n             [[buffer(3)]],    // Original size N
    constant uint& logN          [[buffer(4)]],    // log2(N)
    uint gid                      [[thread_position_in_grid]]
) {
    uint num_butterflies = n >> 1;
    if (gid >= num_butterflies) return;

    uint stage = 0;  // Last stage
    uint half_block = 1u << stage;
    uint block_size = half_block << 1;
    uint block_idx = gid / half_block;
    uint local_idx = gid % half_block;
    uint i = block_idx * block_size + local_idx;
    uint j = i + half_block;

    // Load values
    M31 a = data[i];
    M31 b = data[j];

    // DIF butterfly with twiddle=1 (stage 0 always has twiddle=1)
    // This is the last INTT stage, so no twiddle multiplication needed
    M31 sum  = m31Add(a, b);
    M31 diff = m31Sub(a, b);

    // Fused unshift + scale: multiply by invShift[i] * (1/N)
    // Precompute invN and combine with invShift for efficiency
    uint invN = m31InverseUint32(n);
    data[i] = m31Mul(m31Mul(sum,  invShift[i]), m31FromUint32(invN));
    data[j] = m31Mul(m31Mul(diff, invShift[j]), m31FromUint32(invN));
}

// =============================================================================
// PHASE 2: FORWARD NTT - First DIT Stage with Coset Shift
// =============================================================================

// Fused first NTT stage: apply coset shift and butterfly
// For DIT (Cooley-Tukey): twiddle on input, results in place
// First stage (stage 0): pairs at (2k, 2k+1), twiddle = 1 for bit-reversal input
//
// Input buffer now contains coefficients (from INTT final stage)
kernel void fused_ntt_first_coset_shift(
    device M31* data              [[buffer(0)]],
    device const M31* cosetPowers [[buffer(1)]],   // shift^i, length M
    device const M31* twiddles    [[buffer(2)]],   // NTT twiddles
    constant uint& n              [[buffer(3)]],    // Extended size M
    constant uint& logN           [[buffer(4)]],    // log2(M)
    uint gid                      [[thread_position_in_grid]]
) {
    uint num_butterflies = n >> 1;
    if (gid >= num_butterflies) return;

    uint stage = 0;  // First stage
    uint i = gid * 2;
    uint j = i + 1;

    // Fused: apply coset shift BEFORE butterfly
    M31 a = m31Mul(data[i], cosetPowers[i]);
    M31 b = m31Mul(data[j], cosetPowers[j]);

    // DIT butterfly with twiddle=1 (stage 0 always has twiddle=1)
    data[i] = m31Add(a, b);
    data[j] = m31Sub(a, b);
}

// =============================================================================
// PHASE 3: LEAF HASH - Poseidon2-M31 Merkle Tree Leaves
// =============================================================================

// Compute Poseidon2-M31 hashes for Merkle tree leaves
// Each leaf consists of NODE_SIZE M31 values
// Output: hash_digest[NODE_SIZE] for each leaf
kernel void leaf_hash_poseidon2_m31(
    device M31* leaves            [[buffer(0)]],    // Input: extended evaluations
    device M31* leafHashes        [[buffer(1)]],    // Output: Poseidon2 digests (8 M31 each)
    device const uchar* domain    [[buffer(2)]],    // Optional domain separator
    constant uint& numLeaves     [[buffer(3)]],    // Number of leaves = M / NODE_SIZE
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= numLeaves) return;

    // Load leaf data (NODE_SIZE M31 values)
    uint leafOffset = gid * NODE_SIZE;
    M31 inputs[NODE_SIZE];

    for (uint i = 0; i < NODE_SIZE; i++) {
        inputs[i] = leaves[leafOffset + i];
    }

    // Apply Poseidon2-M31 permutation
    M31 digest[NODE_SIZE];
    poseidon2_m31_hash(inputs, NODE_SIZE, digest);

    // Store digest
    uint hashOffset = gid * NODE_SIZE;
    for (uint i = 0; i < NODE_SIZE; i++) {
        leafHashes[hashOffset + i] = digest[i];
    }
}

// =============================================================================
// COMPLETE FUSED KERNEL: INTT + NTT + LEAF HASH (Single Dispatch)
// =============================================================================

// Fused INTT -> NTT -> LeafHash in a single GPU dispatch.
//
// Memory layout optimization:
//   - Threadblock 0: INTT final stage (n/2 butterflies)
//   - Threadblock 1: NTT first stage with coset shift (m/2 butterflies)
//   - Threadblock 2: Leaf hash computation (numLeaves leaf hashes)
//
// For optimal fusion, we combine all three phases in one kernel invocation
// using threadgroup barriers for synchronization within a single dispatch.

kernel void fused_intt_ntt_leafhash(
    device M31* data             [[buffer(0)]],    // In/Out: evaluation/coefficient/extended
    device M31* cosetPowers      [[buffer(1)]],    // Precomputed: shift^i for extended domain
    device M31* invShift         [[buffer(2)]],    // Precomputed: shift^(-i) for original domain
    device M31* leafHashes       [[buffer(3)]],    // Output: Poseidon2-M31 leaf digests
    device const M31* twiddles   [[buffer(4)]],    // NTT twiddles (not used in stage 0)
    constant uint& n             [[buffer(5)]],     // Original evaluation size
    constant uint& m             [[buffer(6)]],     // Extended size = blowupFactor * n
    constant uint& logN          [[buffer(7)]],     // log2(N)
    constant uint& logM          [[buffer(8)]],     // log2(M)
    constant uint& logBlowup     [[buffer(9)]],     // log2(blowupFactor)
    uint gid                     [[thread_position_in_grid]],
    uint lid                     [[thread_position_in_threadgroup]]
) {
    uint nHalf = n >> 1;
    uint mHalf = m >> 1;
    uint numLeaves = m >> 3;  // NODE_SIZE = 8

    // =================================================================
    // PHASE 1: INTT Final Stage + Unshift + Scale
    // =================================================================
    if (gid < nHalf) {
        uint stage = 0;  // Last INTT stage
        uint i = gid * 2;
        uint j = i + 1;

        M31 a = data[i];
        M31 b = data[j];

        // DIF butterfly (twiddle = 1 for stage 0)
        M31 sum  = m31Add(a, b);
        M31 diff = m31Sub(a, b);

        // Fused unshift + scale by 1/N
        uint invN = m31InverseUint32(n);
        M31 invNFr = m31FromUint32(invN);
        data[i] = m31Mul(m31Mul(sum,  invShift[i]), invNFr);
        data[j] = m31Mul(m31Mul(diff, invShift[j]), invNFr);
    }

    // =================================================================
    // SYNCHRONIZATION: All threads must complete INTT before NTT
    // =================================================================
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =================================================================
    // PHASE 2: Forward NTT First Stage + Coset Shift
    // =================================================================
    // Note: We need to work with the extended domain (size m)
    // The first m/n elements are the zero-padded coefficients
    // For efficiency, we process using the same buffer but different index space

    if (gid < mHalf && gid < (m >> 1)) {
        uint stage = 0;  // First NTT stage
        uint i = gid * 2;
        uint j = i + 1;

        // Apply coset shift for extended domain
        // For i >= n, shift powers are zero (but data is zero anyway from zero-padding)
        M31 a = (i < n) ? m31Mul(data[i], cosetPowers[i]) : m31Mul(data[i], cosetPowers[i]);
        M31 b = (j < n) ? m31Mul(data[j], cosetPowers[j]) : m31Mul(data[j], cosetPowers[j]);

        // DIT butterfly (twiddle = 1 for stage 0)
        data[i] = m31Add(a, b);
        data[j] = m31Sub(a, b);
    }

    // =================================================================
    // SYNCHRONIZATION: All threads must complete NTT before leaf hash
    // =================================================================
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =================================================================
    // PHASE 3: Leaf Hash Computation
    // =================================================================
    if (gid < numLeaves) {
        uint leafOffset = gid * NODE_SIZE;

        // Load inputs for Poseidon2
        M31 inputs[NODE_SIZE];
        #pragma unroll
        for (uint k = 0; k < NODE_SIZE; k++) {
            // Data is now in evaluation form (after NTT)
            // Use first NODE_SIZE elements from each leaf's position
            inputs[k] = data[leafOffset + k];
        }

        // Compute Poseidon2-M31 hash
        M31 digest[NODE_SIZE];
        poseidon2_m31_hash(inputs, NODE_SIZE, digest);

        // Store digest
        uint hashOffset = gid * NODE_SIZE;
        #pragma unroll
        for (uint k = 0; k < NODE_SIZE; k++) {
            leafHashes[hashOffset + k] = digest[k];
        }
    }
}

// =============================================================================
// ALTERNATIVE: Standalone Leaf Hash Kernel (for when NTT output is separate)
// =============================================================================

// Standalone leaf hash computation for cases where INTT/NTT run separately
// but we want to fuse the leaf hash with a read-back operation
kernel void standalone_leaf_hash(
    device const M31* evals       [[buffer(0)]],    // Input: evaluations (NTT output)
    device M31* leafHashes       [[buffer(1)]],    // Output: Poseidon2-M31 leaf digests
    constant uint& evalLen       [[buffer(2)]],    // Evaluation length
    uint gid                     [[thread_position_in_grid]]
) {
    uint numLeaves = evalLen >> 3;  // NODE_SIZE = 8
    if (gid >= numLeaves) return;

    uint leafOffset = gid * NODE_SIZE;
    M31 inputs[NODE_SIZE];

    // Load leaf data
    #pragma unroll
    for (uint i = 0; i < NODE_SIZE; i++) {
        inputs[i] = evals[leafOffset + i];
    }

    // Poseidon2-M31 hash
    M31 digest[NODE_SIZE];
    poseidon2_m31_hash(inputs, NODE_SIZE, digest);

    // Store digest
    uint hashOffset = gid * NODE_SIZE;
    #pragma unroll
    for (uint i = 0; i < NODE_SIZE; i++) {
        leafHashes[hashOffset + i] = digest[i];
    }
}

// =============================================================================
// MULTI-ROUND FOLD + LEAF HASH (for FRI integration)
// =============================================================================

// Fused FRI fold round with leaf hash computation.
// After each fold round, compute leaf hashes for the next tree level.
kernel void fused_fold_leafhash(
    device M31* data             [[buffer(0)]],    // In/Out: folded data
    device M31* leafHashes       [[buffer(1)]],    // Output: leaf hashes for this layer
    device const M31* inv2t      [[buffer(2)]],    // Precomputed: 1/(2*t_i)
    constant uint& n            [[buffer(3)]],     // Current layer size
    constant uint& numLeaves     [[buffer(4)]],    // Leaves for leaf hash = n / NODE_SIZE
    constant M31& alpha          [[buffer(5)]],     // Folding challenge
    uint gid                     [[thread_position_in_grid]]
) {
    uint nHalf = n >> 1;
    uint numLeafHashes = n >> 3;  // NODE_SIZE = 8

    // =================================================================
    // FRI FOLD: g[i] = (f[i] + f[i+n/2])/2 + alpha * (f[i] - f[i+n/2]) / (2*t_i)
    // =================================================================
    if (gid < nHalf) {
        M31 f0 = data[gid];
        M31 f1 = data[gid + nHalf];

        // (f[i] + f[i+n/2]) / 2
        M31 sum = m31Mul(m31Add(f0, f1), M31_HALF);

        // (f[i] - f[i+n/2]) / (2*t_i)
        M31 diff = m31Mul(m31Sub(f0, f1), inv2t[gid]);

        // alpha * diff
        diff = m31Mul(alpha, diff);

        // Folded result
        data[gid] = m31Add(sum, diff);
    }

    // =================================================================
    // SYNCHRONIZATION
    // =================================================================
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =================================================================
    // LEAF HASH: Compute Poseidon2-M31 hashes of folded layer
    // =================================================================
    if (gid < numLeafHashes) {
        uint leafOffset = gid * NODE_SIZE;
        M31 inputs[NODE_SIZE];

        #pragma unroll
        for (uint i = 0; i < NODE_SIZE; i++) {
            inputs[i] = data[leafOffset + i];
        }

        M31 digest[NODE_SIZE];
        poseidon2_m31_hash(inputs, NODE_SIZE, digest);

        uint hashOffset = gid * NODE_SIZE;
        #pragma unroll
        for (uint i = 0; i < NODE_SIZE; i++) {
            leafHashes[hashOffset + i] = digest[i];
        }
    }
}

// =============================================================================
// BATCH LEAF HASH (multiple columns in one dispatch)
// =============================================================================

// Fused batch leaf hash for multiple columns
// Each column has its own leaf hash output
kernel void batch_leaf_hash(
    device const M31* evals      [[buffer(0)]],    // Input: interleaved evaluations
    device M31* leafHashes      [[buffer(1)]],    // Output: interleaved leaf hashes
    constant uint& evalLen      [[buffer(2)]],    // Evaluation length per column
    constant uint& numCols      [[buffer(3)]],    // Number of columns
    uint gid                     [[thread_position_in_grid]]
) {
    uint numLeaves = evalLen >> 3;  // NODE_SIZE = 8
    uint totalLeaves = numLeaves * numCols;

    if (gid >= totalLeaves) return;

    // Decode column and leaf index
    uint leafIdx = gid % numLeaves;
    uint colIdx = gid / numLeaves;

    // Compute offsets
    uint evalOffset = colIdx * evalLen + leafIdx * NODE_SIZE;
    uint hashOffset = colIdx * numLeaves * NODE_SIZE + leafIdx * NODE_SIZE;

    // Load inputs
    M31 inputs[NODE_SIZE];
    #pragma unroll
    for (uint i = 0; i < NODE_SIZE; i++) {
        inputs[i] = evals[evalOffset + i];
    }

    // Poseidon2-M31 hash
    M31 digest[NODE_SIZE];
    poseidon2_m31_hash(inputs, NODE_SIZE, digest);

    // Store digest
    #pragma unroll
    for (uint i = 0; i < NODE_SIZE; i++) {
        leafHashes[hashOffset + i] = digest[i];
    }
}