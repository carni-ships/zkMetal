// GPU trace extension kernel for STARK prover
//
// Extends execution trace columns via coset NTT (Low-Degree Extension).
// Each thread handles one element of one column, enabling massively parallel
// extension of all columns simultaneously.
//
// The extension pipeline is:
//   1. INTT each column (coefficients)          -- handled by NTTEngine
//   2. Zero-pad + coset shift (this kernel)     -- fused GPU operation
//   3. Forward NTT of extended domain           -- handled by NTTEngine
//
// This kernel handles step 2 in batch: all columns in a single dispatch.
// Input layout: columns packed contiguously, each of length n_orig.
// Output layout: columns packed contiguously, each of length n_extended.

#include "../fields/bn254_fr.metal"
#include "../fields/babybear.metal"

// Batch zero-pad + coset shift for trace extension (BN254 Fr).
// Thread gid maps to (column, element) pair in the extended output.
// powers[i] = cosetShift^i, precomputed on CPU, length = n_extended.
kernel void trace_extend_batch_fr(
    device const Fr* input           [[buffer(0)]],   // numCols * n_orig elements
    device Fr* output                [[buffer(1)]],   // numCols * n_extended elements
    device const Fr* powers          [[buffer(2)]],   // coset powers, length n_extended
    constant uint& n_orig            [[buffer(3)]],
    constant uint& n_extended        [[buffer(4)]],
    constant uint& num_cols          [[buffer(5)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint total = n_extended * num_cols;
    if (gid >= total) return;

    uint col = gid / n_extended;
    uint idx = gid % n_extended;

    // Read coefficient (zero if beyond original length)
    Fr val = (idx < n_orig) ? input[col * n_orig + idx] : fr_zero();

    // Apply coset shift: coeff[i] *= g^i
    output[col * n_extended + idx] = fr_mul(val, powers[idx]);
}

// Batch zero-pad + coset shift for trace extension (BabyBear).
kernel void trace_extend_batch_bb(
    device const Bb* input           [[buffer(0)]],
    device Bb* output                [[buffer(1)]],
    device const Bb* powers          [[buffer(2)]],
    constant uint& n_orig            [[buffer(3)]],
    constant uint& n_extended        [[buffer(4)]],
    constant uint& num_cols          [[buffer(5)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint total = n_extended * num_cols;
    if (gid >= total) return;

    uint col = gid / n_extended;
    uint idx = gid % n_extended;

    Bb val = (idx < n_orig) ? input[col * n_orig + idx] : bb_zero();
    output[col * n_extended + idx] = bb_mul(val, powers[idx]);
}

// Row-to-column transpose kernel (Fr).
// Converts row-major trace (numRows x numCols) to column-major (numCols x numRows).
// Each thread copies one element.
kernel void trace_transpose_fr(
    device const Fr* row_major       [[buffer(0)]],  // [row * numCols + col]
    device Fr* col_major             [[buffer(1)]],  // [col * numRows + row]
    constant uint& num_rows          [[buffer(2)]],
    constant uint& num_cols          [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint total = num_rows * num_cols;
    if (gid >= total) return;

    uint row = gid / num_cols;
    uint col = gid % num_cols;

    col_major[col * num_rows + row] = row_major[row * num_cols + col];
}

// Row-to-column transpose kernel (BabyBear).
kernel void trace_transpose_bb(
    device const Bb* row_major       [[buffer(0)]],
    device Bb* col_major             [[buffer(1)]],
    constant uint& num_rows          [[buffer(2)]],
    constant uint& num_cols          [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    uint total = num_rows * num_cols;
    if (gid >= total) return;

    uint row = gid / num_cols;
    uint col = gid % num_cols;

    col_major[col * num_rows + row] = row_major[row * num_cols + col];
}
