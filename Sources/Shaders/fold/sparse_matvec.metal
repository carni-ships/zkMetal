// GPU-accelerated CSR Sparse Matrix-Vector Multiply for BN254 Fr
//
// Implements CSR (Compressed Sparse Row) sparse matvec on Metal GPU.
// Supports both single matrix-vector multiply and fused triple matvec
// when matrices share the same sparsity pattern.
//
// CSR Format:
//   rowPtr[i]  = start index of row i in colIdx[] and values[]
//   rowPtr[m]  = nnz (total non-zeros)
//   colIdx[k]  = column index of non-zero k
//   values[k]  = value of non-zero k
//
// Kernels:
//   sparse_matvec_bn254     — single matrix-vector multiply
//   sparse_matvec_triple_bn254 — fused A*z, B*z, C*z with shared sparsity
//   sparse_matvec_batch_bn254 — batch multiple matvecs for different z vectors
//
// Memory layout:
//   - Fr elements stored as 8x uint32 (Montgomery form, little-endian)
//   - rowPtr, colIdx stored as uint32
//   - Each row is processed by multiple threads cooperatively when row has many non-zeros

#include "../fields/bn254_fr.metal"

// ============================================================================
// Sparse MatVec — one row per threadgroup
// ============================================================================
//
// Each threadgroup handles one row. Threads cooperatively process the row's
// non-zero elements. For rows with few non-zeros, multiple rows can be
// handled by a single threadgroup.
//
// Threadgroup memory layout:
//   - Threadgroup stores partial dot products for the row
//   - Threadgroup barrier syncs after computing all products
//   - Thread 0 performs the final reduction

// Threadgroup size: multiple of 32 (SIMD group size)
// Recommended: 64-128 threads per threadgroup for good utilization
#define SPARSE_MATVEC_TG_SIZE 64

// Maximum non-zeros per row that we handle with threadgroup cooperation
// Rows with more non-zeros are processed with fewer threads per non-zero
#define SPARSE_MAX_ROW_NNZ 256

// ============================================================================
// Sparse MatVec — Single Matrix Multiply
// ============================================================================
//
// result[row] = sum_{k=rowPtr[row]}^{rowPtr[row+1]-1} values[k] * z[colIdx[k]]
//
// Each thread handles one row. This is simpler and works well for sparse matrices
// where each row typically has few non-zeros.

kernel void sparse_matvec_bn254(
    device const uint* rowPtr       [[buffer(0)]],  // m+1 uint32 row pointers
    device const uint* colIdx       [[buffer(1)]],  // nnz uint32 column indices
    device const Fr*   values       [[buffer(2)]],  // nnz Fr values
    device const Fr*   z           [[buffer(3)]],  // n Fr input vector
    device Fr*         result      [[buffer(4)]],  // m Fr output vector
    constant uint&     m           [[buffer(5)]],  // number of rows
    uint tgid                      [[threadgroup_position_in_grid]]
) {
    if (tgid >= m) return;

    uint rowStart = rowPtr[tgid];
    uint rowEnd = rowPtr[tgid + 1];

    Fr acc = fr_zero();
    for (uint nz = rowStart; nz < rowEnd; nz++) {
        uint col = colIdx[nz];
        acc = fr_add(acc, fr_mul(values[nz], z[col]));
    }

    result[tgid] = acc;
}

// ============================================================================
// Sparse MatVec Triple — Fused A*z, B*z, C*z with shared sparsity
// ============================================================================
//
// Computes three matrix-vector products simultaneously when all three matrices
// share the same rowPtr and colIdx (sparsity pattern).
//
// resultA[row] = sum valuesA[k] * z[colIdx[k]]
// resultB[row] = sum valuesB[k] * z[colIdx[k]]
// resultC[row] = sum valuesC[k] * z[colIdx[k]]
//
// This is ~3x faster than three separate matvecs because we:
//   1. Read the sparsity pattern once (rowPtr, colIdx)
//   2. Fetch z[col] once per non-zero, use for all three products
//   3. Compute three multiplications and three accumulations per non-zero

kernel void sparse_matvec_triple_bn254(
    device const uint* rowPtr       [[buffer(0)]],  // m+1 uint32 row pointers
    device const uint* colIdx       [[buffer(1)]],  // nnz uint32 column indices
    device const Fr*   valuesA      [[buffer(2)]],  // nnz Fr values for matrix A
    device const Fr*   valuesB      [[buffer(3)]],  // nnz Fr values for matrix B
    device const Fr*   valuesC      [[buffer(4)]],  // nnz Fr values for matrix C
    device const Fr*   z           [[buffer(5)]],  // n Fr input vector
    device Fr*         resultA      [[buffer(6)]],  // m Fr output vector A*z
    device Fr*         resultB      [[buffer(7)]],  // m Fr output vector B*z
    device Fr*         resultC      [[buffer(8)]],  // m Fr output vector C*z
    constant uint&     m           [[buffer(9)]],   // number of rows
    uint tgid                      [[threadgroup_position_in_grid]]
) {
    if (tgid >= m) return;

    uint rowStart = rowPtr[tgid];
    uint rowEnd = rowPtr[tgid + 1];

    Fr accA = fr_zero();
    Fr accB = fr_zero();
    Fr accC = fr_zero();

    for (uint nz = rowStart; nz < rowEnd; nz++) {
        uint col = colIdx[nz];
        Fr zval = z[col];
        accA = fr_add(accA, fr_mul(valuesA[nz], zval));
        accB = fr_add(accB, fr_mul(valuesB[nz], zval));
        accC = fr_add(accC, fr_mul(valuesC[nz], zval));
    }

    resultA[tgid] = accA;
    resultB[tgid] = accB;
    resultC[tgid] = accC;
}

// ============================================================================
// Sparse MatVec Batch — Multiple z vectors with same sparsity pattern
// ============================================================================
//
// Computes M*z1, M*z2, ..., M*zK for K different z vectors but the same
// sparse matrix M. This is useful when we need to compute A*z1, B*z1, C*z1
// and A*z2, B*z2, C*z2 separately.
//
// Layout:
//   - zVectors: concatenated z1, z2, ..., zK (each of length n)
//   - results: concatenated result1, result2, ..., resultK (each of length m)
//   - zOffsets[k] = start index of zk in zVectors
//   - resultOffsets[k] = start index of resultk in results

kernel void sparse_matvec_batch_bn254(
    device const uint* rowPtr           [[buffer(0)]],  // m+1 uint32 row pointers
    device const uint* colIdx           [[buffer(1)]],  // nnz uint32 column indices
    device const Fr*   values          [[buffer(2)]],  // nnz Fr values
    device const Fr*   zVectors         [[buffer(3)]],  // K * n Fr concatenated
    device Fr*         results          [[buffer(4)]],  // K * m Fr concatenated
    device const uint* zOffsets        [[buffer(5)]],  // K uint32 z vector offsets
    device const uint* resultOffsets    [[buffer(6)]],  // K uint32 result offsets
    constant uint&     m               [[buffer(7)]],  // number of rows
    constant uint&     n               [[buffer(8)]],  // vector dimension
    constant uint&     k               [[buffer(9)]],  // number of vectors to process
    uint tgid                          [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]]
) {
    if (tgid >= m * k) return;

    // Each threadgroup handles one row of one matvec
    uint matvecIdx = tgid / m;
    uint row = tgid % m;

    uint rowStart = rowPtr[row];
    uint rowEnd = rowPtr[row + 1];

    uint zBase = zOffsets[matvecIdx];
    uint resultBase = resultOffsets[matvecIdx];

    Fr acc = fr_zero();
    for (uint nz = rowStart; nz < rowEnd; nz++) {
        uint col = colIdx[nz];
        acc = fr_add(acc, fr_mul(values[nz], zVectors[zBase + col]));
    }

    results[resultBase + row] = acc;
}

// ============================================================================
// Sparse MatVec Triple Batch — A*z, B*z, C*z for multiple z vectors
// ============================================================================
//
// Computes (A*z1, B*z1, C*z1), (A*z2, B*z2, C*z2), ... for K different z vectors.
// All matrices share the same sparsity pattern.
//
// This is the most efficient kernel for Nova folding where we need:
//   az1 = A*z1, bz1 = B*z1, cz1 = C*z1
//   az2 = A*z2, bz2 = B*z2, cz2 = C*z2
//
// Layout:
//   - zVectors: concatenated z1, z2, ..., zK (each of length n)
//   - resultsA, resultsB, resultsC: concatenated results for each matvec

kernel void sparse_matvec_triple_batch_bn254(
    device const uint* rowPtr           [[buffer(0)]],  // m+1 uint32 row pointers
    device const uint* colIdx           [[buffer(1)]],  // nnz uint32 column indices
    device const Fr*   valuesA          [[buffer(2)]],  // nnz Fr values for A
    device const Fr*   valuesB          [[buffer(3)]],  // nnz Fr values for B
    device const Fr*   valuesC          [[buffer(4)]],  // nnz Fr values for C
    device const Fr*   zVectors         [[buffer(5)]],  // K * n Fr concatenated
    device Fr*         resultsA         [[buffer(6)]],  // K * m Fr output A*z
    device Fr*         resultsB         [[buffer(7)]],  // K * m Fr output B*z
    device Fr*         resultsC         [[buffer(8)]],  // K * m Fr output C*z
    device const uint* zOffsets         [[buffer(9)]],  // K uint32 z vector offsets
    device const uint* resultOffsets     [[buffer(10)]], // K uint32 result offsets
    constant uint&     m               [[buffer(11)]], // number of rows
    constant uint&     n               [[buffer(12)]], // vector dimension
    constant uint&     k               [[buffer(13)]], // number of z vectors
    uint tgid                          [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]]
) {
    if (tgid >= m * k) return;

    uint matvecIdx = tgid / m;
    uint row = tgid % m;

    uint rowStart = rowPtr[row];
    uint rowEnd = rowPtr[row + 1];

    uint zBase = zOffsets[matvecIdx];
    uint resultBase = resultOffsets[matvecIdx];

    Fr accA = fr_zero();
    Fr accB = fr_zero();
    Fr accC = fr_zero();

    for (uint nz = rowStart; nz < rowEnd; nz++) {
        uint col = colIdx[nz];
        Fr zval = zVectors[zBase + col];
        accA = fr_add(accA, fr_mul(valuesA[nz], zval));
        accB = fr_add(accB, fr_mul(valuesB[nz], zval));
        accC = fr_add(accC, fr_mul(valuesC[nz], zval));
    }

    resultsA[resultBase + row] = accA;
    resultsB[resultBase + row] = accB;
    resultsC[resultBase + row] = accC;
}
