// ane_tensor.metal — ANE-accelerated Tensor Operations
//
// GPU/ANE-accelerated matrix-vector multiply and matrix-matrix multiply.
// Uses SIMD parallelism and can offload to ANE on Apple Silicon.
//
// Key operations:
// - Matrix-vector multiply: result = M * vec
// - Inner product: sum = Σ a[i] * b[i] (expressed as matvec)
// - Matrix-matrix multiply: C = A * B
//
// Uses uint4 SIMD to process 4 field elements in parallel per thread.

#include <metal_stdlib>
using namespace metal;

// ============================================================
// BabyBear field: p = 2^31 - 2^27 + 1 = 0x78000001
// ============================================================

constant uint BB_P = 0x78000001u;
constant uint BB_P_INV = 2281701377u;

// BabyBear modular multiplication via Barrett reduction
inline uint bb_mul(uint a, uint b) {
    ulong prod = (ulong)a * (ulong)b;
    uint prod_lo = (uint)prod;
    uint prod_hi = (uint)(prod >> 32);
    ulong t1 = (ulong)prod_lo * (ulong)BB_P_INV;
    ulong t2 = (ulong)prod_hi * (ulong)BB_P_INV;
    uint q = (uint)((t2 + (t1 >> 32)) >> 30);
    uint r = (uint)(prod - (ulong)q * BB_P);
    return (r >= BB_P) ? r - BB_P : r;
}

inline uint bb_add(uint a, uint b) {
    uint s = a + b;
    return (s >= BB_P) ? s - BB_P : s;
}

inline uint bb_sub(uint a, uint b) {
    return (a >= b) ? a - b : a + BB_P - b;
}

// ============================================================
// SIMD4 arithmetic (4 elements per vector)
// ============================================================

// SIMD4 Barrett reduction
inline uint4 bb_mul_v4(uint4 a, uint4 b) {
    ulong4 prod = (ulong4)a * (ulong4)b;
    uint4 prod_lo = (uint4)(prod & 0xFFFFFFFFu);
    uint4 prod_hi = (uint4)(prod >> 32);
    ulong4 t1 = (ulong4)prod_lo * (ulong4)BB_P_INV;
    ulong4 t2 = (ulong4)prod_hi * (ulong4)BB_P_INV;
    uint4 q = (uint4)((t2 + (t1 >> 32)) >> 30);
    uint4 r = (uint4)(prod - (ulong4)q * (ulong4)BB_P);
    return select(r - BB_P, r, r < BB_P);
}

inline uint4 bb_add_v4(uint4 a, uint4 b) {
    uint4 s = a + b;
    return select(s - BB_P, s, s < BB_P);
}

// ============================================================
// Matrix-Vector Multiply: result = M * vec
//
// Each thread computes one row of the result
// Threads are grouped for SIMD4 parallelism
// ============================================================

// Single row matvec: result[i] = sum_j M[i*cols+j] * vec[j]
kernel void tensor_matvec_row(
    device const uint *M [[buffer(0)]],
    device const uint *vec [[buffer(1)]],
    device uint *result [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;

    uint4 acc = 0;
    uint base = gid * cols;

    // Process 4 elements at a time with SIMD
    uint j = 0;
    for (; j + 4 <= cols; j += 4) {
        uint4 m_vals = *(device uint4*)(M + base + j);
        uint4 v_vals = *(device uint4*)(vec + j);
        acc += bb_mul_v4(m_vals, v_vals);
    }

    // Handle remainder
    for (; j < cols; j++) {
        acc.x = bb_add(acc.x, bb_mul(M[base + j], vec[j]));
    }

    // Horizontal sum of SIMD4 accumulator
    uint sum = acc.x + acc.y + acc.z + acc.w;
    result[gid] = sum;
}

// Batch matvec: result[i] = M_i * vec_i for i in [0, batch)
kernel void tensor_matvec_batch(
    device const uint *M [[buffer(0)]],
    device const uint *vecs [[buffer(1)]],
    device uint *result [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    constant uint &batch [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint matSize = rows * cols;
    uint vecSize = cols;

    for (uint b = 0; b < batch; b++) {
        uint matBase = b * matSize;
        uint vecBase = b * vecSize;
        uint resBase = b * rows;

        if (gid < rows) {
            uint4 acc = 0;
            uint base = matBase + gid * cols;

            uint j = 0;
            for (; j + 4 <= cols; j += 4) {
                uint4 m_vals = *(device uint4*)(M + base + j);
                uint4 v_vals = *(device uint4*)(vecs + vecBase + j);
                acc += bb_mul_v4(m_vals, v_vals);
            }
            for (; j < cols; j++) {
                acc.x = bb_add(acc.x, bb_mul(M[base + j], vecs[vecBase + j]));
            }

            uint sum = acc.x + acc.y + acc.z + acc.w;
            result[resBase + gid] = sum;
        }
    }
}

// ============================================================
// Inner Product: sum = Σ a[i] * b[i]
//
// Expressed as matrix-vector: one element result from dot product
// ============================================================

// Inner product of two vectors (single result)
kernel void tensor_inner_product(
    device const uint *a [[buffer(0)]],
    device const uint *b [[buffer(1)]],
    device uint *result [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= 1) return;

    uint4 acc = 0;
    uint i = 0;

    // Process 4 elements at a time
    for (; i + 4 <= n; i += 4) {
        uint4 a_vals = *(device uint4*)(a + i);
        uint4 b_vals = *(device uint4*)(b + i);
        acc += bb_mul_v4(a_vals, b_vals);
    }

    // Handle remainder
    for (; i < n; i++) {
        acc.x = bb_add(acc.x, bb_mul(a[i], b[i]));
    }

    // Horizontal sum
    uint sum = acc.x + acc.y + acc.z + acc.w;
    result[0] = sum;
}

// Batch inner products: result[k] = Σ a_batch[k*n+i] * b_batch[k*n+i]
kernel void tensor_inner_product_batch(
    device const uint *a_batch [[buffer(0)]],
    device const uint *b_batch [[buffer(1)]],
    device uint *result [[buffer(2)]],
    constant uint &n [[buffer(3)]],
    constant uint &batch [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batch) return;

    uint4 acc = 0;
    uint base = gid * n;
    uint i = 0;

    for (; i + 4 <= n; i += 4) {
        uint4 a_vals = *(device uint4*)(a_batch + base + i);
        uint4 b_vals = *(device uint4*)(b_batch + base + i);
        acc += bb_mul_v4(a_vals, b_vals);
    }
    for (; i < n; i++) {
        acc.x = bb_add(acc.x, bb_mul(a_batch[base + i], b_batch[base + i]));
    }

    uint sum = acc.x + acc.y + acc.z + acc.w;
    result[gid] = sum;
}

// ============================================================
// Matrix-Matrix Multiply: C = A * B
//
// Each thread computes one element of C: C[i,j] = Σ_k A[i,k] * B[k,j]
// For large matrices, this parallelizes well across rows
// ============================================================

kernel void tensor_matmul(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *C [[buffer(2)]],
    constant uint &rowsA [[buffer(3)]],
    constant uint &colsA [[buffer(4)]],
    constant uint &colsB [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint rows = rowsA;
    uint cols = colsB;
    uint inner = colsA;

    uint row = gid / cols;
    uint col = gid % cols;

    if (row >= rows || col >= cols) return;

    uint4 acc = 0;

    // Compute dot product of row of A with column of B
    uint k = 0;
    for (; k + 4 <= inner; k += 4) {
        uint4 a_vals = *(device uint4*)(A + row * inner + k);
        // B is column-major in our layout, need strided access
        uint4 b_vals;
        b_vals.x = B[(k + 0) * cols + col];
        b_vals.y = B[(k + 1) * cols + col];
        b_vals.z = B[(k + 2) * cols + col];
        b_vals.w = B[(k + 3) * cols + col];
        acc += bb_mul_v4(a_vals, b_vals);
    }

    // Handle remainder
    for (; k < inner; k++) {
        acc.x = bb_add(acc.x, bb_mul(A[row * inner + k], B[k * cols + col]));
    }

    uint sum = acc.x + acc.y + acc.z + acc.w;
    C[row * cols + col] = sum;
}

// Row-major matrix multiply: optimized for C = A * B where B is also row-major
kernel void tensor_matmul_row_major(
    device const uint *A [[buffer(0)]],
    device const uint *B [[buffer(1)]],
    device uint *C [[buffer(2)]],
    constant uint &rowsA [[buffer(3)]],
    constant uint &colsA [[buffer(4)]],
    constant uint &colsB [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint rows = rowsA;
    uint cols = colsB;
    uint inner = colsA;

    uint row = gid / cols;
    uint col = gid % cols;

    if (row >= rows || col >= cols) return;

    uint4 acc = 0;
    uint k = 0;

    // A[row*inner + k], B[col + k*cols] for row-major
    for (; k + 4 <= inner; k += 4) {
        uint4 a_vals = *(device uint4*)(A + row * inner + k);
        uint4 b_vals;
        uint bBase = k * cols + col;
        b_vals.x = B[bBase];
        b_vals.y = B[bBase + cols];
        b_vals.z = B[bBase + 2 * cols];
        b_vals.w = B[bBase + 3 * cols];
        acc += bb_mul_v4(a_vals, b_vals);
    }

    for (; k < inner; k++) {
        acc.x = bb_add(acc.x, bb_mul(A[row * inner + k], B[k * cols + col]));
    }

    uint sum = acc.x + acc.y + acc.z + acc.w;
    C[row * cols + col] = sum;
}
