// ane_tensor.h — ANE-accelerated Tensor Operations
//
// Provides GPU/ANE-accelerated matrix-vector multiply and inner product
// for use in tensor proof compression and sumcheck protocols.
//
// Supported fields: BabyBear (31-bit), M31 (31-bit), BN254 (254-bit via bigint decomposition)
//
// ANE strategy:
// - Express matvec as batched matrix multiply (ANNOTATION: ANE-friendly)
// - Use FP16 intermediate representation where precision allows
// - Batch many small operations to amortize ANE launch overhead

#ifndef ANE_TENSOR_H
#define ANE_TENSOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Lifecycle
// ============================================================

/// Initialize ANE tensor GPU support
/// Returns 0 on success, -1 on failure.
int ane_tensor_init(void);

/// Shutdown ANE tensor GPU support
void ane_tensor_shutdown(void);

/// Check if GPU tensor operations are available
bool ane_tensor_gpu_available(void);

// ============================================================
// Tensor MatVec: result = M * vec
//
// M is stored as rows x cols field elements (row-major)
// vec has cols elements
// result has rows elements
// ============================================================

/// Matrix-vector multiply: result[i] = sum_j M[i*cols+j] * vec[j]
/// Uses GPU acceleration when available, falls back to scalar.
/// @param M Matrix (rows * cols elements)
/// @param vec Vector (cols elements)
/// @param rows Number of rows in M
/// @param cols Number of columns in M
/// @param result Output vector (rows elements)
void ane_tensor_matvec(const uint32_t *M, const uint32_t *vec,
                      int rows, int cols, uint32_t *result);

/// Batch matrix-vector multiply: result[i] = M_i * vec_i for i in [0, batch)
/// Each M_i is rows x cols, each vec_i is cols, each result_i is rows
/// @param M Batch of matrices (batch * rows * cols elements)
/// @param vecs Batch of vectors (batch * cols elements)
/// @param rows Row dimension
/// @param cols Column dimension
/// @param batch Number of independent matvec operations
/// @param result Batch of output vectors (batch * rows elements)
void ane_tensor_matvec_batch(const uint32_t *M, const uint32_t *vecs,
                            int rows, int cols, int batch, uint32_t *result);

// ============================================================
// Inner Product: sum = Σ a[i] * b[i]
// ============================================================

/// Inner product of two vectors: sum = Σ a[i] * b[i]
/// @param a First vector (n elements)
/// @param b Second vector (n elements)
/// @param n Length of vectors
/// @return Inner product as field element
uint32_t ane_tensor_inner_product(const uint32_t *a, const uint32_t *b, int n);

/// Batch inner products: result[i] = Σ_j a_batch[i*n + j] * b_batch[i*n + j]
/// @param a_batch Batch of first vectors (batch * n elements)
/// @param b_batch Batch of second vectors (batch * n elements)
/// @param n Vector length
/// @param batch Number of inner products
/// @param result Output inner products (batch elements)
void ane_tensor_inner_product_batch(const uint32_t *a_batch, const uint32_t *b_batch,
                                   int n, int batch, uint32_t *result);

// ============================================================
// Matrix-Matrix Multiply: C = A * B
//
// A: rowsA x colsA (col-major or row-major)
// B: colsA x colsB
// C: rowsA x colsB
// ============================================================

/// Matrix multiply: C = A * B
/// @param A Matrix A (rowsA * colsA elements)
/// @param B Matrix B (colsA * colsB elements)
/// @param rowsA Rows of A and C
/// @param colsA Columns of A, rows of B
/// @param colsB Columns of B and C
/// @param C Output matrix (rowsA * colsB elements)
void ane_tensor_matmul(const uint32_t *A, const uint32_t *B,
                       int rowsA, int colsA, int colsB, uint32_t *C);

#ifdef __cplusplus
}
#endif

#endif // ANE_TENSOR_H
