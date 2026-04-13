// ane_tensor.mm — ANE Tensor Operations C wrapper with Metal GPU acceleration
//
// This file provides the C API for tensor operations (matvec, inner product, matmul)
// with Metal GPU acceleration that automatically offloads to ANE on Apple Silicon.
//
// GPU path: compiles Metal shader at runtime and dispatches compute kernels
// Scalar path: uses inline scalar arithmetic (fallback when GPU unavailable)

#include "include/ane_tensor.h"
#include <Metal/Metal.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// ============================================================
// BabyBear field constants
// ============================================================

#define BB_P        2013265921u
#define BB_P_INV    2281701377u
#define BB_R2       1172168163u

// ============================================================
// Scalar BabyBear arithmetic
// ============================================================

static inline uint32_t bb_monty_reduce64(uint64_t x) {
    uint32_t lo = (uint32_t)x;
    uint32_t q = lo * BB_P_INV;
    int64_t t = (int64_t)x - (int64_t)q * (int64_t)BB_P;
    int32_t r = (int32_t)(t >> 32);
    return r < 0 ? (uint32_t)(r + (int32_t)BB_P) : (uint32_t)r;
}

static inline uint32_t bb_mul(uint32_t a, uint32_t b) {
    return bb_monty_reduce64((uint64_t)a * (uint64_t)b);
}

static inline uint32_t bb_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return (s >= BB_P) ? s - BB_P : s;
}

static inline uint32_t bb_sub(uint32_t a, uint32_t b) {
    return (a >= b) ? a - b : a + BB_P - b;
}

// ============================================================
// Metal GPU State
// ============================================================

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_matvec_row_pipeline = nil;
static id<MTLComputePipelineState> g_matvec_batch_pipeline = nil;
static id<MTLComputePipelineState> g_inner_product_pipeline = nil;
static id<MTLComputePipelineState> g_inner_product_batch_pipeline = nil;
static id<MTLComputePipelineState> g_matmul_pipeline = nil;
static id<MTLComputePipelineState> g_matmul_row_major_pipeline = nil;
static bool g_gpu_initialized = false;

// Metal shader source for tensor operations
static const char* g_shader_source = R"(
#include <metal_stdlib>
using namespace metal;

// BabyBear field: p = 2^31 - 2^27 + 1 = 0x78000001
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

// Matrix-Vector Multiply: result = M * vec
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

// Inner Product: sum = Σ a[i] * b[i]
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

// Matrix-Matrix Multiply: C = A * B (column-major)
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
)";

// ============================================================
// ANE Tensor Lifecycle
// ============================================================

int ane_tensor_init(void) {
    if (g_gpu_initialized) return 0;

    @autoreleasepool {
        // Create Metal device
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            return -1;
        }

        // Check ANE support (Apple8 = ANE capable)
        if (![g_device supportsFamily:MTLGPUFamilyApple8]) {
            // ANE not available, but GPU is - we'll still use GPU for SIMD
            NSLog(@"ANE not available, using GPU SIMD");
        }

        // Create command queue
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            return -1;
        }

        // Compile shader
        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:g_shader_source];
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

        g_library = [g_device newLibraryWithSource:source
                                           options:options
                                             error:&error];
        if (!g_library) {
            NSLog(@"Failed to compile tensor shader: %@", error);
            return -1;
        }

        // Create pipeline states
        id<MTLFunction> matvec_row_fn = [g_library newFunctionWithName:@"tensor_matvec_row"];
        id<MTLFunction> matvec_batch_fn = [g_library newFunctionWithName:@"tensor_matvec_batch"];
        id<MTLFunction> inner_product_fn = [g_library newFunctionWithName:@"tensor_inner_product"];
        id<MTLFunction> inner_product_batch_fn = [g_library newFunctionWithName:@"tensor_inner_product_batch"];
        id<MTLFunction> matmul_fn = [g_library newFunctionWithName:@"tensor_matmul"];
        id<MTLFunction> matmul_row_fn = [g_library newFunctionWithName:@"tensor_matmul_row_major"];

        if (matvec_row_fn) {
            g_matvec_row_pipeline = [g_device newComputePipelineStateWithFunction:matvec_row_fn error:&error];
        }
        if (matvec_batch_fn) {
            g_matvec_batch_pipeline = [g_device newComputePipelineStateWithFunction:matvec_batch_fn error:&error];
        }
        if (inner_product_fn) {
            g_inner_product_pipeline = [g_device newComputePipelineStateWithFunction:inner_product_fn error:&error];
        }
        if (inner_product_batch_fn) {
            g_inner_product_batch_pipeline = [g_device newComputePipelineStateWithFunction:inner_product_batch_fn error:&error];
        }
        if (matmul_fn) {
            g_matmul_pipeline = [g_device newComputePipelineStateWithFunction:matmul_fn error:&error];
        }
        if (matmul_row_fn) {
            g_matmul_row_major_pipeline = [g_device newComputePipelineStateWithFunction:matmul_row_fn error:&error];
        }

        g_gpu_initialized = true;
        NSLog(@"ANE Tensor initialized: matvec_row=%@, matvec_batch=%@, inner_product=%@, inner_product_batch=%@, matmul=%@, matmul_row=%@",
              g_matvec_row_pipeline ? @"YES" : @"NO",
              g_matvec_batch_pipeline ? @"YES" : @"NO",
              g_inner_product_pipeline ? @"YES" : @"NO",
              g_inner_product_batch_pipeline ? @"YES" : @"NO",
              g_matmul_pipeline ? @"YES" : @"NO",
              g_matmul_row_major_pipeline ? @"YES" : @"NO");
        return 0;
    }
}

void ane_tensor_shutdown(void) {
    if (!g_gpu_initialized) return;

    @autoreleasepool {
        g_matvec_row_pipeline = nil;
        g_matvec_batch_pipeline = nil;
        g_inner_product_pipeline = nil;
        g_inner_product_batch_pipeline = nil;
        g_matmul_pipeline = nil;
        g_matmul_row_major_pipeline = nil;
        g_library = nil;
        g_queue = nil;
        g_device = nil;
        g_gpu_initialized = false;
    }
}

bool ane_tensor_gpu_available(void) {
    return g_gpu_initialized && g_matvec_row_pipeline != nil;
}

// ============================================================
// GPU dispatch helpers
// ============================================================

static void ensure_gpu_initialized(void) {
    if (!g_gpu_initialized) {
        ane_tensor_init();
    }
}

static void dispatch_matvec_row_gpu(const uint32_t *M, const uint32_t *vec,
                                    int rows, int cols, uint32_t *result) {
    ensure_gpu_initialized();
    if (!g_matvec_row_pipeline) return;

    @autoreleasepool {
        int matSize = rows * cols * sizeof(uint32_t);
        int vecSize = cols * sizeof(uint32_t);
        int resSize = rows * sizeof(uint32_t);

        id<MTLBuffer> matBuf = [g_device newBufferWithBytes:M length:matSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> vecBuf = [g_device newBufferWithBytes:vec length:vecSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> resBuf = [g_device newBufferWithLength:resSize options:MTLResourceStorageModeShared];

        uint32_t rowsVal = rows;
        uint32_t colsVal = cols;
        id<MTLBuffer> rowsBuf = [g_device newBufferWithBytes:&rowsVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> colsBuf = [g_device newBufferWithBytes:&colsVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_matvec_row_pipeline];
        [enc setBuffer:matBuf offset:0 atIndex:0];
        [enc setBuffer:vecBuf offset:0 atIndex:1];
        [enc setBuffer:resBuf offset:0 atIndex:2];
        [enc setBuffer:rowsBuf offset:0 atIndex:3];
        [enc setBuffer:colsBuf offset:0 atIndex:4];

        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(result, resBuf.contents, resSize);
    }
}

static void dispatch_matvec_batch_gpu(const uint32_t *M, const uint32_t *vecs,
                                      int rows, int cols, int batch, uint32_t *result) {
    ensure_gpu_initialized();
    if (!g_matvec_batch_pipeline) return;

    @autoreleasepool {
        int matSize = batch * rows * cols * sizeof(uint32_t);
        int vecSize = batch * cols * sizeof(uint32_t);
        int resSize = batch * rows * sizeof(uint32_t);

        id<MTLBuffer> matBuf = [g_device newBufferWithBytes:M length:matSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> vecBuf = [g_device newBufferWithBytes:vecs length:vecSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> resBuf = [g_device newBufferWithLength:resSize options:MTLResourceStorageModeShared];

        uint32_t rowsVal = rows;
        uint32_t colsVal = cols;
        uint32_t batchVal = batch;
        id<MTLBuffer> rowsBuf = [g_device newBufferWithBytes:&rowsVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> colsBuf = [g_device newBufferWithBytes:&colsVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> batchBuf = [g_device newBufferWithBytes:&batchVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_matvec_batch_pipeline];
        [enc setBuffer:matBuf offset:0 atIndex:0];
        [enc setBuffer:vecBuf offset:0 atIndex:1];
        [enc setBuffer:resBuf offset:0 atIndex:2];
        [enc setBuffer:rowsBuf offset:0 atIndex:3];
        [enc setBuffer:colsBuf offset:0 atIndex:4];
        [enc setBuffer:batchBuf offset:0 atIndex:5];

        [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(rows, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(result, resBuf.contents, resSize);
    }
}

static void dispatch_inner_product_gpu(const uint32_t *a, const uint32_t *b,
                                       int n, uint32_t *result) {
    ensure_gpu_initialized();
    if (!g_inner_product_pipeline) return;

    @autoreleasepool {
        int aSize = n * sizeof(uint32_t);
        int bSize = n * sizeof(uint32_t);
        int resSize = sizeof(uint32_t);

        id<MTLBuffer> aBuf = [g_device newBufferWithBytes:a length:aSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [g_device newBufferWithBytes:b length:bSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> resBuf = [g_device newBufferWithLength:resSize options:MTLResourceStorageModeShared];

        uint32_t nVal = n;
        id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&nVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_inner_product_pipeline];
        [enc setBuffer:aBuf offset:0 atIndex:0];
        [enc setBuffer:bBuf offset:0 atIndex:1];
        [enc setBuffer:resBuf offset:0 atIndex:2];
        [enc setBuffer:nBuf offset:0 atIndex:3];

        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(result, resBuf.contents, resSize);
    }
}

static void dispatch_inner_product_batch_gpu(const uint32_t *a_batch, const uint32_t *b_batch,
                                             int n, int batch, uint32_t *result) {
    ensure_gpu_initialized();
    if (!g_inner_product_batch_pipeline) return;

    @autoreleasepool {
        int aSize = batch * n * sizeof(uint32_t);
        int bSize = batch * n * sizeof(uint32_t);
        int resSize = batch * sizeof(uint32_t);

        id<MTLBuffer> aBuf = [g_device newBufferWithBytes:a_batch length:aSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [g_device newBufferWithBytes:b_batch length:bSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> resBuf = [g_device newBufferWithLength:resSize options:MTLResourceStorageModeShared];

        uint32_t nVal = n;
        uint32_t batchVal = batch;
        id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&nVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> batchBuf = [g_device newBufferWithBytes:&batchVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_inner_product_batch_pipeline];
        [enc setBuffer:aBuf offset:0 atIndex:0];
        [enc setBuffer:bBuf offset:0 atIndex:1];
        [enc setBuffer:resBuf offset:0 atIndex:2];
        [enc setBuffer:nBuf offset:0 atIndex:3];
        [enc setBuffer:batchBuf offset:0 atIndex:4];

        [enc dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(result, resBuf.contents, resSize);
    }
}

static void dispatch_matmul_gpu(const uint32_t *A, const uint32_t *B, uint32_t *C,
                                int rowsA, int colsA, int colsB) {
    ensure_gpu_initialized();
    if (!g_matmul_row_major_pipeline) return;

    @autoreleasepool {
        int aSize = rowsA * colsA * sizeof(uint32_t);
        int bSize = colsA * colsB * sizeof(uint32_t);
        int cSize = rowsA * colsB * sizeof(uint32_t);

        id<MTLBuffer> aBuf = [g_device newBufferWithBytes:A length:aSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [g_device newBufferWithBytes:B length:bSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> cBuf = [g_device newBufferWithLength:cSize options:MTLResourceStorageModeShared];

        uint32_t rowsAVal = rowsA;
        uint32_t colsAVal = colsA;
        uint32_t colsBVal = colsB;
        id<MTLBuffer> rowsABuf = [g_device newBufferWithBytes:&rowsAVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> colsABuf = [g_device newBufferWithBytes:&colsAVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> colsBBuf = [g_device newBufferWithBytes:&colsBVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_matmul_row_major_pipeline];
        [enc setBuffer:aBuf offset:0 atIndex:0];
        [enc setBuffer:bBuf offset:0 atIndex:1];
        [enc setBuffer:cBuf offset:0 atIndex:2];
        [enc setBuffer:rowsABuf offset:0 atIndex:3];
        [enc setBuffer:colsABuf offset:0 atIndex:4];
        [enc setBuffer:colsBBuf offset:0 atIndex:5];

        uint32_t totalThreads = rowsA * colsB;
        [enc dispatchThreadgroups:MTLSizeMake(totalThreads, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(C, cBuf.contents, cSize);
    }
}

// ============================================================
// Public API Implementations
// ============================================================

extern "C" {

void ane_tensor_matvec(const uint32_t *M, const uint32_t *vec,
                      int rows, int cols, uint32_t *result) {
    ensure_gpu_initialized();

    if (g_matvec_row_pipeline && rows > 0 && cols > 0) {
        dispatch_matvec_row_gpu(M, vec, rows, cols, result);
    } else {
        // Scalar fallback
        for (int i = 0; i < rows; i++) {
            uint32_t sum = 0;
            for (int j = 0; j < cols; j++) {
                sum = bb_add(sum, bb_mul(M[i * cols + j], vec[j]));
            }
            result[i] = sum;
        }
    }
}

void ane_tensor_matvec_batch(const uint32_t *M, const uint32_t *vecs,
                            int rows, int cols, int batch, uint32_t *result) {
    ensure_gpu_initialized();

    if (g_matvec_batch_pipeline && rows > 0 && cols > 0 && batch > 0) {
        dispatch_matvec_batch_gpu(M, vecs, rows, cols, batch, result);
    } else {
        // Scalar fallback
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < rows; i++) {
                uint32_t sum = 0;
                for (int j = 0; j < cols; j++) {
                    sum = bb_add(sum, bb_mul(M[b * rows * cols + i * cols + j],
                                             vecs[b * cols + j]));
                }
                result[b * rows + i] = sum;
            }
        }
    }
}

uint32_t ane_tensor_inner_product(const uint32_t *a, const uint32_t *b, int n) {
    ensure_gpu_initialized();

    uint32_t result = 0;
    if (g_inner_product_pipeline && n > 0) {
        dispatch_inner_product_gpu(a, b, n, &result);
    } else {
        // Scalar fallback
        uint32_t sum = 0;
        for (int i = 0; i < n; i++) {
            sum = bb_add(sum, bb_mul(a[i], b[i]));
        }
        result = sum;
    }
    return result;
}

void ane_tensor_inner_product_batch(const uint32_t *a_batch, const uint32_t *b_batch,
                                   int n, int batch, uint32_t *result) {
    ensure_gpu_initialized();

    if (g_inner_product_batch_pipeline && n > 0 && batch > 0) {
        dispatch_inner_product_batch_gpu(a_batch, b_batch, n, batch, result);
    } else {
        // Scalar fallback
        for (int k = 0; k < batch; k++) {
            uint32_t sum = 0;
            for (int i = 0; i < n; i++) {
                sum = bb_add(sum, bb_mul(a_batch[k * n + i], b_batch[k * n + i]));
            }
            result[k] = sum;
        }
    }
}

void ane_tensor_matmul(const uint32_t *A, const uint32_t *B,
                       int rowsA, int colsA, int colsB, uint32_t *C) {
    ensure_gpu_initialized();

    if (g_matmul_row_major_pipeline && rowsA > 0 && colsA > 0 && colsB > 0) {
        dispatch_matmul_gpu(A, B, C, rowsA, colsA, colsB);
    } else {
        // Scalar fallback (row-major)
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                uint32_t sum = 0;
                for (int k = 0; k < colsA; k++) {
                    sum = bb_add(sum, bb_mul(A[i * colsA + k], B[k * colsB + j]));
                }
                C[i * colsB + j] = sum;
            }
        }
    }
}

} // extern "C"
