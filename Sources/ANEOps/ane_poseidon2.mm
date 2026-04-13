// ane_poseidon2.mm — ANE Poseidon2 C wrapper with Metal GPU acceleration
//
// This file provides the C API for Poseidon2 S-box operations with
// Metal GPU acceleration that automatically offloads to ANE on Apple Silicon.
//
// GPU path: compiles Metal shader at runtime and dispatches compute kernels
// Scalar path: uses inline scalar arithmetic (fallback when GPU unavailable)

#include "include/ane_poseidon2.h"
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
// M31 field constants
// ============================================================

#define M31_P 0x7FFFFFFFu

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

static inline uint32_t bb_to_monty(uint32_t a) {
    return bb_monty_reduce64((uint64_t)a * BB_R2);
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

// BabyBear x^7 S-box (scalar)
static inline uint32_t bb_sbox_scalar(uint32_t x) {
    uint32_t x2 = bb_mul(x, x);
    uint32_t x3 = bb_mul(x2, x);
    uint32_t x6 = bb_mul(x2, x2);
    return bb_mul(x6, x);
}

// BabyBear internal diagonal constants
static const uint32_t BB_INTERNAL_DIAG[16] = {
    0x77ffffffu, 0x00000001u, 0x00000002u, 0x3c000001u,
    0x00000003u, 0x00000004u, 0x3c000000u, 0x77fffffeu,
    0x77fffffdu, 0x77880001u, 0x5a000001u, 0x69000001u,
    0x77fffff2u, 0x00780000u, 0x07800000u, 0x0000000fu
};

// M4 circulant [2,3,1,1] for BabyBear
static inline void bb_m4(uint32_t *s) {
    uint32_t t0 = bb_add(s[0], s[1]);
    uint32_t t1 = bb_add(s[2], s[3]);
    uint32_t t2 = bb_add(bb_add(s[1], s[1]), t1);
    uint32_t t3 = bb_add(bb_add(s[3], s[3]), t0);
    s[0] = bb_add(t0, t3);
    s[1] = bb_add(t1, t2);
    s[2] = bb_add(t0, t2);
    s[3] = bb_add(t1, t3);
}

// BabyBear external linear layer
static void bb_external_layer(uint32_t *s) {
    bb_m4(s);
    bb_m4(s + 4);
    bb_m4(s + 8);
    bb_m4(s + 12);
    for (int i = 0; i < 4; i++) {
        uint32_t sum = bb_add(bb_add(s[i], s[i+4]), bb_add(s[i+8], s[i+12]));
        s[i]     = bb_add(s[i], sum);
        s[i+4]   = bb_add(s[i+4], sum);
        s[i+8]   = bb_add(s[i+8], sum);
        s[i+12]  = bb_add(s[i+12], sum);
    }
}

// BabyBear internal linear layer
static void bb_internal_layer(uint32_t *s) {
    uint32_t sum = 0;
    for (int i = 0; i < 16; i++) sum = bb_add(sum, s[i]);
    for (int i = 0; i < 16; i++) {
        uint32_t d = BB_INTERNAL_DIAG[i];
        uint32_t prod;
        if (d == 1) {
            prod = s[i];
        } else if (d == 2) {
            prod = bb_add(s[i], s[i]);
        } else {
            prod = bb_mul(s[i], d);
        }
        s[i] = bb_add(prod, sum);
    }
}

// ============================================================
// M31 scalar arithmetic
// ============================================================

static inline uint32_t m31_reduce(uint32_t x) {
    uint32_t r = (x >> 31) + (x & M31_P);
    return (r >= M31_P) ? r - M31_P : r;
}

static inline uint32_t m31_add(uint32_t a, uint32_t b) {
    return m31_reduce(a + b);
}

static inline uint32_t m31_sub(uint32_t a, uint32_t b) {
    return (a >= b) ? a - b : a + M31_P - b;
}

static inline uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    uint32_t lo = (uint32_t)(prod & M31_P);
    uint32_t hi = (uint32_t)(prod >> 31);
    return m31_reduce(lo + hi);
}

// M31 x^5 S-box
static inline uint32_t m31_sbox_scalar(uint32_t x) {
    uint32_t x2 = m31_mul(x, x);
    uint32_t x4 = m31_mul(x2, x2);
    return m31_mul(x4, x);
}

static const uint32_t M31_INTERNAL_DIAG[16] = {
    1, 1, 2, 1, 8, 32, 2, 256, 4096, 8, 65536, 1024, 2, 16384, 512, 32768
};

static inline void m31_m4(uint32_t *s) {
    uint32_t t0 = m31_add(s[0], s[1]);
    uint32_t t1 = m31_add(s[2], s[3]);
    uint32_t t2 = m31_add(m31_add(s[1], s[1]), t1);
    uint32_t t3 = m31_add(m31_add(s[3], s[3]), t0);
    s[0] = m31_add(t0, t3);
    s[1] = m31_add(t1, t2);
    s[2] = m31_add(t0, t2);
    s[3] = m31_add(t1, t3);
}

static void m31_external_layer(uint32_t *s) {
    m31_m4(s);
    m31_m4(s + 4);
    m31_m4(s + 8);
    m31_m4(s + 12);
    for (int i = 0; i < 4; i++) {
        uint32_t sum = m31_add(m31_add(s[i], s[i+4]), m31_add(s[i+8], s[i+12]));
        s[i]     = m31_add(s[i], sum);
        s[i+4]   = m31_add(s[i+4], sum);
        s[i+8]   = m31_add(s[i+8], sum);
        s[i+12]  = m31_add(s[i+12], sum);
    }
}

static void m31_internal_layer(uint32_t *s) {
    uint32_t sum = 0;
    for (int i = 0; i < 16; i++) sum = m31_add(sum, s[i]);
    for (int i = 0; i < 16; i++) {
        uint32_t d = M31_INTERNAL_DIAG[i];
        uint32_t prod;
        if (d == 1) {
            prod = s[i];
        } else if (d == 2) {
            prod = m31_add(s[i], s[i]);
        } else {
            prod = m31_mul(s[i], d);
        }
        s[i] = m31_add(prod, sum);
    }
}

// ============================================================
// Full round functions
// ============================================================

static void bb_full_round(uint32_t *s, const uint32_t *rc) {
    for (int i = 0; i < 16; i++) s[i] = bb_add(s[i], rc[i]);
    for (int i = 0; i < 16; i++) s[i] = bb_sbox_scalar(s[i]);
    bb_external_layer(s);
}

static void bb_partial_round(uint32_t *s, uint32_t rc0) {
    s[0] = bb_add(s[0], rc0);
    s[0] = bb_sbox_scalar(s[0]);
    bb_internal_layer(s);
}

static void m31_full_round(uint32_t *s, const uint32_t *rc) {
    for (int i = 0; i < 16; i++) s[i] = m31_add(s[i], rc[i]);
    for (int i = 0; i < 16; i++) s[i] = m31_sbox_scalar(s[i]);
    m31_external_layer(s);
}

static void m31_partial_round(uint32_t *s, uint32_t rc0) {
    s[0] = m31_add(s[0], rc0);
    s[0] = m31_sbox_scalar(s[0]);
    m31_internal_layer(s);
}

// ============================================================
// Metal GPU State
// ============================================================

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_bb_sbox_pipeline = nil;
static id<MTLComputePipelineState> g_m31_sbox_pipeline = nil;
static id<MTLComputePipelineState> g_bb_batch_pipeline = nil;
static id<MTLComputePipelineState> g_m31_batch_pipeline = nil;
static id<MTLComputePipelineState> g_bb_perm_pipeline = nil;
static id<MTLComputePipelineState> g_m31_perm_pipeline = nil;
static id<MTLComputePipelineState> g_bb_perm_batch_pipeline = nil;
static id<MTLComputePipelineState> g_m31_perm_batch_pipeline = nil;
static id<MTLComputePipelineState> g_m31_perm_chunk_pipeline = nil;
static id<MTLComputePipelineState> g_m31_perm_12_pipeline = nil;
static bool g_gpu_initialized = false;

// ============================================================
// Buffer Pool for Reusing Metal Buffers
// ============================================================

#define BUFFER_POOL_SIZE 8
#define MAX_BUFFER_SIZE (1 << 20)  // 1M elements max

typedef struct {
    id<MTLBuffer> buffer;
    int capacity;  // in uint32_t elements
    bool inUse;
} BufferEntry;

static BufferEntry g_inputPool[BUFFER_POOL_SIZE];
static BufferEntry g_outputPool[BUFFER_POOL_SIZE];
static id<MTLBuffer> g_rcBuffer = nil;  // Round constants buffer (shared, reusable)
static NSUInteger g_bufferPoolLock = 0;

static void init_buffer_pools(void) {
    for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
        g_inputPool[i].buffer = nil;
        g_inputPool[i].capacity = 0;
        g_inputPool[i].inUse = false;
        g_outputPool[i].buffer = nil;
        g_outputPool[i].capacity = 0;
        g_outputPool[i].inUse = false;
    }
    // Pre-allocate round constants buffer
    if (!g_rcBuffer) {
        g_rcBuffer = [g_device newBufferWithLength:560 * sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
    }
}

static id<MTLBuffer> acquire_buffer(BufferEntry *pool, int size, bool *gotNew) {
    *gotNew = false;
    for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
        if (!pool[i].inUse && pool[i].capacity >= size) {
            pool[i].inUse = true;
            return pool[i].buffer;
        }
    }
    // Need to allocate new
    for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
        if (!pool[i].inUse) {
            pool[i].capacity = size;
            pool[i].buffer = [g_device newBufferWithLength:size * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
            pool[i].inUse = true;
            *gotNew = true;
            return pool[i].buffer;
        }
    }
    return nil;  // Pool exhausted
}

static void release_buffer(BufferEntry *pool, id<MTLBuffer> buffer) {
    for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
        if (pool[i].buffer == buffer) {
            pool[i].inUse = false;
            return;
        }
    }
}

// Forward declaration (defined after shader source)
static void ensure_gpu_initialized(void);

// Synchronous version with buffer pooling - waits for completion
static bool dispatch_m31_perm_batch_gpu(const uint32_t *states, int n_perms,
                                        const uint32_t *round_constants,
                                        uint32_t *output) {
    ensure_gpu_initialized();
    if (!g_m31_perm_batch_pipeline || n_perms <= 0) return false;

    @autoreleasepool {
        int totalSize = n_perms * 16;

        // Try to use pooled buffers for small/medium sizes
        if (totalSize <= MAX_BUFFER_SIZE) {
            bool gotNewInput, gotNewOutput;
            id<MTLBuffer> inputBuf = acquire_buffer(g_inputPool, totalSize, &gotNewInput);
            id<MTLBuffer> outputBuf = acquire_buffer(g_outputPool, totalSize, &gotNewOutput);

            if (inputBuf && outputBuf) {
                // Copy input data
                memcpy(inputBuf.contents, states, totalSize * sizeof(uint32_t));
                memcpy(g_rcBuffer.contents, round_constants, 560 * sizeof(uint32_t));

                uint32_t nPermsVal = n_perms;
                id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&nPermsVal
                                                           length:sizeof(uint32_t)
                                                          options:MTLResourceStorageModeShared];

                id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

                [enc setComputePipelineState:g_m31_perm_batch_pipeline];
                [enc setBuffer:inputBuf offset:0 atIndex:0];
                [enc setBuffer:outputBuf offset:0 atIndex:1];
                [enc setBuffer:g_rcBuffer offset:0 atIndex:2];
                [enc setBuffer:nBuf offset:0 atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(n_perms, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                [enc endEncoding];

                [cmdBuf commit];
                [cmdBuf waitUntilCompleted];

                memcpy(output, outputBuf.contents, totalSize * sizeof(uint32_t));
                release_buffer(g_inputPool, inputBuf);
                release_buffer(g_outputPool, outputBuf);
                return true;
            }
            if (inputBuf) release_buffer(g_inputPool, inputBuf);
            if (outputBuf) release_buffer(g_outputPool, outputBuf);
        }

        // Fallback: allocate new buffers
        id<MTLBuffer> inputBuf = [g_device newBufferWithBytes:states
                                                      length:totalSize * sizeof(uint32_t)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuf = [g_device newBufferWithLength:totalSize * sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> rcBuf = [g_device newBufferWithBytes:round_constants
                                                    length:560 * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
        uint32_t nPermsVal = n_perms;
        id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&nPermsVal
                                                   length:sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_m31_perm_batch_pipeline];
        [enc setBuffer:inputBuf offset:0 atIndex:0];
        [enc setBuffer:outputBuf offset:0 atIndex:1];
        [enc setBuffer:rcBuf offset:0 atIndex:2];
        [enc setBuffer:nBuf offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(n_perms, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(output, outputBuf.contents, totalSize * sizeof(uint32_t));
        return true;
    }
}

// Metal shader source (standard Metal, no metal_ane dependency)
// Restructured for GPU parallelism: uses vectorized ops and processes 16 elements per thread
static const char* g_shader_source = R"(
#include <metal_stdlib>
using namespace metal;

// BabyBear field: p = 2^31 - 2^27 + 1 = 0x78000001
constant uint BB_P = 0x78000001u;
constant uint BB_P_INV = 2281701377u;

// M31 field: p = 2^31 - 1 = 0x7FFFFFFF
constant uint M31_P = 0x7FFFFFFFu;

// ============================================================
// BabyBear SIMD arithmetic (4 elements at once)
// ============================================================

// Barrett reduction for 4 elements at once
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

// BabyBear x^7 S-box for 4 elements using vectorized ops
inline uint4 bb_sbox_v4(uint4 x) {
    uint4 x2 = bb_mul_v4(x, x);
    uint4 x3 = bb_mul_v4(x2, x);
    uint4 x6 = bb_mul_v4(x2, x2);
    return bb_mul_v4(x6, x);
}

// ============================================================
// M31 SIMD arithmetic (4 elements at once)
// ============================================================

inline uint4 m31_reduce_v4(uint4 x) {
    uint4 r = (x >> 31) + (x & M31_P);
    return select(r - M31_P, r, r < M31_P);
}

inline uint4 m31_mul_v4(uint4 a, uint4 b) {
    ulong4 prod = (ulong4)a * (ulong4)b;
    uint4 lo = (uint4)(prod & M31_P);
    uint4 hi = (uint4)(prod >> 31);
    uint4 s = lo + hi;
    uint4 r = (s & M31_P) + (s >> 31);
    return select(r - M31_P, r, r < M31_P);
}

inline uint4 m31_add_v4(uint4 a, uint4 b) {
    uint4 s = a + b;
    uint4 r = (s & M31_P) + (s >> 31);
    return select(r - M31_P, r, r < M31_P);
}

inline uint4 m31_sbox_v4(uint4 x) {
    uint4 x2 = m31_mul_v4(x, x);
    uint4 x4 = m31_mul_v4(x2, x2);
    return m31_mul_v4(x4, x);
}

// ============================================================
// BabyBear Poseidon2 S-box: process 16 elements per thread group
// Uses vectorized operations for SIMD parallelism
// ============================================================

kernel void bb_poseidon2_sbox_ane(
    device uint32_t *state [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread processes 16 elements (4 uint4 vectors)
    if (gid >= 1) return;

    uint4 v0 = *(device uint4*)(state + 0);
    uint4 v1 = *(device uint4*)(state + 4);
    uint4 v2 = *(device uint4*)(state + 8);
    uint4 v3 = *(device uint4*)(state + 12);

    uint4 r0 = bb_sbox_v4(v0);
    uint4 r1 = bb_sbox_v4(v1);
    uint4 r2 = bb_sbox_v4(v2);
    uint4 r3 = bb_sbox_v4(v3);

    *(device uint4*)(state + 0) = r0;
    *(device uint4*)(state + 4) = r1;
    *(device uint4*)(state + 8) = r2;
    *(device uint4*)(state + 12) = r3;
}

// Batch version: process n_groups of 16 elements
kernel void bb_poseidon2_sbox_batch_ane(
    device const uint32_t *states [[buffer(0)]],
    device uint32_t *output [[buffer(1)]],
    constant uint &n_groups [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_groups) return;

    uint base = gid * 16;
    uint4 v0 = *(device uint4*)(states + base + 0);
    uint4 v1 = *(device uint4*)(states + base + 4);
    uint4 v2 = *(device uint4*)(states + base + 8);
    uint4 v3 = *(device uint4*)(states + base + 12);

    uint4 r0 = bb_sbox_v4(v0);
    uint4 r1 = bb_sbox_v4(v1);
    uint4 r2 = bb_sbox_v4(v2);
    uint4 r3 = bb_sbox_v4(v3);

    *(device uint4*)(output + base + 0) = r0;
    *(device uint4*)(output + base + 4) = r1;
    *(device uint4*)(output + base + 8) = r2;
    *(device uint4*)(output + base + 12) = r3;
}

// ============================================================
// M31 Poseidon2 S-box: process 16 elements per thread group
// ============================================================

kernel void m31_poseidon2_sbox_ane(
    device uint32_t *state [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= 1) return;

    uint4 v0 = *(device uint4*)(state + 0);
    uint4 v1 = *(device uint4*)(state + 4);
    uint4 v2 = *(device uint4*)(state + 8);
    uint4 v3 = *(device uint4*)(state + 12);

    uint4 r0 = m31_sbox_v4(v0);
    uint4 r1 = m31_sbox_v4(v1);
    uint4 r2 = m31_sbox_v4(v2);
    uint4 r3 = m31_sbox_v4(v3);

    *(device uint4*)(state + 0) = r0;
    *(device uint4*)(state + 4) = r1;
    *(device uint4*)(state + 8) = r2;
    *(device uint4*)(state + 12) = r3;
}

kernel void m31_poseidon2_sbox_batch_ane(
    device const uint32_t *states [[buffer(0)]],
    device uint32_t *output [[buffer(1)]],
    constant uint &n_groups [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_groups) return;

    uint base = gid * 16;
    uint4 v0 = *(device uint4*)(states + base + 0);
    uint4 v1 = *(device uint4*)(states + base + 4);
    uint4 v2 = *(device uint4*)(states + base + 8);
    uint4 v3 = *(device uint4*)(states + base + 12);

    uint4 r0 = m31_sbox_v4(v0);
    uint4 r1 = m31_sbox_v4(v1);
    uint4 r2 = m31_sbox_v4(v2);
    uint4 r3 = m31_sbox_v4(v3);

    *(device uint4*)(output + base + 0) = r0;
    *(device uint4*)(output + base + 4) = r1;
    *(device uint4*)(output + base + 8) = r2;
    *(device uint4*)(output + base + 12) = r3;
}

// ============================================================
// BabyBear Poseidon2 permutation helpers (scalar, inlined)
// ============================================================

inline uint bb_p_mul(uint a, uint b) {
    ulong prod = (ulong)a * (ulong)b;
    uint prod_lo = (uint)prod;
    uint prod_hi = (uint)(prod >> 32);
    ulong t1 = (ulong)prod_lo * (ulong)BB_P_INV;
    ulong t2 = (ulong)prod_hi * (ulong)BB_P_INV;
    uint q = (uint)((t2 + (t1 >> 32)) >> 30);
    uint r = (uint)(prod - (ulong)q * (ulong)BB_P);
    return (r >= BB_P) ? r - BB_P : r;
}

inline uint bb_p_add(uint a, uint b) {
    uint s = a + b;
    return (s >= BB_P) ? s - BB_P : s;
}

inline uint bb_p_sbox(uint x) {
    uint x2 = bb_p_mul(x, x);
    uint x3 = bb_p_mul(x2, x);
    uint x6 = bb_p_mul(x2, x2);
    return bb_p_mul(x6, x);
}

inline void bb_p_m4(thread uint &s0, thread uint &s1, thread uint &s2, thread uint &s3) {
    uint t0 = bb_p_add(s0, s1);
    uint t1 = bb_p_add(s2, s3);
    uint t2 = bb_p_add(bb_p_add(s1, s1), t1);
    uint t3 = bb_p_add(bb_p_add(s3, s3), t0);
    s0 = bb_p_add(t0, t3);
    s1 = bb_p_add(t1, t2);
    s2 = bb_p_add(t0, t2);
    s3 = bb_p_add(t1, t3);
}

inline void bb_p_external_layer(thread uint *s) {
    bb_p_m4(s[0], s[1], s[2], s[3]);
    bb_p_m4(s[4], s[5], s[6], s[7]);
    bb_p_m4(s[8], s[9], s[10], s[11]);
    bb_p_m4(s[12], s[13], s[14], s[15]);
    for (uint i = 0; i < 4; i++) {
        uint sum = bb_p_add(bb_p_add(s[i], s[i+4]), bb_p_add(s[i+8], s[i+12]));
        s[i]     = bb_p_add(s[i], sum);
        s[i+4]   = bb_p_add(s[i+4], sum);
        s[i+8]   = bb_p_add(s[i+8], sum);
        s[i+12]  = bb_p_add(s[i+12], sum);
    }
}

inline void bb_p_internal_layer(thread uint *s, constant uint32_t *diag) {
    uint sum = 0;
    for (uint i = 0; i < 16; i++) sum = bb_p_add(sum, s[i]);
    for (uint i = 0; i < 16; i++) {
        uint d = diag[i];
        uint prod;
        if (d == 1) {
            prod = s[i];
        } else if (d == 2) {
            prod = bb_p_add(s[i], s[i]);
        } else {
            prod = bb_p_mul(s[i], d);
        }
        s[i] = bb_p_add(prod, sum);
    }
}

inline void bb_p_full_round(thread uint *s, constant uint32_t *rc) {
    for (uint i = 0; i < 16; i++) s[i] = bb_p_add(s[i], rc[i]);
    for (uint i = 0; i < 16; i++) s[i] = bb_p_sbox(s[i]);
    bb_p_external_layer(s);
}

inline void bb_p_partial_round(thread uint *s, uint32_t rc0, constant uint32_t *diag) {
    s[0] = bb_p_add(s[0], rc0);
    s[0] = bb_p_sbox(s[0]);
    bb_p_internal_layer(s, diag);
}

// M31 Poseidon2 permutation helpers
inline uint m31_p_reduce(uint x) {
    uint r = (x >> 31) + (x & M31_P);
    return (r >= M31_P) ? r - M31_P : r;
}

inline uint m31_p_mul(uint a, uint b) {
    ulong prod = (ulong)a * (ulong)b;
    uint lo = (uint)(prod & M31_P);
    uint hi = (uint)(prod >> 31);
    uint s = lo + hi;
    uint r = (s & M31_P) + (s >> 31);
    return (r == M31_P) ? 0 : r;
}

inline uint m31_p_add(uint a, uint b) {
    return m31_p_reduce(a + b);
}

inline uint m31_p_sbox(uint x) {
    uint x2 = m31_p_mul(x, x);
    uint x4 = m31_p_mul(x2, x2);
    return m31_p_mul(x4, x);
}

inline void m31_p_m4(thread uint &s0, thread uint &s1, thread uint &s2, thread uint &s3) {
    uint t0 = m31_p_add(s0, s1);
    uint t1 = m31_p_add(s2, s3);
    uint t2 = m31_p_add(m31_p_add(s1, s1), t1);
    uint t3 = m31_p_add(m31_p_add(s3, s3), t0);
    s0 = m31_p_add(t0, t3);
    s1 = m31_p_add(t1, t2);
    s2 = m31_p_add(t0, t2);
    s3 = m31_p_add(t1, t3);
}

inline void m31_p_external_layer(thread uint *s) {
    m31_p_m4(s[0], s[1], s[2], s[3]);
    m31_p_m4(s[4], s[5], s[6], s[7]);
    m31_p_m4(s[8], s[9], s[10], s[11]);
    m31_p_m4(s[12], s[13], s[14], s[15]);
    for (uint i = 0; i < 4; i++) {
        uint sum = m31_p_add(m31_p_add(s[i], s[i+4]), m31_p_add(s[i+8], s[i+12]));
        s[i]     = m31_p_add(s[i], sum);
        s[i+4]   = m31_p_add(s[i+4], sum);
        s[i+8]   = m31_p_add(s[i+8], sum);
        s[i+12]  = m31_p_add(s[i+12], sum);
    }
}

// M31 internal diagonal constants for Poseidon2
constant uint M31_INTERNAL_DIAG[16] = {
    1, 1, 2, 1, 8, 32, 2, 256, 4096, 8, 65536, 1024, 2, 16384, 512, 32768
};

inline void m31_p_internal_layer(thread uint *s) {
    uint sum = 0;
    for (uint i = 0; i < 16; i++) sum = m31_p_add(sum, s[i]);
    for (uint i = 0; i < 16; i++) {
        uint d = M31_INTERNAL_DIAG[i];
        uint prod;
        if (d == 1) {
            prod = s[i];
        } else if (d == 2) {
            prod = m31_p_add(s[i], s[i]);
        } else {
            prod = m31_p_mul(s[i], d);
        }
        s[i] = m31_p_add(prod, sum);
    }
}

inline void m31_p_full_round(thread uint *s, constant uint32_t *rc) {
    for (uint i = 0; i < 16; i++) s[i] = m31_p_add(s[i], rc[i]);
    for (uint i = 0; i < 16; i++) s[i] = m31_p_sbox(s[i]);
    m31_p_external_layer(s);
}

inline void m31_p_partial_round(thread uint *s, uint32_t rc0) {
    s[0] = m31_p_add(s[0], rc0);
    s[0] = m31_p_sbox(s[0]);
    m31_p_internal_layer(s);
}

// ============================================================
// BabyBear Poseidon2 Full Permutation Kernel
// Width=16, x^7 S-box, 4 full + 13 partial + 4 full = 21 rounds
// ============================================================

kernel void bb_poseidon2_permutation_ane(
    device uint32_t *state [[buffer(0)]],
    constant uint32_t *round_constants [[buffer(1)]],
    constant uint32_t *internal_diag [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= 1) return;  // Single permutation per dispatch

    uint s[16];
    for (uint i = 0; i < 16; i++) s[i] = state[i];

    int rc_idx = 0;

    // First half of full rounds (0..3)
    for (uint r = 0; r < 4; r++) {
        bb_p_full_round(s, round_constants + rc_idx);
        rc_idx += 16;
    }

    // Partial rounds (4..16)
    for (uint r = 0; r < 13; r++) {
        bb_p_partial_round(s, round_constants[rc_idx], internal_diag);
        rc_idx += 1;
    }

    // Second half of full rounds (17..20)
    for (uint r = 0; r < 4; r++) {
        bb_p_full_round(s, round_constants + rc_idx);
        rc_idx += 16;
    }

    for (uint i = 0; i < 16; i++) state[i] = s[i];
}

// ============================================================
// Batched Poseidon2 Permutation Kernels
// Process N permutations in a single GPU dispatch
// ============================================================

// BabyBear Poseidon2 batched full permutation
kernel void bb_poseidon2_permutation_batch_ane(
    device const uint32_t *states [[buffer(0)]],
    device uint32_t *output [[buffer(1)]],
    constant uint32_t *round_constants [[buffer(2)]],
    constant uint32_t *internal_diag [[buffer(3)]],
    constant uint &n_perms [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_perms) return;

    uint base = gid * 16;
    uint s[16];
    for (uint i = 0; i < 16; i++) s[i] = states[base + i];

    int rc_idx = 0;

    // First half of full rounds (0..3)
    for (uint r = 0; r < 4; r++) {
        bb_p_full_round(s, round_constants + rc_idx);
        rc_idx += 16;
    }

    // Partial rounds (4..16)
    for (uint r = 0; r < 13; r++) {
        bb_p_partial_round(s, round_constants[rc_idx], internal_diag);
        rc_idx += 1;
    }

    // Second half of full rounds (17..20)
    for (uint r = 0; r < 4; r++) {
        bb_p_full_round(s, round_constants + rc_idx);
        rc_idx += 16;
    }

    for (uint i = 0; i < 16; i++) output[base + i] = s[i];
}

// M31 Poseidon2 batched full permutation
// Width=16, x^5 S-box, 7 full + 21 partial + 7 full = 35 rounds
kernel void m31_poseidon2_permutation_batch_ane(
    device const uint32_t *states [[buffer(0)]],
    device uint32_t *output [[buffer(1)]],
    constant uint32_t *round_constants [[buffer(2)]],
    constant uint &n_perms [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_perms) return;

    uint base = gid * 16;
    uint s[16];
    for (uint i = 0; i < 16; i++) s[i] = states[base + i];

    // Initial external linear layer (required before first round, matches CPU)
    m31_p_external_layer(s);

    int rc_idx = 0;

    // First half of full rounds (0..6)
    for (uint r = 0; r < 7; r++) {
        m31_p_full_round(s, round_constants + rc_idx);
        rc_idx += 16;
    }

    // Partial rounds (7..27) - use stride-16 indexing to match Metal shader
    for (uint r = 7; r < 28; r++) {
        m31_p_partial_round(s, round_constants[r * 16]);
    }

    // Second half of full rounds (28..34)
    for (uint r = 28; r < 35; r++) {
        m31_p_full_round(s, round_constants + r * 16);
    }

    for (uint i = 0; i < 16; i++) output[base + i] = s[i];
}

// CHUNKED VERSION: processes 7 rounds per dispatch to avoid Metal complexity limits
// Total: 35 rounds = 5 dispatches (7 + 7 + 7 + 7 + 7)
kernel void m31_poseidon2_permutation_chunk_ane(
    device const uint32_t *states [[buffer(0)]],
    device uint32_t *output [[buffer(1)]],
    constant uint32_t *round_constants [[buffer(2)]],
    constant uint &n_perms [[buffer(3)]],
    constant uint &round_start [[buffer(4)]],  // 0, 7, 14, 21, 28
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_perms) return;

    uint base = gid * 16;
    uint s[16];
    for (uint i = 0; i < 16; i++) s[i] = states[base + i];

    // Initial external linear layer (only for first chunk, matches CPU)
    if (round_start == 0) {
        m31_p_external_layer(s);
    }

    uint rc_idx = round_start * 16;

    // Process 7 rounds (full or partial depending on stage)
    for (uint r = 0; r < 7; r++) {
        uint abs_round = round_start + r;
        if (abs_round < 7) {
            // First full rounds (0-6)
            m31_p_full_round(s, round_constants + rc_idx);
            rc_idx += 16;
        } else if (abs_round < 28) {
            // Partial rounds (7-27) - use stride-16 indexing to match Metal shader
            m31_p_partial_round(s, round_constants[abs_round * 16]);
        } else {
            // Last full rounds (28-34)
            m31_p_full_round(s, round_constants + rc_idx);
            rc_idx += 16;
        }
    }

    for (uint i = 0; i < 16; i++) output[base + i] = s[i];
}

// SINGLE DISPATCH VERSION: processes up to 12 rounds (avoids full 35-round kernel)
kernel void m31_poseidon2_permutation_12_ane(
    device const uint32_t *states [[buffer(0)]],
    device uint32_t *output [[buffer(1)]],
    constant uint32_t *round_constants [[buffer(2)]],
    constant uint &n_perms [[buffer(3)]],
    constant uint &round_offset [[buffer(4)]],  // 0, 12, 24
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_perms) return;

    uint base = gid * 16;
    uint s[16];
    for (uint i = 0; i < 16; i++) s[i] = states[base + i];

    // Initial external linear layer (only for first chunk, matches CPU)
    if (round_offset == 0) {
        m31_p_external_layer(s);
    }

    uint rc_idx = round_offset * 16;
    uint total_rounds = 35;

    for (uint r = 0; r < 12 && (round_offset + r) < total_rounds; r++) {
        uint abs_round = round_offset + r;

        if (abs_round < 7) {
            // Full rounds 0-6
            m31_p_full_round(s, round_constants + rc_idx);
            rc_idx += 16;
        } else if (abs_round < 28) {
            // Partial rounds 7-27 - use stride-16 indexing to match Metal shader
            m31_p_partial_round(s, round_constants[abs_round * 16]);
        } else {
            // Full rounds 28-34
            m31_p_full_round(s, round_constants + rc_idx);
            rc_idx += 16;
        }
    }

    for (uint i = 0; i < 16; i++) output[base + i] = s[i];
})"
;


int ane_poseidon2_init(void) {
    if (g_gpu_initialized) return 0;

    @autoreleasepool {
        // Create Metal device
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            return -1;
        }

        // Check ANE support (Apple8 = ANE capable)
        if (![g_device supportsFamily:MTLGPUFamilyApple8]) {
            // ANE not available, but GPU is - we'll use scalar
            g_gpu_initialized = true;
            return 0;
        }

        // Create command queue
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            return -1;
        }

        // Initialize buffer pools for reuse
        init_buffer_pools();

        // Compile shader
        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:g_shader_source];
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

        g_library = [g_device newLibraryWithSource:source
                                           options:options
                                             error:&error];
        if (!g_library) {
            // Shader compilation failed - log error
            NSLog(@"Metal shader compilation failed: %@", error);
            return -1;
        }

        // Create pipeline states
        id<MTLFunction> bb_sbox_fn = [g_library newFunctionWithName:@"bb_poseidon2_sbox_ane"];
        id<MTLFunction> m31_sbox_fn = [g_library newFunctionWithName:@"m31_poseidon2_sbox_ane"];
        id<MTLFunction> bb_batch_fn = [g_library newFunctionWithName:@"bb_poseidon2_sbox_batch_ane"];
        id<MTLFunction> m31_batch_fn = [g_library newFunctionWithName:@"m31_poseidon2_sbox_batch_ane"];
        id<MTLFunction> bb_perm_fn = [g_library newFunctionWithName:@"bb_poseidon2_permutation_ane"];
        id<MTLFunction> m31_perm_fn = [g_library newFunctionWithName:@"m31_poseidon2_permutation_ane"];
        id<MTLFunction> bb_perm_batch_fn = [g_library newFunctionWithName:@"bb_poseidon2_permutation_batch_ane"];
        id<MTLFunction> m31_perm_batch_fn = [g_library newFunctionWithName:@"m31_poseidon2_permutation_batch_ane"];
        id<MTLFunction> m31_perm_chunk_fn = [g_library newFunctionWithName:@"m31_poseidon2_permutation_chunk_ane"];
        id<MTLFunction> m31_perm_12_fn = [g_library newFunctionWithName:@"m31_poseidon2_permutation_12_ane"];

        if (bb_sbox_fn) {
            g_bb_sbox_pipeline = [g_device newComputePipelineStateWithFunction:bb_sbox_fn error:&error];
        }
        if (m31_sbox_fn) {
            g_m31_sbox_pipeline = [g_device newComputePipelineStateWithFunction:m31_sbox_fn error:&error];
        }
        if (bb_batch_fn) {
            g_bb_batch_pipeline = [g_device newComputePipelineStateWithFunction:bb_batch_fn error:&error];
        }
        if (m31_batch_fn) {
            g_m31_batch_pipeline = [g_device newComputePipelineStateWithFunction:m31_batch_fn error:&error];
        }
        if (bb_perm_fn) {
            g_bb_perm_pipeline = [g_device newComputePipelineStateWithFunction:bb_perm_fn error:&error];
        }
        if (m31_perm_fn) {
            g_m31_perm_pipeline = [g_device newComputePipelineStateWithFunction:m31_perm_fn error:&error];
        }
        if (bb_perm_batch_fn) {
            g_bb_perm_batch_pipeline = [g_device newComputePipelineStateWithFunction:bb_perm_batch_fn error:&error];
        }
        if (m31_perm_batch_fn) {
            g_m31_perm_batch_pipeline = [g_device newComputePipelineStateWithFunction:m31_perm_batch_fn error:&error];
        }
        if (m31_perm_chunk_fn) {
            g_m31_perm_chunk_pipeline = [g_device newComputePipelineStateWithFunction:m31_perm_chunk_fn error:&error];
        }
        if (m31_perm_12_fn) {
            g_m31_perm_12_pipeline = [g_device newComputePipelineStateWithFunction:m31_perm_12_fn error:&error];
        }

        g_gpu_initialized = true;
        return 0;
    }
}

void ane_poseidon2_shutdown(void) {
    if (!g_gpu_initialized) return;

    @autoreleasepool {
        g_bb_sbox_pipeline = nil;
        g_m31_sbox_pipeline = nil;
        g_bb_batch_pipeline = nil;
        g_m31_batch_pipeline = nil;
        g_bb_perm_pipeline = nil;
        g_m31_perm_pipeline = nil;
        g_bb_perm_batch_pipeline = nil;
        g_m31_perm_batch_pipeline = nil;
        g_m31_perm_chunk_pipeline = nil;
        g_m31_perm_12_pipeline = nil;
        g_library = nil;
        g_queue = nil;
        g_device = nil;
        g_gpu_initialized = false;
    }
}

bool ane_poseidon2_gpu_available(void) {
    return g_gpu_initialized && (g_bb_sbox_pipeline != nil || g_bb_perm_pipeline != nil);
}

// Debug: get pipeline status (bitmask: bit0=BB_SBOX, bit1=M31_SBOX, bit2=BB_BATCH, bit3=M31_BATCH, bit4=BB_PERM, bit5=M31_PERM, bit6=BB_PERM_BATCH, bit7=M31_PERM_BATCH, bit8=M31_PERM_CHUNK, bit9=M31_PERM_12)
int ane_poseidon2_debug_pipeline_status(void) {
    int status = 0;
    if (g_bb_sbox_pipeline) status |= 1;
    if (g_m31_sbox_pipeline) status |= 2;
    if (g_bb_batch_pipeline) status |= 4;
    if (g_m31_batch_pipeline) status |= 8;
    if (g_bb_perm_pipeline) status |= 16;
    if (g_m31_perm_pipeline) status |= 32;
    if (g_bb_perm_batch_pipeline) status |= 64;
    if (g_m31_perm_batch_pipeline) status |= 128;
    if (g_m31_perm_chunk_pipeline) status |= 256;
    if (g_m31_perm_12_pipeline) status |= 512;
    return status;
}

// ============================================================
// GPU dispatch helpers
// ============================================================

static void ensure_gpu_initialized(void) {
    if (!g_gpu_initialized) {
        ane_poseidon2_init();
    }
}

static void dispatch_bb_sbox_gpu(uint32_t *state) {
    ensure_gpu_initialized();
    if (!g_bb_sbox_pipeline) return;

    @autoreleasepool {
        // Create buffers
        id<MTLBuffer> stateBuf = [g_device newBufferWithBytes:state
                                                     length:16 * sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_bb_sbox_pipeline];
        [enc setBuffer:stateBuf offset:0 atIndex:0];
        // Use 4 threadgroups for single S-box (lower latency)
        [enc dispatchThreadgroups:MTLSizeMake(4, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // Copy back
        memcpy(state, stateBuf.contents, 16 * sizeof(uint32_t));
    }
}

static void dispatch_m31_sbox_gpu(uint32_t *state) {
    ensure_gpu_initialized();
    if (!g_m31_sbox_pipeline) return;

    @autoreleasepool {
        id<MTLBuffer> stateBuf = [g_device newBufferWithBytes:state
                                                     length:16 * sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_m31_sbox_pipeline];
        [enc setBuffer:stateBuf offset:0 atIndex:0];
        // Use 4 threadgroups for single S-box (lower latency)
        [enc dispatchThreadgroups:MTLSizeMake(4, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(state, stateBuf.contents, 16 * sizeof(uint32_t));
    }
}

static void dispatch_bb_batch_gpu(const uint32_t *states, int n, uint32_t *output) {
    ensure_gpu_initialized();
    if (!g_bb_batch_pipeline) return;

    @autoreleasepool {
        int totalSize = n * 16;
        id<MTLBuffer> inputBuf = [g_device newBufferWithBytes:states
                                                      length:totalSize * sizeof(uint32_t)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuf = [g_device newBufferWithLength:totalSize * sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];

        uint32_t nGroups = n;
        id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&nGroups
                                                   length:sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_bb_batch_pipeline];
        [enc setBuffer:inputBuf offset:0 atIndex:0];
        [enc setBuffer:outputBuf offset:0 atIndex:1];
        [enc setBuffer:nBuf offset:0 atIndex:2];
        // Each thread processes 16 elements, use n threads
        [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(output, outputBuf.contents, totalSize * sizeof(uint32_t));
    }
}

static void dispatch_m31_batch_gpu(const uint32_t *states, int n, uint32_t *output) {
    ensure_gpu_initialized();
    if (!g_m31_batch_pipeline) return;

    @autoreleasepool {
        int totalSize = n * 16;
        id<MTLBuffer> inputBuf = [g_device newBufferWithBytes:states
                                                      length:totalSize * sizeof(uint32_t)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuf = [g_device newBufferWithLength:totalSize * sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];

        uint32_t nGroups = n;
        id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&nGroups
                                                   length:sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_m31_batch_pipeline];
        [enc setBuffer:inputBuf offset:0 atIndex:0];
        [enc setBuffer:outputBuf offset:0 atIndex:1];
        [enc setBuffer:nBuf offset:0 atIndex:2];
        // Each thread processes 16 elements, use n threads
        [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(output, outputBuf.contents, totalSize * sizeof(uint32_t));
    }
}

// Single permutation dispatch - processes 1 full permutation in one GPU call
static void dispatch_bb_perm_gpu(uint32_t *state,
                                  const uint32_t *round_constants,
                                  const uint32_t *internal_diag) {
    ensure_gpu_initialized();
    if (!g_bb_perm_pipeline) return;

    @autoreleasepool {
        id<MTLBuffer> stateBuf = [g_device newBufferWithBytes:state
                                                     length:16 * sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> rcBuf = [g_device newBufferWithBytes:round_constants
                                                    length:336 * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> diagBuf = [g_device newBufferWithBytes:internal_diag
                                                     length:16 * sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_bb_perm_pipeline];
        [enc setBuffer:stateBuf offset:0 atIndex:0];
        [enc setBuffer:rcBuf offset:0 atIndex:1];
        [enc setBuffer:diagBuf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(state, stateBuf.contents, 16 * sizeof(uint32_t));
    }
}

// Batched permutation dispatch - processes N full permutations in one GPU call
static void dispatch_bb_perm_batch_gpu(const uint32_t *states, int n_perms,
                                      const uint32_t *round_constants,
                                      const uint32_t *internal_diag,
                                      uint32_t *output) {
    ensure_gpu_initialized();
    if (!g_bb_perm_batch_pipeline) return;

    @autoreleasepool {
        int totalSize = n_perms * 16;
        id<MTLBuffer> inputBuf = [g_device newBufferWithBytes:states
                                                      length:totalSize * sizeof(uint32_t)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuf = [g_device newBufferWithLength:totalSize * sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> rcBuf = [g_device newBufferWithBytes:round_constants
                                                    length:336 * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> diagBuf = [g_device newBufferWithBytes:internal_diag
                                                     length:16 * sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];
        uint32_t nPermsVal = n_perms;
        id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&nPermsVal
                                                   length:sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_bb_perm_batch_pipeline];
        [enc setBuffer:inputBuf offset:0 atIndex:0];
        [enc setBuffer:outputBuf offset:0 atIndex:1];
        [enc setBuffer:rcBuf offset:0 atIndex:2];
        [enc setBuffer:diagBuf offset:0 atIndex:3];
        [enc setBuffer:nBuf offset:0 atIndex:4];
        // Each threadgroup handles one full permutation
        [enc dispatchThreadgroups:MTLSizeMake(n_perms, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(output, outputBuf.contents, totalSize * sizeof(uint32_t));
    }
}

// M31 Poseidon2 chunked permutation using 12-round kernel
// Processes 35 rounds in 3 dispatches: rounds 0-11, 12-23, 24-34
static bool dispatch_m31_perm_chunk_gpu(const uint32_t *states, int n_perms,
                                        const uint32_t *round_constants,
                                        uint32_t *output) {
    ensure_gpu_initialized();
    if (!g_m31_perm_12_pipeline || n_perms <= 0) return false;

    @autoreleasepool {
        int totalSize = n_perms * 16;
        uint32_t *current = (uint32_t*)malloc(totalSize * sizeof(uint32_t));
        uint32_t *next = (uint32_t*)malloc(totalSize * sizeof(uint32_t));
        memcpy(current, states, totalSize * sizeof(uint32_t));

        // 3 dispatches: 12 + 12 + 11 = 35 rounds
        uint round_offsets[3] = {0, 12, 24};

        for (int dispatch = 0; dispatch < 3; dispatch++) {
            id<MTLBuffer> inputBuf = [g_device newBufferWithBytes:current
                                                          length:totalSize * sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
            id<MTLBuffer> outputBuf = [g_device newBufferWithLength:totalSize * sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
            id<MTLBuffer> rcBuf = [g_device newBufferWithBytes:round_constants
                                                        length:560 * sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];
            uint32_t nPermsVal = n_perms;
            id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&nPermsVal
                                                       length:sizeof(uint32_t)
                                                      options:MTLResourceStorageModeShared];
            uint32_t offsetVal = round_offsets[dispatch];
            id<MTLBuffer> offsetBuf = [g_device newBufferWithBytes:&offsetVal
                                                           length:sizeof(uint32_t)
                                                          options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

            [enc setComputePipelineState:g_m31_perm_12_pipeline];
            [enc setBuffer:inputBuf offset:0 atIndex:0];
            [enc setBuffer:outputBuf offset:0 atIndex:1];
            [enc setBuffer:rcBuf offset:0 atIndex:2];
            [enc setBuffer:nBuf offset:0 atIndex:3];
            [enc setBuffer:offsetBuf offset:0 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(n_perms, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];

            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];

            // Swap buffers
            uint32_t *temp = current;
            current = (uint32_t*)outputBuf.contents;
            next = temp;
        }

        memcpy(output, current, totalSize * sizeof(uint32_t));
        free(next);  // 'next' now points to the original malloc'd buffer
        return true;
    }
}

// ============================================================
// Public API Implementations
// ============================================================

extern "C" {

void bb_poseidon2_sbox_ane(uint32_t state[16]) {
    // Ensure GPU is initialized first
    ensure_gpu_initialized();

    // Try GPU first, fall back to scalar
    if (g_bb_sbox_pipeline) {
        dispatch_bb_sbox_gpu(state);
    } else {
        for (int i = 0; i < 16; i++) {
            state[i] = bb_sbox_scalar(state[i]);
        }
    }
}

void m31_poseidon2_sbox_ane(uint32_t state[16]) {
    ensure_gpu_initialized();

    if (g_m31_sbox_pipeline) {
        dispatch_m31_sbox_gpu(state);
    } else {
        for (int i = 0; i < 16; i++) {
            state[i] = m31_sbox_scalar(state[i]);
        }
    }
}

void bb_poseidon2_permutation_ane(uint32_t state[16],
                                  const uint32_t *round_constants,
                                  const uint32_t internal_diag[16]) {
    ensure_gpu_initialized();

    if (g_bb_perm_pipeline) {
        dispatch_bb_perm_gpu(state, round_constants, internal_diag);
    } else {
        // Scalar fallback
        int rc_idx = 0;
        for (int r = 0; r < 4; r++) {
            bb_full_round(state, round_constants + rc_idx);
            rc_idx += 16;
        }
        for (int r = 0; r < 13; r++) {
            bb_partial_round(state, round_constants[rc_idx]);
            rc_idx += 1;
        }
        for (int r = 0; r < 4; r++) {
            bb_full_round(state, round_constants + rc_idx);
            rc_idx += 16;
        }
    }
}

void m31_poseidon2_permutation_ane(uint32_t state[16],
                                  const uint32_t *round_constants,
                                  const uint32_t internal_diag[16]) {
    (void)internal_diag;
    // Initial external linear layer (required before first round, matches CPU)
    m31_external_layer(state);

    int rc_idx = 0;
    for (int r = 0; r < 7; r++) {
        m31_full_round(state, round_constants + rc_idx);
        rc_idx += 16;
    }
    // Partial rounds (7..27) - use stride-16 indexing to match Metal shader
    for (int r = 7; r < 28; r++) {
        m31_partial_round(state, round_constants[r * 16]);
    }
    // Second half of full rounds (28..34)
    for (int r = 28; r < 35; r++) {
        m31_full_round(state, round_constants + r * 16);
    }
}

void bb_poseidon2_sbox_batch_ane(const uint32_t *states, int n, uint32_t *output) {
    if (g_bb_batch_pipeline && n > 0) {
        dispatch_bb_batch_gpu(states, n, output);
    } else {
        for (int i = 0; i < n; i++) {
            uint32_t s[16];
            memcpy(s, states + i * 16, 16 * sizeof(uint32_t));
            bb_poseidon2_sbox_ane(s);
            memcpy(output + i * 16, s, 16 * sizeof(uint32_t));
        }
    }
}

void m31_poseidon2_sbox_batch_ane(const uint32_t *states, int n, uint32_t *output) {
    if (g_m31_batch_pipeline && n > 0) {
        dispatch_m31_batch_gpu(states, n, output);
    } else {
        for (int i = 0; i < n; i++) {
            uint32_t s[16];
            memcpy(s, states + i * 16, 16 * sizeof(uint32_t));
            m31_poseidon2_sbox_ane(s);
            memcpy(output + i * 16, s, 16 * sizeof(uint32_t));
        }
    }
}

// Batched full permutation - processes N Poseidon2 permutations in one GPU call
// States: n_perms * 16 input elements
// round_constants: 21 * 16 = 336 BabyBear elements
// internal_diag: 16 BabyBear elements
// output: n_perms * 16 result elements
void bb_poseidon2_permutation_batch_ane(const uint32_t *states, int n_perms,
                                         const uint32_t *round_constants,
                                         const uint32_t *internal_diag,
                                         uint32_t *output) {
    ensure_gpu_initialized();

    if (g_bb_perm_batch_pipeline && n_perms > 0) {
        dispatch_bb_perm_batch_gpu(states, n_perms, round_constants, internal_diag, output);
    } else {
        // Scalar fallback - process one at a time
        for (int i = 0; i < n_perms; i++) {
            uint32_t s[16];
            memcpy(s, states + i * 16, 16 * sizeof(uint32_t));
            bb_poseidon2_permutation_ane(s, round_constants, internal_diag);
            memcpy(output + i * 16, s, 16 * sizeof(uint32_t));
        }
    }
}

// M31 batched full permutation with GPU chunked fallback
void m31_poseidon2_permutation_batch_ane(const uint32_t *states, int n_perms,
                                          const uint32_t *round_constants,
                                          uint32_t *output) {
    ensure_gpu_initialized();

    // Try full 35-round GPU kernel first
    if (n_perms > 0 && dispatch_m31_perm_batch_gpu(states, n_perms, round_constants, output)) {
        return;  // GPU full kernel succeeded
    }

    // Try chunked 12-round GPU kernel (3 dispatches)
    if (n_perms > 0 && dispatch_m31_perm_chunk_gpu(states, n_perms, round_constants, output)) {
        return;  // GPU chunked kernel succeeded
    }

    // CPU scalar fallback
    if (n_perms > 0) {
        dispatch_apply(n_perms, DISPATCH_APPLY_AUTO, ^(size_t i) {
            uint32_t s[16];
            memcpy(s, states + i * 16, 16 * sizeof(uint32_t));
            m31_poseidon2_permutation_ane(s, round_constants, NULL);
            memcpy(output + i * 16, s, 16 * sizeof(uint32_t));
        });
        return;
    }
}

} // extern "C"
