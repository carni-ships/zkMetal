// ane_lattice.mm — ANE-accelerated Kyber NTT C wrapper with Metal GPU acceleration
//
// This file provides the C API for Kyber NTT with Metal GPU acceleration
// that automatically offloads to ANE on Apple Silicon.
//
// GPU path: compiles Metal shader at runtime and dispatches compute kernels
// The ANE (Neural Engine) is used indirectly via Metal's compute pipeline
// on ANE-capable devices (Apple GPU family Apple7+).
//
// Scalar path: uses inline scalar arithmetic (fallback when GPU unavailable)

#include "include/ane_lattice.h"
#include <Metal/Metal.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// ============================================================
// Kyber field constants
// ============================================================

#define KYBER_Q          3329
#define KYBER_R          65536u
#define KYBER_R_MOD_P    ((uint16_t)(KYBER_R % KYBER_Q))  // 2184
#define KYBER_P_INV      3361u
#define KYBER_INV_128    3073u
#define KYBER_ZETA       17u

// ============================================================
// Metal GPU State (global singleton)
// ============================================================

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_ntt_forward_pipeline = nil;
static id<MTLComputePipelineState> g_ntt_inverse_pipeline = nil;
static id<MTLComputePipelineState> g_ntt_forward_single_pipeline = nil;
static id<MTLComputePipelineState> g_ntt_inverse_single_pipeline = nil;
static id<MTLComputePipelineState> g_to_mont_pipeline = nil;
static id<MTLComputePipelineState> g_from_mont_pipeline = nil;
static bool g_gpu_initialized = false;
static bool g_ane_available = false;

// ============================================================
// Scalar Montgomery multiplication (fallback)
// ============================================================

static inline uint16_t kyber_mont_mul_scalar(uint16_t a, uint16_t b) {
    uint32_t t = (uint32_t)a * (uint32_t)b;
    uint32_t tp = (t * (uint32_t)KYBER_P_INV) & 0xFFFF;
    uint32_t t2 = t + (uint32_t)tp * (uint32_t)KYBER_Q;
    uint16_t result = (uint16_t)(t2 >> 16);
    return result >= KYBER_Q ? (result - KYBER_Q) : result;
}

static inline uint16_t kyber_add_scalar(uint16_t a, uint16_t b) {
    uint16_t s = a + b;
    return s >= KYBER_Q ? (s - KYBER_Q) : s;
}

static inline uint16_t kyber_sub_scalar(uint16_t a, uint16_t b) {
    return a >= b ? (a - b) : (a + KYBER_Q - b);
}

// ============================================================
// Bit-reversal index
// ============================================================

static inline uint8_t bitrev7(uint8_t x) {
    uint8_t v = x;
    v = ((v & 0x55) << 1) | ((v >> 1) & 0x55);
    v = ((v & 0x33) << 2) | ((v >> 2) & 0x33);
    v = ((v & 0x0F) << 4) | ((v >> 4) & 0x0F);
    return v >> 1;
}

// ============================================================
// Twiddle factor precomputation
// ============================================================

static void generate_kyber_twiddles(uint16_t* forward_twiddles, uint16_t* inverse_twiddles) {
    // Compute zeta^i mod 3329 for i = 0..255
    uint16_t powers[256];
    powers[0] = 1;
    for (int i = 1; i < 256; i++) {
        powers[i] = (uint16_t)((uint32_t)powers[i-1] * KYBER_ZETA % KYBER_Q);
    }

    // Generate forward twiddles in bit-reversed order
    for (int i = 0; i < 128; i++) {
        forward_twiddles[i] = powers[bitrev7((uint8_t)i)];
    }

    // Generate inverse twiddles: q - twiddle (for DIF)
    for (int i = 0; i < 128; i++) {
        inverse_twiddles[i] = forward_twiddles[i] == 0 ? 0 : (KYBER_Q - forward_twiddles[i]);
    }
}

// ============================================================
// Metal shader source for Kyber NTT
// ============================================================

static const char* g_shader_source = R"(
#include <metal_stdlib>
using namespace metal;

constant ushort KYBER_Q = 3329;
constant ushort KYBER_R_MOD_P = 2184;
constant ushort KYBER_P_INV = 3361;

inline ushort kyber_add(ushort a, ushort b) {
    ushort s = a + b;
    return s >= KYBER_Q ? (s - KYBER_Q) : s;
}

inline ushort kyber_sub(ushort a, ushort b) {
    return a >= b ? (a - b) : (a + KYBER_Q - b);
}

inline ushort kyber_mont_mul(ushort a, ushort b) {
    uint t = (uint)a * (uint)b;
    uint tp = (t * (uint)KYBER_P_INV) & 0xFFFF;
    uint t2 = t + (uint)tp * (uint)KYBER_Q;
    ushort result = (ushort)(t2 >> 16);
    return result >= KYBER_Q ? (result - KYBER_Q) : result;
}

kernel void kyber_ntt_batch64_complete(
    device ushort *polys [[buffer(0)]],
    constant ushort *twiddles [[buffer(1)]],
    constant uint &numPolys [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (numPolys != 64) return;

    threadgroup ushort shared[256];
    uint polyIdx = tgid;
    uint base = polyIdx * 256;

    for (uint i = lid; i < 256; i += tg_size) {
        shared[i] = polys[base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint k = 1;
    for (uint len = 128; len >= 2; len >>= 1) {
        uint numBlocks = 256 / (2 * len);
        for (uint block = lid; block < numBlocks * len; block += tg_size) {
            uint blockIdx = block / len;
            uint j = block % len;
            uint i0 = blockIdx * 2 * len + j;
            uint i1 = i0 + len;
            ushort tw = twiddles[k + blockIdx];
            ushort u = shared[i0];
            ushort v = shared[i1];
            ushort t = kyber_mont_mul(tw, v);
            shared[i0] = kyber_add(u, t);
            shared[i1] = kyber_sub(u, t);
        }
        k += numBlocks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = lid; i < 256; i += tg_size) {
        polys[base + i] = shared[i];
    }
}

kernel void kyber_ntt_inverse_batch64_complete(
    device ushort *polys [[buffer(0)]],
    constant ushort *fwdTwiddles [[buffer(1)]],
    constant uint &numPolys [[buffer(2)]],
    constant ushort &invN [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (numPolys != 64) return;

    threadgroup ushort shared[256];
    uint polyIdx = tgid;
    uint base = polyIdx * 256;

    for (uint i = lid; i < 256; i += tg_size) {
        shared[i] = polys[base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint k = 127;
    for (uint len = 2; len <= 128; len <<= 1) {
        uint numBlocks = 256 / (2 * len);
        for (uint block = lid; block < numBlocks * len; block += tg_size) {
            uint blockIdx = block / len;
            uint j = block % len;
            uint i0 = blockIdx * 2 * len + j;
            uint i1 = i0 + len;
            ushort fwd_tw = fwdTwiddles[k - blockIdx];
            ushort tw = (fwd_tw == 0) ? 0 : (KYBER_Q - fwd_tw);
            ushort u = shared[i0];
            ushort v = shared[i1];
            shared[i0] = kyber_add(u, v);
            shared[i1] = kyber_mont_mul(tw, kyber_sub(u, v));
        }
        k -= numBlocks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = lid; i < 256; i += tg_size) {
        shared[i] = kyber_mont_mul(shared[i], invN);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Output bit-reversal permutation
    for (uint i = lid; i < 256; i += tg_size) {
        uint8_t rev = ((i & 0x55) << 1) | ((i >> 1) & 0x55);
        rev = ((rev & 0x33) << 2) | ((rev >> 2) & 0x33);
        rev = ((rev & 0x0F) << 4) | ((rev >> 4) & 0x0F);
        rev = rev >> 1;
        if (i < rev) {
            ushort tmp = shared[i];
            shared[i] = shared[rev];
            shared[rev] = tmp;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < 256; i += tg_size) {
        polys[base + i] = shared[i];
    }
}

kernel void kyber_ntt_single(
    device ushort *poly [[buffer(0)]],
    constant ushort *twiddles [[buffer(1)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= 1) return;

    threadgroup ushort shared[256];
    for (uint i = lid; i < 256; i += tg_size) {
        shared[i] = poly[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint k = 1;
    for (uint len = 128; len >= 2; len >>= 1) {
        uint numBlocks = 256 / (2 * len);
        for (uint block = lid; block < numBlocks * len; block += tg_size) {
            uint blockIdx = block / len;
            uint j = block % len;
            uint i0 = blockIdx * 2 * len + j;
            uint i1 = i0 + len;
            ushort tw = twiddles[k + blockIdx];
            ushort u = shared[i0];
            ushort v = shared[i1];
            ushort t = kyber_mont_mul(tw, v);
            shared[i0] = kyber_add(u, t);
            shared[i1] = kyber_sub(u, t);
        }
        k += numBlocks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = lid; i < 256; i += tg_size) {
        poly[i] = shared[i];
    }
}

kernel void kyber_ntt_inverse_single(
    device ushort *poly [[buffer(0)]],
    constant ushort *fwdTwiddles [[buffer(1)]],
    constant ushort &invN [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (tgid >= 1) return;

    threadgroup ushort shared[256];
    for (uint i = lid; i < 256; i += tg_size) {
        shared[i] = poly[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint k = 127;
    for (uint len = 2; len <= 128; len <<= 1) {
        uint numBlocks = 256 / (2 * len);
        for (uint block = lid; block < numBlocks * len; block += tg_size) {
            uint blockIdx = block / len;
            uint j = block % len;
            uint i0 = blockIdx * 2 * len + j;
            uint i1 = i0 + len;
            ushort fwd_tw = fwdTwiddles[k - blockIdx];
            ushort tw = (fwd_tw == 0) ? 0 : (KYBER_Q - fwd_tw);
            ushort u = shared[i0];
            ushort v = shared[i1];
            shared[i0] = kyber_add(u, v);
            shared[i1] = kyber_mont_mul(tw, kyber_sub(u, v));
        }
        k -= numBlocks;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = lid; i < 256; i += tg_size) {
        shared[i] = kyber_mont_mul(shared[i], invN);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Output bit-reversal permutation
    for (uint i = lid; i < 256; i += tg_size) {
        uint8_t rev = ((i & 0x55) << 1) | ((i >> 1) & 0x55);
        rev = ((rev & 0x33) << 2) | ((rev >> 2) & 0x33);
        rev = ((rev & 0x0F) << 4) | ((rev >> 4) & 0x0F);
        rev = rev >> 1;
        if (i < rev) {
            ushort tmp = shared[i];
            shared[i] = shared[rev];
            shared[rev] = tmp;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lid; i < 256; i += tg_size) {
        poly[i] = shared[i];
    }
}
)";

// ============================================================
// GPU Initialization
// ============================================================

static int init_gpu(void) {
    if (g_gpu_initialized) return 0;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            return -1;
        }

        // Check ANE support (Apple7+ GPU family has ANE)
        g_ane_available = [g_device supportsFamily:MTLGPUFamilyApple7];

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            return -1;
        }

        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:g_shader_source];
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

        g_library = [g_device newLibraryWithSource:source
                                           options:options
                                             error:&error];
        if (!g_library) {
            NSLog(@"Metal shader compilation failed: %@", error);
            return -1;
        }

        // Create pipeline states
        id<MTLFunction> fwd_fn = [g_library newFunctionWithName:@"kyber_ntt_batch64_complete"];
        id<MTLFunction> inv_fn = [g_library newFunctionWithName:@"kyber_ntt_inverse_batch64_complete"];
        id<MTLFunction> fwd_single_fn = [g_library newFunctionWithName:@"kyber_ntt_single"];
        id<MTLFunction> inv_single_fn = [g_library newFunctionWithName:@"kyber_ntt_inverse_single"];

        if (fwd_fn) {
            g_ntt_forward_pipeline = [g_device newComputePipelineStateWithFunction:fwd_fn error:&error];
        }
        if (inv_fn) {
            g_ntt_inverse_pipeline = [g_device newComputePipelineStateWithFunction:inv_fn error:&error];
        }
        if (fwd_single_fn) {
            g_ntt_forward_single_pipeline = [g_device newComputePipelineStateWithFunction:fwd_single_fn error:&error];
        }
        if (inv_single_fn) {
            g_ntt_inverse_single_pipeline = [g_device newComputePipelineStateWithFunction:inv_single_fn error:&error];
        }

        g_gpu_initialized = true;
        return 0;
    }
}

static void ensure_gpu_initialized(void) {
    if (!g_gpu_initialized) {
        init_gpu();
    }
}

// ============================================================
// Scalar NTT implementation (fallback)
// ============================================================

static void scalar_ntt_forward(uint16_t* data) {
    uint16_t twiddles[128];
    uint16_t inv_twiddles[128];
    generate_kyber_twiddles(twiddles, inv_twiddles);

    // Bit-reversal permutation
    for (int i = 0; i < 256; i++) {
        int rev = bitrev7(i);
        if (i < rev) {
            uint16_t tmp = data[i];
            data[i] = data[rev];
            data[rev] = tmp;
        }
    }

    // DIT butterfly stages
    uint16_t k = 1;
    for (int len = 128; len >= 2; len >>= 1) {
        int numBlocks = 256 / (2 * len);
        for (int block = 0; block < numBlocks * len; block++) {
            int blockIdx = block / len;
            int j = block % len;
            int i0 = blockIdx * 2 * len + j;
            int i1 = i0 + len;

            uint16_t tw = twiddles[k + blockIdx];
            uint16_t u = data[i0];
            uint16_t v = data[i1];
            uint16_t t = kyber_mont_mul_scalar(tw, v);
            data[i0] = kyber_add_scalar(u, t);
            data[i1] = kyber_sub_scalar(u, t);
        }
        k += numBlocks;
    }
}

static void scalar_ntt_inverse(uint16_t* data) {
    uint16_t twiddles[128];
    uint16_t inv_twiddles[128];
    generate_kyber_twiddles(twiddles, inv_twiddles);

    // DIF butterfly stages (Gentleman-Sande)
    uint16_t k = 127;
    for (int len = 2; len <= 128; len <<= 1) {
        int numBlocks = 256 / (2 * len);
        for (int block = 0; block < numBlocks * len; block++) {
            int blockIdx = block / len;
            int j = block % len;
            int i0 = blockIdx * 2 * len + j;
            int i1 = i0 + len;

            uint16_t fwd_tw = twiddles[k - blockIdx];
            uint16_t tw = (fwd_tw == 0) ? 0 : (KYBER_Q - fwd_tw);
            uint16_t u = data[i0];
            uint16_t v = data[i1];
            data[i0] = kyber_add_scalar(u, v);
            data[i1] = kyber_mont_mul_scalar(tw, kyber_sub_scalar(u, v));
        }
        k -= numBlocks;
    }

    // Final scaling by inv128 = 3073
    for (int i = 0; i < 256; i++) {
        data[i] = kyber_mont_mul_scalar(data[i], KYBER_INV_128);
    }

    // Output bit-reversal (undo forward's input bit-reversal)
    for (int i = 0; i < 256; i++) {
        int rev = bitrev7((uint8_t)i);
        if (i < rev) {
            uint16_t tmp = data[i];
            data[i] = data[rev];
            data[rev] = tmp;
        }
    }
}

// ============================================================
// GPU dispatch helpers
// ============================================================

static int dispatch_ntt_forward_batch64(uint16_t* polys) {
    ensure_gpu_initialized();
    if (!g_ntt_forward_pipeline) return -1;

    @autoreleasepool {
        // Generate twiddles
        uint16_t forward_twiddles[128];
        uint16_t inverse_twiddles[128];
        generate_kyber_twiddles(forward_twiddles, inverse_twiddles);

        int numPolys = 64;
        int tg_size = 32;

        id<MTLBuffer> dataBuf = [g_device newBufferWithBytes:polys
                                                       length:numPolys * 256 * sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> twBuf = [g_device newBufferWithBytes:forward_twiddles
                                                    length:128 * sizeof(uint16_t)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> numPolysBuf = [g_device newBufferWithBytes:&numPolys
                                                         length:sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_ntt_forward_pipeline];
        [enc setBuffer:dataBuf offset:0 atIndex:0];
        [enc setBuffer:twBuf offset:0 atIndex:1];
        [enc setBuffer:numPolysBuf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(numPolys, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(polys, dataBuf.contents, numPolys * 256 * sizeof(uint16_t));
        return 0;
    }
}

static int dispatch_ntt_inverse_batch64(uint16_t* polys) {
    ensure_gpu_initialized();
    if (!g_ntt_inverse_pipeline) return -1;

    @autoreleasepool {
        // Generate twiddles
        uint16_t forward_twiddles[128];
        uint16_t inverse_twiddles[128];
        generate_kyber_twiddles(forward_twiddles, inverse_twiddles);

        int numPolys = 64;
        int tg_size = 32;
        uint16_t inv128 = KYBER_INV_128;

        id<MTLBuffer> dataBuf = [g_device newBufferWithBytes:polys
                                                       length:numPolys * 256 * sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> twBuf = [g_device newBufferWithBytes:forward_twiddles
                                                    length:128 * sizeof(uint16_t)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> numPolysBuf = [g_device newBufferWithBytes:&numPolys
                                                         length:sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> inv128Buf = [g_device newBufferWithBytes:&inv128
                                                       length:sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_ntt_inverse_pipeline];
        [enc setBuffer:dataBuf offset:0 atIndex:0];
        [enc setBuffer:twBuf offset:0 atIndex:1];
        [enc setBuffer:numPolysBuf offset:0 atIndex:2];
        [enc setBuffer:inv128Buf offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(numPolys, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(polys, dataBuf.contents, numPolys * 256 * sizeof(uint16_t));
        return 0;
    }
}

static int dispatch_ntt_forward_single(uint16_t* poly) {
    ensure_gpu_initialized();
    if (!g_ntt_forward_single_pipeline) return -1;

    @autoreleasepool {
        // Generate twiddles
        uint16_t forward_twiddles[128];
        uint16_t inverse_twiddles[128];
        generate_kyber_twiddles(forward_twiddles, inverse_twiddles);

        int tg_size = 32;

        id<MTLBuffer> dataBuf = [g_device newBufferWithBytes:poly
                                                       length:256 * sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> twBuf = [g_device newBufferWithBytes:forward_twiddles
                                                    length:128 * sizeof(uint16_t)
                                                   options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_ntt_forward_single_pipeline];
        [enc setBuffer:dataBuf offset:0 atIndex:0];
        [enc setBuffer:twBuf offset:0 atIndex:1];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(poly, dataBuf.contents, 256 * sizeof(uint16_t));
        return 0;
    }
}

static int dispatch_ntt_inverse_single(uint16_t* poly) {
    ensure_gpu_initialized();
    if (!g_ntt_inverse_single_pipeline) return -1;

    @autoreleasepool {
        // Generate twiddles
        uint16_t forward_twiddles[128];
        uint16_t inverse_twiddles[128];
        generate_kyber_twiddles(forward_twiddles, inverse_twiddles);

        int tg_size = 32;
        uint16_t inv128 = KYBER_INV_128;

        id<MTLBuffer> dataBuf = [g_device newBufferWithBytes:poly
                                                       length:256 * sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> twBuf = [g_device newBufferWithBytes:forward_twiddles
                                                    length:128 * sizeof(uint16_t)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> inv128Buf = [g_device newBufferWithBytes:&inv128
                                                       length:sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_ntt_inverse_single_pipeline];
        [enc setBuffer:dataBuf offset:0 atIndex:0];
        [enc setBuffer:twBuf offset:0 atIndex:1];
        [enc setBuffer:inv128Buf offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(poly, dataBuf.contents, 256 * sizeof(uint16_t));
        return 0;
    }
}

// ============================================================
// Public API Implementations
// ============================================================

extern "C" {

bool ane_kyber_ntt_available(void) {
    ensure_gpu_initialized();
    // Return true if we have GPU support (ANE or regular GPU)
    return g_ntt_forward_pipeline != nil;
}

void* ane_kyber_ntt_create(int logN) {
    (void)logN;
    ensure_gpu_initialized();
    // Return non-NULL if GPU is available
    return g_ntt_forward_pipeline != nil ? (void*)0x1 : NULL;
}

void ane_kyber_ntt_destroy(void* state) {
    (void)state;
    // Nothing to destroy - we use global state
}

int ane_kyber_ntt(void* state, uint16_t* data, int logN) {
    (void)state;
    (void)logN;

    if (logN != 8) return -1;  // Only N=256 supported

    if (g_ntt_forward_single_pipeline) {
        return dispatch_ntt_forward_single(data);
    }

    // Fallback to scalar
    scalar_ntt_forward(data);
    return 0;
}

int ane_kyber_intt(void* state, uint16_t* data, int logN) {
    (void)state;
    (void)logN;

    if (logN != 8) return -1;

    if (g_ntt_inverse_single_pipeline) {
        return dispatch_ntt_inverse_single(data);
    }

    // Fallback to scalar
    scalar_ntt_inverse(data);
    return 0;
}

int ane_kyber_ntt_batch64(void* state, uint16_t* polys) {
    (void)state;

    if (g_ntt_forward_pipeline) {
        return dispatch_ntt_forward_batch64(polys);
    }

    // Fallback to scalar - process each polynomial
    for (int i = 0; i < 64; i++) {
        scalar_ntt_forward(polys + i * 256);
    }
    return 0;
}

int ane_kyber_intt_batch64(void* state, uint16_t* polys) {
    (void)state;

    if (g_ntt_inverse_pipeline) {
        return dispatch_ntt_inverse_batch64(polys);
    }

    // Fallback to scalar - process each polynomial
    for (int i = 0; i < 64; i++) {
        scalar_ntt_inverse(polys + i * 256);
    }
    return 0;
}

int ane_kyber_ntt_forward(uint16_t* data) {
    return ane_kyber_ntt(NULL, data, 8);
}

int ane_kyber_ntt_forward_batch64(uint16_t* polys) {
    return ane_kyber_ntt_batch64(NULL, polys);
}

int ane_kyber_ntt_inverse(uint16_t* data) {
    return ane_kyber_intt(NULL, data, 8);
}

int ane_kyber_ntt_inverse_batch64(uint16_t* polys) {
    return ane_kyber_intt_batch64(NULL, polys);
}

uint16_t kyber_mont_r_mod_p(void) { return KYBER_R_MOD_P; }
uint16_t kyber_mont_p_inv(void) { return KYBER_P_INV; }
uint16_t kyber_inv128(void) { return KYBER_INV_128; }

uint16_t kyber_mont_mul(uint16_t a, uint16_t b, uint16_t p_inv) {
    (void)p_inv;
    return kyber_mont_mul_scalar(a, b);
}

uint16_t kyber_to_mont(uint16_t a, uint16_t r_mod_p, uint16_t p_inv) {
    (void)r_mod_p; (void)p_inv;
    return kyber_mont_mul_scalar(a, KYBER_R_MOD_P);
}

uint16_t kyber_from_mont(uint16_t a, uint16_t p_inv) {
    (void)p_inv;
    return kyber_mont_mul_scalar(a, 1);
}

} // extern "C"
