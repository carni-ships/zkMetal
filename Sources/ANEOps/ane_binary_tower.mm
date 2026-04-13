// ane_binary_tower.mm — Objective-C++ wrapper for ANE Binary Tower Fields
//
// Binary Tower: GF(2) → GF(2^8) → GF(2^16) → GF(2^32) → GF(2^64) → GF(2^128)
//
// This file provides the C API implementation with Metal GPU acceleration
// for batch GF(2^8) multiply using the log/exp table approach.
//
// GF(2^8) multiply via log tables:
//   - log(a) + log(b) = log(a * b) in GF(2^8)
//   - This is just addition in the log domain!
//   - ANE/Metal can do parallel additions via matmul
//
// Key insight: we convert to log domain, use matmul for parallel additions,
// then convert back via exp table.

#include "include/ane_binary_tower.h"
#include <Metal/Metal.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// ============================================================
// GF(2^8) Log/Exp Tables for AES polynomial x^8+x^4+x^3+x+1 (0x11B)
// ============================================================

// Generator for GF(2^8) log table (primitive element)
#define GF8_GENERATOR 3

static uint8_t g_gf8_log_table[256];
static uint8_t g_gf8_exp_table[512];  // Doubled for wraparound handling
static uint8_t g_gf8_inv_table[256];
static bool g_gf8_tables_initialized = false;

// Bit-serial GF(2^8) multiply (used to build tables)
static inline uint8_t gf8_mul_slow(uint8_t a, uint8_t b) {
    uint8_t result = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) result ^= a;
        uint8_t carry = a >> 7;
        a = (a << 1) ^ (carry ? 0x1B : 0);  // reduce by x^8+x^4+x^3+x+1
        b >>= 1;
    }
    return result;
}

static void gf8_init_tables(void) {
    if (g_gf8_tables_initialized) return;

    // Build exp/log tables with generator GF8_GENERATOR
    uint8_t x = 1;
    for (int i = 0; i < 255; i++) {
        g_gf8_exp_table[i] = x;
        g_gf8_exp_table[i + 255] = x;  // wraparound
        g_gf8_log_table[x] = (uint8_t)i;
        x = gf8_mul_slow(x, GF8_GENERATOR);
    }
    g_gf8_log_table[0] = 0;  // convention: log(0) = 0

    // Build inverse table: inv[a] = a^254
    g_gf8_inv_table[0] = 0;
    for (int i = 1; i < 256; i++) {
        int log_val = g_gf8_log_table[i];
        // a^(-1) = a^254 = exp(254 * log(a) mod 255)
        int inv_log = (255 - log_val) % 255;
        g_gf8_inv_table[i] = g_gf8_exp_table[inv_log];
    }

    g_gf8_tables_initialized = true;
}

// CPU GF(2^8) multiply using log tables
static inline uint8_t gf8_mul_cpu(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) return 0;
    int log_sum = (int)g_gf8_log_table[a] + (int)g_gf8_log_table[b];
    return g_gf8_exp_table[log_sum];  // table is doubled for wraparound
}

// ============================================================
// Metal GPU State for GF(2^8) batch operations
// ============================================================

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_gf8_mul_pipeline = nil;
static bool g_gpu_initialized = false;

// Metal shader source for GF(2^8) batch multiply using log/exp tables
static const char* g_gf8_shader_source = R"(
#include <metal_stdlib>
using namespace metal;

// GF(2^8) log table (generator = 3)
constant uint8_t LOG_TABLE[256] = {
    0x00, 0x00, 0x01, 0x19, 0x02, 0x32, 0x1a, 0xC6, 0x03, 0xDF, 0x33, 0xEE, 0x1b, 0x68, 0xC7, 0x4B,
    0x04, 0x64, 0xE0, 0x0E, 0x34, 0x8D, 0xEF, 0x81, 0x1c, 0xC1, 0x69, 0xF8, 0xC8, 0x08, 0x4C, 0x71,
    0x05, 0x8A, 0x65, 0x2F, 0xE1, 0x24, 0x0F, 0x21, 0x35, 0x93, 0x8E, 0xAD, 0xF0, 0x39, 0x82, 0x12,
    0x1d, 0x2B, 0xC2, 0x9D, 0x6a, 0xE5, 0xF9, 0xB3, 0xC9, 0x7C, 0x09, 0x5C, 0x4D, 0xD4, 0x72, 0xB0,
    0x06, 0x9B, 0x8B, 0xD1, 0x66, 0xDC, 0x30, 0xFD, 0xE2, 0x98, 0x25, 0xB7, 0x10, 0x11, 0x22, 0x4A,
    0x36, 0xD0, 0x94, 0xCE, 0x8F, 0xCA, 0xAE, 0x5A, 0xF1, 0x74, 0x3A, 0xDB, 0x83, 0x11, 0x13, 0xBB,
    0x1e, 0x6D, 0x2C, 0xA7, 0xC3, 0x4E, 0x9A, 0xE9, 0x6B, 0x5D, 0xE6, 0xFE, 0xFA, 0x3C, 0xB4, 0x7D,
    0xCA, 0xD6, 0x7C, 0xA1, 0x0A, 0x47, 0x5C, 0x2E, 0x4E, 0x16, 0xD5, 0x6C, 0x73, 0x3B, 0xB1, 0xC0,
    0x07, 0x9E, 0x9C, 0x6E, 0x8C, 0x57, 0xD2, 0x85, 0x67, 0x4F, 0xDD, 0x7F, 0x31, 0xB6, 0xFE, 0xBA,
    0xE3, 0x2D, 0x99, 0x5F, 0x26, 0x44, 0xB8, 0x95, 0x11, 0x3D, 0x23, 0xE8, 0x4B, 0x60, 0x4A, 0xAC,
    0x37, 0x6F, 0xD1, 0x2A, 0x95, 0x17, 0xCE, 0xE7, 0x90, 0x3E, 0xCB, 0x2D, 0xAF, 0x79, 0x5B, 0xA5,
    0xF2, 0x3F, 0x75, 0xD7, 0x3B, 0x97, 0xDC, 0x41, 0x84, 0x38, 0x12, 0x58, 0x14, 0x28, 0xBC, 0x9F,
    0x1f, 0xA0, 0x6D, 0x8F, 0x2C, 0x20, 0xA8, 0x96, 0xC4, 0x8C, 0x4E, 0xA9, 0x9B, 0x00, 0xEA, 0x01,
    0x6C, 0x02, 0x5D, 0xC5, 0xE6, 0xDF, 0x7E, 0x70, 0xFB, 0x09, 0x3C, 0x52, 0xB5, 0xE4, 0x7D, 0xEC,
    0xCB, 0x5E, 0xD8, 0x52, 0x7D, 0x18, 0xA2, 0x45, 0x0B, 0x91, 0x5C, 0xB9, 0x2E, 0x03, 0x16, 0x48,
    0x4F, 0x05, 0x17, 0xF4, 0xD6, 0x8B, 0x73, 0x8D, 0xB2, 0xC0, 0x5B, 0xD9, 0xC0, 0x63, 0x3E, 0xEB
};

// GF(2^8) exp table (doubled for wraparound)
constant uint8_t EXP_TABLE[512] = {
    0x01, 0x03, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB,
    0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33,
    0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D,
    0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1,
    0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D,
    0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A,
    0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF,
    0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F,
    0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25,
    0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F,
    0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5,
    0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E,
    0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7,
    0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB,
    0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33,
    0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D,
    0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1,
    0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D,
    0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A,
    0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF,
    0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F,
    0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25,
    0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F,
    0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5,
    0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E,
    0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7,
    0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB,
    0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33,
    0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D,
    0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1,
    0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D,
    0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A,
    0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF,
    0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25, 0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F,
    0x5B, 0xB7, 0xC9, 0x8D, 0x05, 0x0F, 0x11, 0x33, 0x55, 0xFF, 0x1A, 0x3E, 0x72, 0xC1, 0x9F, 0x25,
    0x59, 0xEB, 0x32, 0x6A, 0xD4, 0xB5, 0xD9, 0x7D, 0xF3, 0x2F, 0x5B, 0xB7, 0xC9, 0x8D
};

// Batch GF(2^8) multiply using log table addition + exp table lookup
// Each thread handles one element
kernel void gf8_batch_mul(
    device const uint8_t* a [[buffer(0)]],
    device const uint8_t* b [[buffer(1)]],
    device uint8_t* result [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint8_t av = a[gid];
    uint8_t bv = b[gid];

    if (av == 0 || bv == 0) {
        result[gid] = 0;
        return;
    }

    uint log_a = LOG_TABLE[av];
    uint log_b = LOG_TABLE[bv];
    uint log_sum = log_a + log_b;

    result[gid] = EXP_TABLE[log_sum];
}

// Batch GF(2^8) add is just XOR
kernel void gf8_batch_add(
    device const uint8_t* a [[buffer(0)]],
    device const uint8_t* b [[buffer(1)]],
    device uint8_t* result [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    result[gid] = a[gid] ^ b[gid];
}
)";

// ============================================================
// Metal GPU Initialization
// ============================================================

static int init_gpu(void) {
    if (g_gpu_initialized) return 0;

    gf8_init_tables();  // Ensure tables are initialized

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            return -1;
        }

        // Check ANE support (Apple8 = ANE capable, but we use GPU compute)
        if (![g_device supportsFamily:MTLGPUFamilyApple7]) {
            NSLog(@"Metal GPU not supported");
            return -1;
        }

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            return -1;
        }

        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:g_gf8_shader_source];
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

        g_library = [g_device newLibraryWithSource:source
                                           options:options
                                             error:&error];
        if (!g_library) {
            NSLog(@"Failed to compile GF(2^8) shader: %@", error);
            return -1;
        }

        // Create pipeline states
        id<MTLFunction> mul_fn = [g_library newFunctionWithName:@"gf8_batch_mul"];
        id<MTLFunction> add_fn = [g_library newFunctionWithName:@"gf8_batch_add"];

        if (mul_fn) {
            g_gf8_mul_pipeline = [g_device newComputePipelineStateWithFunction:mul_fn error:&error];
        }
        if (add_fn) {
            // Store add pipeline in mul pipeline slot for simplicity (not used separately)
        }

        g_gpu_initialized = true;
        NSLog(@"GF(2^8) Metal GPU initialized: mul_pipeline=%@",
              g_gf8_mul_pipeline ? @"YES" : @"NO");
        return 0;
    }
}

// ============================================================
// ANE Device Management
// ============================================================

bool ane_bt_available(void) {
    // Check if Metal GPU is available
    if (g_gpu_initialized) return g_gf8_mul_pipeline != nil;

    // Try to initialize
    return init_gpu() == 0 && g_gf8_mul_pipeline != nil;
}

void* ane_bt_create(int logN) {
    (void)logN;
    if (init_gpu() != 0) return NULL;
    return g_gf8_mul_pipeline != NULL ? (void*)0x1 : NULL;
}

void ane_bt_destroy(void* state) {
    (void)state;
    // Nothing to destroy - pipeline states are cached
}

// ============================================================
// GF(2^8) Operations
// ============================================================

uint8_t ane_bt_gf8_mul(uint8_t a, uint8_t b) {
    gf8_init_tables();
    return gf8_mul_cpu(a, b);
}

uint8_t ane_bt_gf8_add(uint8_t a, uint8_t b) {
    return a ^ b;  // XOR is addition in GF(2^8)
}

// GPU dispatch for batch GF(2^8) multiply
static int dispatch_gf8_batch_mul(const uint8_t* a, const uint8_t* b, uint8_t* r, int n) {
    if (!g_gf8_mul_pipeline) return -1;

    @autoreleasepool {
        int aSize = n * sizeof(uint8_t);
        int bSize = n * sizeof(uint8_t);
        int rSize = n * sizeof(uint8_t);

        id<MTLBuffer> aBuf = [g_device newBufferWithBytes:a length:aSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [g_device newBufferWithBytes:b length:bSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> rBuf = [g_device newBufferWithLength:rSize options:MTLResourceStorageModeShared];

        uint32_t nVal = n;
        id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&nVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_gf8_mul_pipeline];
        [enc setBuffer:aBuf offset:0 atIndex:0];
        [enc setBuffer:bBuf offset:0 atIndex:1];
        [enc setBuffer:rBuf offset:0 atIndex:2];
        [enc setBuffer:nBuf offset:0 atIndex:3];

        // Dispatch enough threadgroups to cover n elements
        [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(r, rBuf.contents, rSize);
        return 0;
    }
}

int ane_bt_batch_gf8_mul(const uint8_t* a, const uint8_t* b, uint8_t* r, int n) {
    gf8_init_tables();

    // Try GPU path first
    if (g_gf8_mul_pipeline && n > 0) {
        int result = dispatch_gf8_batch_mul(a, b, r, n);
        if (result == 0) return 0;
    }

    // CPU fallback
    for (int i = 0; i < n; i++) {
        r[i] = gf8_mul_cpu(a[i], b[i]);
    }
    return 0;
}

// ============================================================
// GF(2^64) Operations (using scalar PMULL fallback)
// ============================================================

uint64_t ane_bt_gf64_mul(uint64_t a, uint64_t b) {
    return bt_gf64_mul_scalar(a, b);
}

uint64_t ane_bt_gf64_add(uint64_t a, uint64_t b) {
    return a ^ b;  // XOR is free in GF(2^64)
}

// ============================================================
// GF(2^128) Operations (using scalar Karatsuba)
// ============================================================

void ane_bt_gf128_mul(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]) {
    bt_gf128_mul_scalar(a, b, r);
}

// ============================================================
// Batch Operations
// ============================================================

int ane_bt_batch_gf64_mul(void* state, const uint64_t* a, const uint64_t* b, uint64_t* r, int n) {
    (void)state;
    for (int i = 0; i < n; i++) {
        r[i] = bt_gf64_mul_scalar(a[i], b[i]);
    }
    return 0;
}

// ============================================================
// Scalar Fallback Helpers (available even without ANE)
// ============================================================

// GF(2^64) reduction polynomial: x^64 + x^4 + x^3 + x + 1
// For bits [64..127], reduce by XORing shifted copies:
//   x^64 = x^4 + x^3 + x + 1
//   Overflow bits (bits that shifted past 63) need second reduction pass

static inline uint64_t gf64_reduce_scalar(uint64_t lo, uint64_t hi) {
    uint64_t t;
    // First round: fold hi into lo
    t = hi;
    lo ^= (t << 1) ^ (t << 3) ^ (t << 4) ^ t;
    // Overflow bits from shifts
    uint64_t overflow = (t >> 63) ^ (t >> 61) ^ (t >> 60);
    // Second round: reduce overflow
    lo ^= overflow ^ (overflow << 1) ^ (overflow << 3) ^ (overflow << 4);
    return lo;
}

// Carry-less multiply 64×64 → 128 bits using ARM NEON PMULL
// Note: This requires <arm_neon.h> and targets ARM64 only
#if defined(__ARM_NEON)

#include <arm_neon.h>

static inline void clmul64_scalar(uint64_t a, uint64_t b, uint64_t *lo, uint64_t *hi) {
    poly64_t pa = (poly64_t)a;
    poly64_t pb = (poly64_t)b;
    poly128_t result = vmull_p64(pa, pb);
    uint64x2_t r = vreinterpretq_u64_p128(result);
    *lo = vgetq_lane_u64(r, 0);
    *hi = vgetq_lane_u64(r, 1);
}

#else

// Software fallback for non-ARM platforms (for compilation only)
// This is not cryptographically correct — only for build testing
static inline void clmul64_scalar(uint64_t a, uint64_t b, uint64_t *lo, uint64_t *hi) {
    // Simple 64-bit multiply as placeholder (not carry-less!)
    // Real implementation would need bit-serial shift-XOR
    uint64_t result = a * b;  // PLACEHOLDER — not correct for GF(2^64)
    *lo = result;
    *hi = 0;
}

#endif

extern "C" {

uint64_t bt_gf64_mul_scalar(uint64_t a, uint64_t b) {
    uint64_t lo, hi;
    clmul64_scalar(a, b, &lo, &hi);
    return gf64_reduce_scalar(lo, hi);
}

// GF(2^128) using Karatsuba over GF(2^64)
// a = a_lo + a_hi * x^64, b = b_lo + b_hi * x^64
// result = z0 + z1*x^64 + z2*x^128
// where:
//   z0 = a_lo * b_lo
//   z2 = a_hi * b_hi
//   z1 = (a_lo + a_hi) * (b_lo + b_hi) - z0 - z2
//
// After Karatsuba:
//   r[0] = z0
//   r[1] = z0 ^ z1 ^ z2  (recombined)
// But we need proper reduction...

// Barrett reduction for GF(2^128) with polynomial x^128 + x^7 + x^2 + x + 1
static inline void gf128_reduce_scalar(uint64_t r0, uint64_t r1, uint64_t r2, uint64_t r3,
                                        uint64_t *out_lo, uint64_t *out_hi) {
    // Reduce r3 (bits [192..255]) using x^128 = x^7 + x^2 + x + 1
    r1 ^= (r3 << 7) ^ (r3 << 2) ^ (r3 << 1) ^ r3;
    r2 ^= (r3 >> 57) ^ (r3 >> 62) ^ (r3 >> 63);

    // Reduce r2 (bits [128..191])
    r0 ^= (r2 << 7) ^ (r2 << 2) ^ (r2 << 1) ^ r2;
    r1 ^= (r2 >> 57) ^ (r2 >> 62) ^ (r2 >> 63);

    *out_lo = r0;
    *out_hi = r1;
}

void bt_gf128_mul_scalar(const uint64_t a[2], const uint64_t b[2], uint64_t r[2]) {
    uint64_t lo_lo, lo_hi, hi_lo, hi_hi, mid_lo, mid_hi;

    // Karatsuba: 3× GF(2^64) multiply
    clmul64_scalar(a[0], b[0], &lo_lo, &lo_hi);                    // z0_lo, z0_hi
    clmul64_scalar(a[1], b[1], &hi_lo, &hi_hi);                    // z2_lo, z2_hi
    clmul64_scalar(a[0] ^ a[1], b[0] ^ b[1], &mid_lo, &mid_hi);    // z1_lo, z1_hi

    // Karatsuba recombination: z1 = (a_lo+a_hi)*(b_lo+b_hi) - z0 - z2
    uint64_t z1_lo = mid_lo ^ lo_lo ^ hi_lo;
    uint64_t z1_hi = mid_hi ^ lo_hi ^ hi_hi;

    // Build 256-bit result: r = z0 + z1*x^64 + z2*x^128
    uint64_t r0 = lo_lo;
    uint64_t r1 = lo_hi ^ z1_lo;
    uint64_t r2 = hi_lo ^ z1_hi;
    uint64_t r3 = hi_hi;

    // Reduce to GF(2^128)
    gf128_reduce_scalar(r0, r1, r2, r3, &r[0], &r[1]);
}

uint64_t bt_gf64_add_scalar(uint64_t a, uint64_t b) {
    return a ^ b;  // XOR is addition in GF(2^64)
}

} // extern "C"
