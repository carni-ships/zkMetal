// ane_circle_ntt.mm — ANE Circle NTT C wrapper with Metal GPU acceleration
//
// Circle NTT differs from standard NTT:
// - Layer 0 uses y-coordinate twiddles (twin-coset decomposition)
// - Layers 1+ use x-coordinate twiddles with the squaring map
//
// This implementation provides ANE acceleration for the butterfly operations.
// The fused kernel approach processes multiple stages in one GPU dispatch.

#include "include/ane_circle_ntt.h"
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// ============================================================
// M31 field constants (p = 2^31 - 1)
// ============================================================

#define M31_P  0x7FFFFFFFu

// ============================================================
// Scalar M31 arithmetic
// ============================================================

static inline uint32_t m31_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    uint32_t lo = (uint32_t)(prod & M31_P);
    uint32_t hi = (uint32_t)(prod >> 31);
    uint32_t s = lo + hi;
    uint32_t r = (s & M31_P) + (s >> 31);
    return (r == M31_P) ? 0u : r;
}

static inline uint32_t m31_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    uint32_t r = (s & M31_P) + (s >> 31);
    return (r == M31_P) ? 0u : r;
}

static inline uint32_t m31_sub(uint32_t a, uint32_t b) {
    return (a >= b) ? a - b : a + M31_P - b;
}

// ============================================================
// Bit-reversal permutation
// ============================================================

static inline uint32_t m31_bitrev(uint32_t val, uint32_t num_bits) {
    uint32_t rev = 0;
    for (uint32_t i = 0; i < num_bits; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }
    return rev;
}

// ============================================================
// Metal GPU State
// ============================================================

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLComputePipelineState> g_butterfly_pipeline = nil;
static id<MTLComputePipelineState> g_intt_butterfly_pipeline = nil;
static id<MTLComputePipelineState> g_scale_pipeline = nil;
static id<MTLComputePipelineState> g_bitrev_pipeline = nil;
static bool g_gpu_initialized = false;

// Preallocated reusable buffers for minimizing dispatch overhead
static id<MTLBuffer> g_data_buffer = nil;
static id<MTLBuffer> g_twiddle_buffer = nil;
static id<MTLBuffer> g_param_buffer = nil;  // n, stage, scalar
static id<MTLCommandBuffer> g_pending_cmdbuf = nil;
static uint32_t g_data_buffer_capacity = 0;
static uint32_t g_twiddle_buffer_capacity = 0;

// ============================================================
// Shader loading helper
// ============================================================

// Clean the #include lines from shader source
static NSString* cleanShaderSource(NSString* source) {
    NSMutableString* result = [NSMutableString string];
    [source enumerateLinesUsingBlock:^(NSString *line, BOOL *stop) {
        if (![line containsString:@"#include"]) {
            [result appendString:line];
            [result appendString:@"\n"];
        }
    }];
    return result;
}

// Find the Shaders directory
static NSString* findShaderDir(void) {
    NSArray* searchPaths = @[
        @"./Sources/Shaders",
        @"../Sources/Shaders",
        @"../../Sources/Shaders",
        [[NSBundle mainBundle] resourcePath]
    ];

    for (NSString* path in searchPaths) {
        NSString* mersennePath = [path stringByAppendingPathComponent:@"fields/mersenne31.metal"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:mersennePath]) {
            return path;
        }
    }
    return @"./Sources/Shaders";
}

// Load and combine shaders from external files
static NSString* loadShaderSource(void) {
    NSString* shaderDir = findShaderDir();
    NSString* fieldPath = [shaderDir stringByAppendingPathComponent:@"fields/mersenne31.metal"];
    NSString* nttPath = [shaderDir stringByAppendingPathComponent:@"ntt/ntt_circle.metal"];

    NSError* error = nil;
    NSString* fieldSource = [NSString stringWithContentsOfFile:fieldPath
                                                      encoding:NSUTF8StringEncoding
                                                         error:&error];
    if (!fieldSource) {
        NSLog(@"Failed to load field shader: %@", error);
        return nil;
    }

    NSString* nttSource = [NSString stringWithContentsOfFile:nttPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
    if (!nttSource) {
        NSLog(@"Failed to load NTT shader: %@", error);
        return nil;
    }

    // Clean the #ifndef/#define/#endif guards from field source
    NSMutableString* cleanField = [NSMutableString string];
    [fieldSource enumerateLinesUsingBlock:^(NSString *line, BOOL *stop) {
        if (![line containsString:@"#ifndef MERSENNE31_METAL"] &&
            ![line containsString:@"#define MERSENNE31_METAL"] &&
            ![line containsString:@"#endif // MERSENNE31_METAL"] &&
            ![line containsString:@"#endif // MERSENNE31"]) {
            [cleanField appendString:line];
            [cleanField appendString:@"\n"];
        }
    }];

    // Clean the #include lines from NTT source
    NSString* cleanNTT = cleanShaderSource(nttSource);

    return [cleanField stringByAppendingString:cleanNTT];
}

// ============================================================
// ANE Circle NTT Lifecycle
// ============================================================

int ane_circle_ntt_init(void) {
    if (g_gpu_initialized) return 0;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            return -1;
        }

        // Check ANE support (Apple8 = ANE capable)
        if (![g_device supportsFamily:MTLGPUFamilyApple8]) {
            g_gpu_initialized = true;
            return 0;
        }

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            return -1;
        }

        // Load shader source from external files
        NSString* source = loadShaderSource();
        if (!source) {
            NSLog(@"Failed to load shader source");
            return -1;
        }

        NSError *error = nil;
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

        g_library = [g_device newLibraryWithSource:source
                                           options:options
                                             error:&error];
        if (!g_library) {
            NSLog(@"Failed to compile shader: %@", error);
            return -1;
        }

        // Create pipeline states - prefer combined-param variants for efficiency
        id<MTLFunction> butterfly_fn = [g_library newFunctionWithName:@"circle_ntt_butterfly_combined"];
        if (!butterfly_fn) {
            butterfly_fn = [g_library newFunctionWithName:@"circle_ntt_butterfly"];
        }
        id<MTLFunction> intt_butterfly_fn = [g_library newFunctionWithName:@"circle_intt_butterfly_combined"];
        if (!intt_butterfly_fn) {
            intt_butterfly_fn = [g_library newFunctionWithName:@"circle_intt_butterfly"];
        }
        id<MTLFunction> scale_fn = [g_library newFunctionWithName:@"circle_ntt_scale_combined"];
        if (!scale_fn) {
            scale_fn = [g_library newFunctionWithName:@"circle_ntt_scale"];
        }
        id<MTLFunction> bitrev_fn = [g_library newFunctionWithName:@"circle_ntt_bitrev"];

        if (butterfly_fn) {
            g_butterfly_pipeline = [g_device newComputePipelineStateWithFunction:butterfly_fn error:&error];
        }
        if (intt_butterfly_fn) {
            g_intt_butterfly_pipeline = [g_device newComputePipelineStateWithFunction:intt_butterfly_fn error:&error];
        }
        if (scale_fn) {
            g_scale_pipeline = [g_device newComputePipelineStateWithFunction:scale_fn error:&error];
        }
        if (bitrev_fn) {
            g_bitrev_pipeline = [g_device newComputePipelineStateWithFunction:bitrev_fn error:&error];
        }

        g_gpu_initialized = true;
        return 0;
    }
}

void ane_circle_ntt_shutdown(void) {
    if (!g_gpu_initialized) return;

    @autoreleasepool {
        g_butterfly_pipeline = nil;
        g_intt_butterfly_pipeline = nil;
        g_scale_pipeline = nil;
        g_bitrev_pipeline = nil;
        g_data_buffer = nil;
        g_twiddle_buffer = nil;
        g_param_buffer = nil;
        g_pending_cmdbuf = nil;
        g_data_buffer_capacity = 0;
        g_twiddle_buffer_capacity = 0;
        g_library = nil;
        g_queue = nil;
        g_device = nil;
        g_gpu_initialized = false;
    }
}

static bool ane_circle_ntt_gpu_available_internal(void) {
    return g_gpu_initialized && g_butterfly_pipeline != nil;
}

// ============================================================
// Buffer management helpers
// ============================================================

// Ensure data buffer is large enough, recreate only if needed
static bool ensure_data_buffer(uint32_t n) {
    uint32_t needed = n * sizeof(uint32_t);
    if (g_data_buffer_capacity < needed) {
        g_data_buffer = [g_device newBufferWithLength:needed
                                              options:MTLResourceStorageModeShared];
        if (!g_data_buffer) return false;
        g_data_buffer_capacity = needed;
    }
    return true;
}

// Ensure twiddle buffer is large enough, recreate only if needed
static bool ensure_twiddle_buffer(uint32_t n) {
    uint32_t needed = (n / 2) * sizeof(uint32_t);
    if (g_twiddle_buffer_capacity < needed) {
        g_twiddle_buffer = [g_device newBufferWithLength:needed
                                                options:MTLResourceStorageModeShared];
        if (!g_twiddle_buffer) return false;
        g_twiddle_buffer_capacity = needed;
    }
    return true;
}

// Wait for any pending command buffer to complete
static void sync_pending(void) {
    if (g_pending_cmdbuf) {
        [g_pending_cmdbuf waitUntilCompleted];
        g_pending_cmdbuf = nil;
    }
}

// ============================================================
// GPU dispatch helpers
// ============================================================

static void ensure_initialized(void) {
    if (!g_gpu_initialized) {
        ane_circle_ntt_init();
    }
}

// Dispatch one butterfly stage with buffer reuse
// Uses kernel variant that takes combined [n, stage] params at buffer index 2
static int dispatch_butterfly(uint32_t* data, const uint32_t* twiddles,
                              int n, int logN, int stage) {
    ensure_initialized();
    if (!g_butterfly_pipeline) return -1;

    sync_pending();  // Ensure previous dispatch is done before we read data

    if (!ensure_data_buffer(n) || !ensure_twiddle_buffer(n)) {
        return -1;
    }

    @autoreleasepool {
        int num_butterflies = n / 2;
        int tg_size = 256;

        // Copy data and twiddles to preallocated buffers
        memcpy(g_data_buffer.contents, data, n * sizeof(uint32_t));
        memcpy(g_twiddle_buffer.contents, twiddles, (n/2) * sizeof(uint32_t));

        // Set up param buffer with [n, stage]
        uint32_t params[2] = { (uint32_t)n, (uint32_t)stage };
        if (!g_param_buffer || g_param_buffer.length < sizeof(params)) {
            g_param_buffer = [g_device newBufferWithBytes:params
                                                  length:sizeof(params)
                                                 options:MTLResourceStorageModeShared];
        } else {
            memcpy(g_param_buffer.contents, params, sizeof(params));
        }

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_butterfly_pipeline];
        [enc setBuffer:g_data_buffer offset:0 atIndex:0];
        [enc setBuffer:g_twiddle_buffer offset:0 atIndex:1];
        [enc setBuffer:g_param_buffer offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((num_butterflies + tg_size - 1) / tg_size, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        g_pending_cmdbuf = cmdBuf;

        // Wait and copy result back
        [cmdBuf waitUntilCompleted];
        memcpy(data, g_data_buffer.contents, n * sizeof(uint32_t));
        g_pending_cmdbuf = nil;

        return 0;
    }
}

// Dispatch inverse butterfly stage with buffer reuse
static int dispatch_intt_butterfly(uint32_t* data, const uint32_t* inv_twiddles,
                                   int n, int logN, int stage) {
    ensure_initialized();
    if (!g_intt_butterfly_pipeline) return -1;

    sync_pending();

    if (!ensure_data_buffer(n) || !ensure_twiddle_buffer(n)) {
        return -1;
    }

    @autoreleasepool {
        int num_butterflies = n / 2;
        int tg_size = 256;

        // Copy data and twiddles to preallocated buffers
        memcpy(g_data_buffer.contents, data, n * sizeof(uint32_t));
        memcpy(g_twiddle_buffer.contents, inv_twiddles, (n/2) * sizeof(uint32_t));

        // Set up param buffer with [n, stage]
        uint32_t params[2] = { (uint32_t)n, (uint32_t)stage };
        if (!g_param_buffer || g_param_buffer.length < sizeof(params)) {
            g_param_buffer = [g_device newBufferWithBytes:params
                                                  length:sizeof(params)
                                                 options:MTLResourceStorageModeShared];
        } else {
            memcpy(g_param_buffer.contents, params, sizeof(params));
        }

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_intt_butterfly_pipeline];
        [enc setBuffer:g_data_buffer offset:0 atIndex:0];
        [enc setBuffer:g_twiddle_buffer offset:0 atIndex:1];
        [enc setBuffer:g_param_buffer offset:0 atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((num_butterflies + tg_size - 1) / tg_size, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        g_pending_cmdbuf = cmdBuf;

        // Wait and copy result back
        [cmdBuf waitUntilCompleted];
        memcpy(data, g_data_buffer.contents, n * sizeof(uint32_t));
        g_pending_cmdbuf = nil;

        return 0;
    }
}

// Dispatch scale with buffer reuse
static int dispatch_scale(uint32_t* data, uint32_t scalar, int n) {
    ensure_initialized();
    if (!g_scale_pipeline) return -1;

    sync_pending();

    if (!ensure_data_buffer(n)) {
        return -1;
    }

    @autoreleasepool {
        int tg_size = 256;

        // Copy data to preallocated buffer
        memcpy(g_data_buffer.contents, data, n * sizeof(uint32_t));

        // Set up param buffer with [n, scalar]
        uint32_t params[2] = { (uint32_t)n, scalar };
        if (!g_param_buffer || g_param_buffer.length < sizeof(params)) {
            g_param_buffer = [g_device newBufferWithBytes:params
                                                  length:sizeof(params)
                                                 options:MTLResourceStorageModeShared];
        } else {
            memcpy(g_param_buffer.contents, params, sizeof(params));
        }

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_scale_pipeline];
        [enc setBuffer:g_data_buffer offset:0 atIndex:0];
        [enc setBuffer:g_param_buffer offset:sizeof(uint32_t) atIndex:1];  // scalar at index 1
        [enc setBuffer:g_param_buffer offset:0 atIndex:2];  // n at index 2
        [enc dispatchThreadgroups:MTLSizeMake((n + tg_size - 1) / tg_size, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        g_pending_cmdbuf = cmdBuf;

        // Wait and copy result back
        [cmdBuf waitUntilCompleted];
        memcpy(data, g_data_buffer.contents, n * sizeof(uint32_t));
        g_pending_cmdbuf = nil;

        return 0;
    }
}

// ============================================================
// Public API Implementations
// ============================================================

extern "C" {

bool ane_circle_ntt_gpu_available(void) {
    return ane_circle_ntt_gpu_available_internal();
}

void* ane_circle_ntt_create(int logN) {
    (void)logN;
    ensure_initialized();
    return g_butterfly_pipeline != NULL ? (void*)0x1 : NULL;
}

void ane_circle_ntt_destroy(void* state) {
    (void)state;
}

int ane_circle_ntt(void* opaque_state, uint32_t* data,
                   const uint32_t* twiddles, int logN) {
    (void)opaque_state;

    int n = 1 << logN;

    if (g_butterfly_pipeline) {
        // Per-layer butterfly dispatch
        int twiddle_stride = n / 2;

        // Layers k-1 down to 1
        for (int layer = logN - 1; layer >= 1; layer--) {
            int stage = logN - 1 - layer;
            const uint32_t* layer_tw = twiddles + layer * twiddle_stride;
            if (dispatch_butterfly(data, layer_tw, n, logN, stage) != 0) {
                return -1;
            }
        }

        // Layer 0 (y-twiddle)
        if (logN >= 1) {
            int stage = logN - 1;
            const uint32_t* layer_tw = twiddles;
            if (dispatch_butterfly(data, layer_tw, n, logN, stage) != 0) {
                return -1;
            }
        }

        return 0;
    }

    return -1;
}

int ane_circle_intt(void* opaque_state, uint32_t* data,
                    const uint32_t* inv_twiddles, uint32_t inv_n, int logN) {
    (void)opaque_state;

    int n = 1 << logN;
    int twiddle_stride = n / 2;

    if (g_intt_butterfly_pipeline) {
        // Layer 0 (y-twiddle) first
        if (logN >= 1) {
            int stage = logN - 1;
            const uint32_t* layer_tw = inv_twiddles;
            if (dispatch_intt_butterfly(data, layer_tw, n, logN, stage) != 0) {
                return -1;
            }
        }

        // Layers 1..k-1 (x-twiddle)
        for (int layer = 1; layer < logN; layer++) {
            int stage = logN - 1 - layer;
            const uint32_t* layer_tw = inv_twiddles + layer * twiddle_stride;
            if (dispatch_intt_butterfly(data, layer_tw, n, logN, stage) != 0) {
                return -1;
            }
        }

        // Scale by inv_n
        if (dispatch_scale(data, inv_n, n) != 0) {
            return -1;
        }

        return 0;
    }

    return -1;
}

int ane_circle_ntt_forward(uint32_t* data, const uint32_t* twiddles, int logN) {
    return ane_circle_ntt(NULL, data, twiddles, logN);
}

int ane_circle_ntt_inverse(uint32_t* data, const uint32_t* inv_twiddles,
                          uint32_t inv_n, int logN) {
    return ane_circle_intt(NULL, data, inv_twiddles, inv_n, logN);
}

} // extern "C"
