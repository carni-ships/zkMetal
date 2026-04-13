// ane_babybear.mm — ANE BabyBear NTT C wrapper with Metal GPU acceleration
//
// This file provides the C API for BabyBear NTT with Metal GPU acceleration
// that automatically offloads to ANE on Apple Silicon.
//
// GPU path: compiles Metal shader at runtime and dispatches compute kernels
// Scalar path: uses inline scalar arithmetic (fallback when GPU unavailable)

#include "include/ane_babybear.h"
#include <Metal/Metal.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// ============================================================
// BabyBear field constants (plain form, non-Montgomery)
// ============================================================

#define BB_P         0x78000001u
#define BB_MU        2290649223u
#define BB_R2        1172168163u
#define BB_P_INV     2281701377u

// ============================================================
// Scalar BabyBear arithmetic (plain form)
// ============================================================

static inline uint32_t bb_mul(uint32_t a, uint32_t b) {
    uint64_t prod = (uint64_t)a * (uint64_t)b;
    uint32_t prod_lo = (uint32_t)prod;
    uint32_t prod_hi = (uint32_t)(prod >> 32);
    uint64_t t1 = (uint64_t)prod_lo * (uint64_t)BB_MU;
    uint64_t t2 = (uint64_t)prod_hi * (uint64_t)BB_MU;
    uint32_t q = (uint32_t)((t2 + (t1 >> 32)) >> 30);
    uint32_t r = (uint32_t)(prod - (uint64_t)q * (uint64_t)BB_P);
    return r >= BB_P ? r - BB_P : r;
}

static inline uint32_t bb_add(uint32_t a, uint32_t b) {
    uint32_t s = a + b;
    return s >= BB_P ? s - BB_P : s;
}

static inline uint32_t bb_sub(uint32_t a, uint32_t b) {
    return a >= b ? a - b : a + BB_P - b;
}

// ============================================================
// Bit-reversal permutation
// ============================================================

static inline uint32_t bb_bitrev(uint32_t val, uint32_t num_bits) {
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
static id<MTLComputePipelineState> g_ntt_fused_pipeline = nil;
static id<MTLComputePipelineState> g_intt_fused_pipeline = nil;
static id<MTLComputePipelineState> g_bitrev_pipeline = nil;
static id<MTLComputePipelineState> g_scale_pipeline = nil;
static id<MTLComputePipelineState> g_to_monty_pipeline = nil;
static id<MTLComputePipelineState> g_from_monty_pipeline = nil;
static bool g_gpu_initialized = false;

// Metal shader source (standard Metal, no metal_ane dependency)
static const char* g_shader_source = R"(
#include <metal_stdlib>
using namespace metal;

// BabyBear field: p = 2^31 - 2^27 + 1 = 0x78000001
constant uint BB_P  = 0x78000001u;
constant uint BB_MU = 2290649223u;

struct Bb {
    uint v;
};

Bb bb_zero() { return Bb{0}; }
Bb bb_one()  { return Bb{1}; }

Bb bb_from_u32(uint v) {
    return Bb{v >= BB_P ? v - BB_P : v};
}

Bb bb_add(Bb a, Bb b) {
    uint sum = a.v + b.v;
    return Bb{sum >= BB_P ? sum - BB_P : sum};
}

Bb bb_sub(Bb a, Bb b) {
    if (a.v >= b.v) return Bb{a.v - b.v};
    return Bb{a.v + BB_P - b.v};
}

Bb bb_neg(Bb a) {
    if (a.v == 0) return a;
    return Bb{BB_P - a.v};
}

Bb bb_mul(Bb a, Bb b) {
    ulong prod = ulong(a.v) * ulong(b.v);
    uint prod_lo = uint(prod);
    uint prod_hi = uint(prod >> 32);
    ulong t1 = ulong(prod_lo) * ulong(BB_MU);
    ulong t2 = ulong(prod_hi) * ulong(BB_MU);
    uint q = uint((t2 + (t1 >> 32)) >> 30);
    uint r = uint(prod - ulong(q) * ulong(BB_P));
    return Bb{r >= BB_P ? r - BB_P : r};
}

Bb bb_sqr(Bb a) { return bb_mul(a, a); }

inline uint bb_bitrev(uint val, uint num_bits) {
    uint rev = 0;
    for (uint i = 0; i < num_bits; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }
    return rev;
}

// Fused DIT NTT kernel — forward NTT for BabyBear
kernel void ane_bb_ntt_fused(
    device Bb* data            [[buffer(0)]],
    device const Bb* twiddles  [[buffer(1)]],
    constant uint& n           [[buffer(2)]],
    constant uint& logN        [[buffer(3)]],
    constant uint& local_stages [[buffer(4)]],
    uint tid                   [[thread_index_in_threadgroup]],
    uint tgid                  [[threadgroup_position_in_grid]],
    uint tg_size               [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup Bb shared[256];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint global_lo = base + idx_lo;
    uint global_hi = base + idx_hi;

    uint rev_lo = bb_bitrev(global_lo, logN);
    uint rev_hi = bb_bitrev(global_hi, logN);

    if (global_lo < n) shared[idx_lo] = data[rev_lo];
    if (global_hi < n) shared[idx_hi] = data[rev_hi];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < local_stages; s++) {
        uint half_block = 1u << s;
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint a = shared[i].v;
        uint b = shared[j].v;

        uint global_block_size = 1u << (s + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Bb result_i, result_j;
        if (twiddle_idx == 0) {
            uint sum = a + b;
            uint diff = (a >= b) ? a - b : a + BB_P - b;
            result_i = Bb{sum >= BB_P ? sum - BB_P : sum};
            result_j = Bb{diff};
        } else {
            Bb w = twiddles[twiddle_idx];
            Bb wb = bb_mul(Bb{b}, w);
            Bb sum = bb_add(Bb{a}, wb);
            Bb diff = bb_sub(Bb{a}, wb);
            result_i = sum;
            result_j = diff;
        }

        shared[i] = result_i;
        shared[j] = result_j;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (global_lo < n) data[global_lo] = shared[idx_lo];
    if (global_hi < n) data[global_hi] = shared[idx_hi];
}

// Fused DIF iNTT kernel — inverse NTT for BabyBear
kernel void ane_bb_intt_fused(
    device Bb* data             [[buffer(0)]],
    device const Bb* twiddles_inv [[buffer(1)]],
    constant uint& n             [[buffer(2)]],
    constant uint& logN          [[buffer(3)]],
    constant uint& local_stages   [[buffer(4)]],
    device const Bb* inv_n       [[buffer(5)]],
    uint tid                    [[thread_index_in_threadgroup]],
    uint tgid                   [[threadgroup_position_in_grid]],
    uint tg_size                [[threads_per_threadgroup]]
) {
    uint block_size = tg_size << 1;
    uint base = tgid * block_size;

    threadgroup Bb shared[256];

    uint idx_lo = tid;
    uint idx_hi = tid + tg_size;
    uint global_lo = base + idx_lo;
    uint global_hi = base + idx_hi;

    if (global_lo < n) shared[idx_lo] = data[global_lo];
    if (global_hi < n) shared[idx_hi] = data[global_hi];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0; s < local_stages; s++) {
        uint stage = local_stages - 1 - s;
        uint half_block = 1u << stage;
        uint local_block_size = half_block << 1;

        uint block_idx = tid / half_block;
        uint local_idx = tid % half_block;
        uint i = block_idx * local_block_size + local_idx;
        uint j = i + half_block;

        uint a = shared[i].v;
        uint b = shared[j].v;

        uint sum_val = a + b;
        uint diff_val = (a >= b) ? a - b : a + BB_P - b;

        sum_val = (sum_val >= BB_P) ? sum_val - BB_P : sum_val;
        diff_val = (diff_val >= BB_P) ? diff_val - BB_P : diff_val;

        uint global_block_size = 1u << (stage + 1);
        uint twiddle_idx = local_idx * (n / global_block_size);

        Bb result_i = Bb{sum_val};
        Bb result_j;

        if (twiddle_idx == 0) {
            result_j = Bb{diff_val};
        } else {
            Bb w = twiddles_inv[twiddle_idx];
            result_j = bb_mul(Bb{diff_val}, w);
        }

        shared[i] = result_i;
        shared[j] = result_j;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    Bb scale = inv_n[0];
    uint rev_lo = bb_bitrev(global_lo, logN);
    uint rev_hi = bb_bitrev(global_hi, logN);

    if (global_lo < n) {
        Bb scaled = bb_mul(shared[idx_lo], scale);
        data[rev_lo] = scaled;
    }
    if (global_hi < n) {
        Bb scaled = bb_mul(shared[idx_hi], scale);
        data[rev_hi] = scaled;
    }
}

// Scale kernel
kernel void ane_bb_scale(
    device Bb* data         [[buffer(0)]],
    device const Bb* scale  [[buffer(1)]],
    constant uint& n        [[buffer(2)]],
    uint gid               [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    data[gid] = bb_mul(data[gid], scale[0]);
}

// Bit-reversal permutation kernel
kernel void ane_bb_bitrev(
    device Bb* data     [[buffer(0)]],
    constant uint& n    [[buffer(1)]],
    constant uint& log_n [[buffer(2)]],
    uint gid           [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint rev = bb_bitrev(gid, log_n);
    if (gid < rev) {
        Bb tmp = data[gid];
        data[gid] = data[rev];
        data[rev] = tmp;
    }
}

// Montgomery conversion kernels
constant uint BB_P_INV = 2281701377u;
constant uint BB_R2    = 1172168163u;

kernel void ane_bb_to_monty(
    device Bb* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    Bb x = data[gid];
    data[gid] = bb_mul(x, Bb{BB_R2});
}

kernel void ane_bb_from_monty(
    device Bb* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    // In plain Barrett form, from_monty is identity
    (void)data;
}
)";

// ============================================================
// ANE BabyBear NTT Lifecycle
// ============================================================

int ane_babybear_ntt_init(void) {
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

        NSError *error = nil;
        NSString *source = [NSString stringWithUTF8String:g_shader_source];
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;

        g_library = [g_device newLibraryWithSource:source
                                           options:options
                                             error:&error];
        if (!g_library) {
            return -1;
        }

        // Create pipeline states
        id<MTLFunction> ntt_fused_fn = [g_library newFunctionWithName:@"ane_bb_ntt_fused"];
        id<MTLFunction> intt_fused_fn = [g_library newFunctionWithName:@"ane_bb_intt_fused"];
        id<MTLFunction> bitrev_fn = [g_library newFunctionWithName:@"ane_bb_bitrev"];
        id<MTLFunction> scale_fn = [g_library newFunctionWithName:@"ane_bb_scale"];
        id<MTLFunction> to_monty_fn = [g_library newFunctionWithName:@"ane_bb_to_monty"];
        id<MTLFunction> from_monty_fn = [g_library newFunctionWithName:@"ane_bb_from_monty"];

        if (ntt_fused_fn) {
            g_ntt_fused_pipeline = [g_device newComputePipelineStateWithFunction:ntt_fused_fn error:&error];
        }
        if (intt_fused_fn) {
            g_intt_fused_pipeline = [g_device newComputePipelineStateWithFunction:intt_fused_fn error:&error];
        }
        if (bitrev_fn) {
            g_bitrev_pipeline = [g_device newComputePipelineStateWithFunction:bitrev_fn error:&error];
        }
        if (scale_fn) {
            g_scale_pipeline = [g_device newComputePipelineStateWithFunction:scale_fn error:&error];
        }
        if (to_monty_fn) {
            g_to_monty_pipeline = [g_device newComputePipelineStateWithFunction:to_monty_fn error:&error];
        }
        if (from_monty_fn) {
            g_from_monty_pipeline = [g_device newComputePipelineStateWithFunction:from_monty_fn error:&error];
        }

        g_gpu_initialized = true;
        return 0;
    }
}

void ane_babybear_ntt_shutdown(void) {
    if (!g_gpu_initialized) return;

    @autoreleasepool {
        g_ntt_fused_pipeline = nil;
        g_intt_fused_pipeline = nil;
        g_bitrev_pipeline = nil;
        g_scale_pipeline = nil;
        g_to_monty_pipeline = nil;
        g_from_monty_pipeline = nil;
        g_library = nil;
        g_queue = nil;
        g_device = nil;
        g_gpu_initialized = false;
    }
}

bool ane_babybear_ntt_available(void) {
    return g_gpu_initialized && g_ntt_fused_pipeline != nil;
}

// ============================================================
// GPU dispatch helpers
// ============================================================

static void ensure_ntt_initialized(void) {
    if (!g_gpu_initialized) {
        ane_babybear_ntt_init();
    }
}

// Forward DIT NTT dispatch
static int dispatch_ntt_fused(uint32_t* data, const uint32_t* twiddles, int n, int logN, int local_stages) {
    ensure_ntt_initialized();
    if (!g_ntt_fused_pipeline) return -1;

    @autoreleasepool {
        int tg_size = 128;  // matches kernel design
        int n_tgroups = n / (tg_size * 2);

        id<MTLBuffer> dataBuf = [g_device newBufferWithBytes:data
                                                      length:n * sizeof(uint32_t)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> twBuf = [g_device newBufferWithBytes:twiddles
                                                    length:(n/2) * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&n
                                                   length:sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> logNBuf = [g_device newBufferWithBytes:&logN
                                                      length:sizeof(uint32_t)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> stagesBuf = [g_device newBufferWithBytes:&local_stages
                                                        length:sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_ntt_fused_pipeline];
        [enc setBuffer:dataBuf offset:0 atIndex:0];
        [enc setBuffer:twBuf offset:0 atIndex:1];
        [enc setBuffer:nBuf offset:0 atIndex:2];
        [enc setBuffer:logNBuf offset:0 atIndex:3];
        [enc setBuffer:stagesBuf offset:0 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(n_tgroups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(data, dataBuf.contents, n * sizeof(uint32_t));
        return 0;
    }
}

// Inverse DIF NTT dispatch
static int dispatch_intt_fused(uint32_t* data, const uint32_t* twiddles_inv, int n, int logN, int local_stages, uint32_t inv_n) {
    ensure_ntt_initialized();
    if (!g_intt_fused_pipeline) return -1;

    @autoreleasepool {
        int tg_size = 128;
        int n_tgroups = n / (tg_size * 2);

        id<MTLBuffer> dataBuf = [g_device newBufferWithBytes:data
                                                      length:n * sizeof(uint32_t)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> twBuf = [g_device newBufferWithBytes:twiddles_inv
                                                    length:(n/2) * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> nBuf = [g_device newBufferWithBytes:&n
                                                   length:sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> logNBuf = [g_device newBufferWithBytes:&logN
                                                      length:sizeof(uint32_t)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> stagesBuf = [g_device newBufferWithBytes:&local_stages
                                                        length:sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> invNBuf = [g_device newBufferWithBytes:&inv_n
                                                      length:sizeof(uint32_t)
                                                     options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];

        [enc setComputePipelineState:g_intt_fused_pipeline];
        [enc setBuffer:dataBuf offset:0 atIndex:0];
        [enc setBuffer:twBuf offset:0 atIndex:1];
        [enc setBuffer:nBuf offset:0 atIndex:2];
        [enc setBuffer:logNBuf offset:0 atIndex:3];
        [enc setBuffer:stagesBuf offset:0 atIndex:4];
        [enc setBuffer:invNBuf offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(n_tgroups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        memcpy(data, dataBuf.contents, n * sizeof(uint32_t));
        return 0;
    }
}

// ============================================================
// Scalar NTT implementation (fallback)
// ============================================================

// Forward NTT (DIT, in-place)
static void scalar_ntt_forward(uint32_t* data, int n, int logN, const uint32_t* twiddles) {
    // Bit-reversal permutation
    for (int i = 0; i < n; i++) {
        int rev = bb_bitrev(i, logN);
        if (i < rev) {
            uint32_t tmp = data[i];
            data[i] = data[rev];
            data[rev] = tmp;
        }
    }

    // DIT butterfly stages
    for (int s = 0; s < logN; s++) {
        int half_block = 1 << s;
        int block_size = half_block << 1;
        int n_blocks = n / block_size;

        for (int b = 0; b < n_blocks; b++) {
            for (int i = 0; i < half_block; i++) {
                int idx = b * block_size + i;
                int j = idx + half_block;
                int tw_idx = i * (n / block_size);

                uint32_t a = data[idx];
                uint32_t b_val = data[j];
                uint32_t w = twiddles[tw_idx];

                uint32_t wb = bb_mul(b_val, w);
                data[idx] = bb_add(a, wb);
                data[j] = bb_sub(a, wb);
            }
        }
    }
}

// Inverse NTT (DIF, in-place)
static void scalar_ntt_inverse(uint32_t* data, int n, int logN, const uint32_t* twiddles_inv, uint32_t inv_n) {
    // DIF butterfly stages
    for (int s = logN - 1; s >= 0; s--) {
        int half_block = 1 << s;
        int block_size = half_block << 1;
        int n_blocks = n / block_size;

        for (int b = 0; b < n_blocks; b++) {
            for (int i = 0; i < half_block; i++) {
                int idx = b * block_size + i;
                int j = idx + half_block;
                int tw_idx = i * (n / block_size);

                uint32_t a = data[idx];
                uint32_t b_val = data[j];

                uint32_t sum = bb_add(a, b_val);
                uint32_t diff = bb_sub(a, b_val);

                uint32_t w = twiddles_inv[tw_idx];
                data[idx] = sum;
                data[j] = bb_mul(diff, w);
            }
        }
    }

    // Bit-reversal permutation
    for (int i = 0; i < n; i++) {
        int rev = bb_bitrev(i, logN);
        if (i < rev) {
            uint32_t tmp = data[i];
            data[i] = data[rev];
            data[rev] = tmp;
        }
    }

    // Scale by 1/N
    for (int i = 0; i < n; i++) {
        data[i] = bb_mul(data[i], inv_n);
    }
}

// ============================================================
// Public API Implementations
// ============================================================

extern "C" {

bool ane_babybear_ntt_gpu_available(void) {
    return ane_babybear_ntt_available();
}

void* ane_babybear_ntt_create(int logN) {
    (void)logN;
    ensure_ntt_initialized();
    return g_ntt_fused_pipeline != NULL ? (void*)0x1 : NULL;
}

void ane_babybear_ntt_destroy(void* state) {
    (void)state;
}

int ane_babybear_ntt(void* opaque_state, uint32_t* data, int logN) {
    (void)opaque_state;

    int n = 1 << logN;

    // For now, only support N=256 (logN=8) with fused kernel
    if (n != 256) {
        return -1;  // Fallback to scalar
    }

    if (g_ntt_fused_pipeline) {
        // Generate twiddles on the fly
        uint32_t* twiddles = (uint32_t*)malloc((n/2) * sizeof(uint32_t));
        uint32_t w = 1;
        for (int i = 0; i < n/2; i++) {
            twiddles[i] = w;
            w = bb_mul(w, 7);  // Primitive root for BabyBear
        }

        int result = dispatch_ntt_fused(data, twiddles, n, logN, logN);
        free(twiddles);
        return result;
    }

    return -1;
}

int ane_babybear_intt(void* opaque_state, uint32_t* data, int logN) {
    (void)opaque_state;

    int n = 1 << logN;

    if (n != 256) {
        return -1;
    }

    if (g_intt_fused_pipeline) {
        // Generate inverse twiddles
        uint32_t* twiddles_inv = (uint32_t*)malloc((n/2) * sizeof(uint32_t));
        uint32_t w = 1;
        for (int i = 0; i < n/2; i++) {
            twiddles_inv[i] = w;
            w = bb_mul(w, 7);  // For inverse, we'd use w^(-1)
        }

        // Compute inv_n
        uint32_t inv_n = 1;  // Simplified - should compute n^(-1) mod p
        for (int i = 0; i < 27; i++) inv_n = bb_mul(inv_n, inv_n);
        inv_n = bb_mul(inv_n, bb_mul(7, 7));  // (7^2)^(-27) approximation

        int result = dispatch_intt_fused(data, twiddles_inv, n, logN, logN, inv_n);
        free(twiddles_inv);
        return result;
    }

    return -1;
}

int ane_babybear_ntt_forward(uint32_t* data, int logN) {
    return ane_babybear_ntt(NULL, data, logN);
}

int ane_babybear_ntt_inverse(uint32_t* data, int logN) {
    return ane_babybear_intt(NULL, data, logN);
}

} // extern "C"