// MetalLibLoader.m — Load precompiled .metallib and dispatch GPU compute kernels
//
// This bridges C FFI consumers to Metal GPU compute. It loads a pre-built
// zkmetal.metallib (no runtime shader compilation) and provides functions
// matching the MetalLibLoader.h API.
//
// Build: clang -framework Metal -framework Foundation -c MetalLibLoader.m

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "MetalLibLoader.h"
#include <string.h>

// ============================================================================
// Global state
// ============================================================================

static id<MTLDevice>          g_device      = nil;
static id<MTLLibrary>         g_library     = nil;
static id<MTLCommandQueue>    g_queue       = nil;
static NSArray<NSString*>*    g_kernelNames = nil;
static const char*            g_deviceName  = NULL;

// ============================================================================
// Initialization
// ============================================================================

ZkGpuStatus zkmetal_gpu_init(const char* metallib_path) {
    @autoreleasepool {
        // Get default Metal device
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            return ZK_GPU_ERR_NO_DEVICE;
        }

        // Load the precompiled .metallib
        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError* error = nil;
        NSURL* url = [NSURL fileURLWithPath:path];
        g_library = [g_device newLibraryWithURL:url error:&error];
        if (!g_library) {
            NSLog(@"zkmetal_gpu_init: failed to load %@: %@", path, error);
            g_device = nil;
            return ZK_GPU_ERR_METALLIB;
        }

        // Create command queue
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            g_library = nil;
            g_device = nil;
            return ZK_GPU_ERR_NO_DEVICE;
        }

        // Cache kernel function names
        g_kernelNames = [g_library functionNames];

        // Cache device name
        static char deviceNameBuf[256];
        strncpy(deviceNameBuf, [[g_device name] UTF8String], sizeof(deviceNameBuf) - 1);
        deviceNameBuf[sizeof(deviceNameBuf) - 1] = '\0';
        g_deviceName = deviceNameBuf;

        return ZK_GPU_SUCCESS;
    }
}

void zkmetal_gpu_shutdown(void) {
    @autoreleasepool {
        g_kernelNames = nil;
        g_queue = nil;
        g_library = nil;
        g_device = nil;
        g_deviceName = NULL;
    }
}

int32_t zkmetal_gpu_is_ready(void) {
    return (g_device != nil && g_library != nil && g_queue != nil) ? 1 : 0;
}

// ============================================================================
// Device info
// ============================================================================

const char* zkmetal_gpu_device_name(void) {
    return g_deviceName;
}

uint32_t zkmetal_gpu_kernel_count(void) {
    return g_kernelNames ? (uint32_t)[g_kernelNames count] : 0;
}

const char* zkmetal_gpu_kernel_name(uint32_t index) {
    if (!g_kernelNames || index >= [g_kernelNames count]) {
        return NULL;
    }
    // Return pointer into the NSString's UTF8 representation.
    // Valid as long as g_kernelNames is alive (until shutdown).
    return [g_kernelNames[index] UTF8String];
}

// ============================================================================
// Pipeline creation
// ============================================================================

void* zkmetal_gpu_create_pipeline(const char* kernel_name) {
    @autoreleasepool {
        if (!g_device || !g_library) return NULL;

        NSString* name = [NSString stringWithUTF8String:kernel_name];
        id<MTLFunction> func = [g_library newFunctionWithName:name];
        if (!func) return NULL;

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [g_device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) {
            NSLog(@"zkmetal: failed to create pipeline for %@: %@", name, error);
            return NULL;
        }

        // Retain the pipeline so it survives beyond this autorelease pool.
        // Caller must balance with zkmetal_gpu_destroy_pipeline().
        CFTypeRef ref = (__bridge CFTypeRef)pipeline;
        CFRetain(ref);
        return (void*)ref;
    }
}

void zkmetal_gpu_destroy_pipeline(void* pipeline) {
    if (pipeline) {
        CFRelease((CFTypeRef)pipeline);
    }
}

// ============================================================================
// Helper: synchronous GPU dispatch
// ============================================================================

/// Run a single compute kernel synchronously with the given buffers.
/// Buffers are passed as an array of {pointer, size} pairs.
typedef struct {
    const void* data;
    size_t      size;
    int         writable; // 0 = read-only, 1 = read-write
} ZkGpuBuffer;

static ZkGpuStatus dispatch_kernel(
    const char*        kernel_name,
    const ZkGpuBuffer* buffers,
    uint32_t           n_buffers,
    uint32_t           threadgroup_size,
    uint32_t           grid_size
) {
    @autoreleasepool {
        if (!zkmetal_gpu_is_ready()) return ZK_GPU_ERR_NOT_INIT;

        // Get pipeline
        NSString* name = [NSString stringWithUTF8String:kernel_name];
        id<MTLFunction> func = [g_library newFunctionWithName:name];
        if (!func) return ZK_GPU_ERR_KERNEL;

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [g_device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) return ZK_GPU_ERR_KERNEL;

        // Create command buffer
        id<MTLCommandBuffer> cmdBuf = [g_queue commandBuffer];
        if (!cmdBuf) return ZK_GPU_ERR_DISPATCH;

        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];

        // Bind buffers
        NSMutableArray<id<MTLBuffer>>* mtlBuffers = [NSMutableArray new];
        for (uint32_t i = 0; i < n_buffers; i++) {
            MTLResourceOptions opts = buffers[i].writable
                ? MTLResourceStorageModeShared
                : (MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache);

            id<MTLBuffer> buf;
            if (buffers[i].writable) {
                buf = [g_device newBufferWithBytes:buffers[i].data
                                            length:buffers[i].size
                                           options:opts];
            } else {
                buf = [g_device newBufferWithBytes:buffers[i].data
                                            length:buffers[i].size
                                           options:opts];
            }
            if (!buf) {
                [encoder endEncoding];
                return ZK_GPU_ERR_BUFFER;
            }
            [encoder setBuffer:buf offset:0 atIndex:i];
            [mtlBuffers addObject:buf];
        }

        // Dispatch
        MTLSize tgSize = MTLSizeMake(threadgroup_size, 1, 1);
        MTLSize gridSz = MTLSizeMake(grid_size, 1, 1);
        [encoder dispatchThreads:gridSz threadsPerThreadgroup:tgSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if ([cmdBuf error]) {
            NSLog(@"zkmetal: GPU error in %@: %@", name, [cmdBuf error]);
            return ZK_GPU_ERR_DISPATCH;
        }

        // Copy back writable buffers
        for (uint32_t i = 0; i < n_buffers; i++) {
            if (buffers[i].writable) {
                memcpy((void*)buffers[i].data,
                       [mtlBuffers[i] contents],
                       buffers[i].size);
            }
        }

        return ZK_GPU_SUCCESS;
    }
}

// ============================================================================
// BN254 MSM
// ============================================================================

ZkGpuStatus zkmetal_gpu_msm_bn254(
    const uint8_t* scalars,
    const uint8_t* points,
    uint32_t n_points,
    uint8_t* result_x,
    uint8_t* result_y,
    uint8_t* result_z
) {
    // NOTE: Full MSM is a multi-phase GPU pipeline (signed-digit extraction,
    // radix sort, bucket accumulation, bucket sum, Horner combine).
    // This stub demonstrates the loader pattern. A complete implementation
    // would orchestrate multiple kernel dispatches with intermediate buffers,
    // matching the logic in Sources/zkMetal/MSM/MSMEngine.swift.
    //
    // For production use, prefer the full zkMetal-ffi library which handles
    // the complete MSM pipeline. This loader is primarily for:
    // 1. Running individual kernels directly
    // 2. Building custom pipelines from the available kernels
    // 3. Verifying the metallib loaded correctly
    (void)scalars; (void)points; (void)n_points;
    (void)result_x; (void)result_y; (void)result_z;

    if (!zkmetal_gpu_is_ready()) return ZK_GPU_ERR_NOT_INIT;

    // Verify the required kernels exist
    id<MTLFunction> fn = [g_library newFunctionWithName:@"msm_reduce_sorted_buckets"];
    if (!fn) return ZK_GPU_ERR_KERNEL;

    // Full pipeline would go here — for now, return success if kernels are available
    return ZK_GPU_ERR_INVALID_INPUT; // Not yet implemented as a complete pipeline
}

// ============================================================================
// BN254 NTT
// ============================================================================

ZkGpuStatus zkmetal_gpu_ntt_bn254(uint8_t* data, uint32_t log_n) {
    (void)data; (void)log_n;
    if (!zkmetal_gpu_is_ready()) return ZK_GPU_ERR_NOT_INIT;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"ntt_butterfly"];
    if (!fn) return ZK_GPU_ERR_KERNEL;
    return ZK_GPU_ERR_INVALID_INPUT; // Stub — see note in MSM above
}

ZkGpuStatus zkmetal_gpu_intt_bn254(uint8_t* data, uint32_t log_n) {
    (void)data; (void)log_n;
    if (!zkmetal_gpu_is_ready()) return ZK_GPU_ERR_NOT_INIT;
    id<MTLFunction> fn = [g_library newFunctionWithName:@"intt_butterfly"];
    if (!fn) return ZK_GPU_ERR_KERNEL;
    return ZK_GPU_ERR_INVALID_INPUT;
}

// ============================================================================
// Poseidon2
// ============================================================================

ZkGpuStatus zkmetal_gpu_poseidon2_hash_pairs(
    const uint8_t* input, uint32_t n_pairs, uint8_t* output
) {
    if (!zkmetal_gpu_is_ready()) return ZK_GPU_ERR_NOT_INIT;
    if (!input || !output || n_pairs == 0) return ZK_GPU_ERR_INVALID_INPUT;

    ZkGpuBuffer bufs[2] = {
        { .data = input,  .size = (size_t)n_pairs * 2 * 32, .writable = 0 },
        { .data = output, .size = (size_t)n_pairs * 32,     .writable = 1 },
    };

    return dispatch_kernel(
        "poseidon2_hash_pairs",
        bufs, 2,
        /* threadgroup */ 256,
        /* grid */        n_pairs
    );
}

// ============================================================================
// Keccak-256
// ============================================================================

ZkGpuStatus zkmetal_gpu_keccak256(
    const uint8_t* input, uint32_t n_inputs, uint8_t* output
) {
    if (!zkmetal_gpu_is_ready()) return ZK_GPU_ERR_NOT_INIT;
    if (!input || !output || n_inputs == 0) return ZK_GPU_ERR_INVALID_INPUT;

    ZkGpuBuffer bufs[2] = {
        { .data = input,  .size = (size_t)n_inputs * 64, .writable = 0 },
        { .data = output, .size = (size_t)n_inputs * 32, .writable = 1 },
    };

    return dispatch_kernel(
        "keccak256_hash_batch",
        bufs, 2,
        /* threadgroup */ 256,
        /* grid */        n_inputs
    );
}
