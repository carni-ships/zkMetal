/**
 * @file metal_msm.mm
 * @brief Metal GPU MSM implementation for BN254 using Objective-C++ Metal API
 *
 * Ports the Swift MetalMSM class from zkmsm to C++ for integration with
 * Barretenberg's proving pipeline. Uses Objective-C++ (.mm) for direct
 * Metal API access without needing the metal-cpp header-only library.
 */

#if defined(__APPLE__) && !defined(__EMSCRIPTEN__)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <mutex>
#include <future>
#include <dispatch/dispatch.h>
#include <CoreFoundation/CFDate.h>

#include "metal_msm.hpp"
#include "barretenberg/common/assert.hpp"

namespace bb::scalar_multiplication::metal {

// ======================== GPU-side struct layouts ========================
// Must match the Metal shader's struct definitions exactly.

struct alignas(4) GpuFp {
    uint32_t v[8]; // 8x32-bit limbs, little-endian, Montgomery form
};

struct GpuPointAffine {
    GpuFp x;
    GpuFp y;
};

struct GpuPointProjective {
    GpuFp x;
    GpuFp y;
    GpuFp z;
};

struct MsmParams {
    uint32_t n_points;
    uint32_t window_bits;
    uint32_t n_buckets;
};

// ======================== Singleton Metal Context ========================

class MetalMSMContext {
  public:
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLComputePipelineState> reduce_sorted_fn;
    id<MTLComputePipelineState> reduce_parallel_fn;
    id<MTLComputePipelineState> reduce_half_fn;
    id<MTLComputePipelineState> combine_halves_fn;
    id<MTLComputePipelineState> reduce_quarter_fn;
    id<MTLComputePipelineState> combine_quarter_fn;
    id<MTLComputePipelineState> reduce_large_bucket_fn;
    id<MTLComputePipelineState> bucket_sum_direct_fn;
    id<MTLComputePipelineState> combine_segments_fn;
    id<MTLComputePipelineState> combine_segments_l1_fn; // 2-level combine L1 pass
    id<MTLComputePipelineState> glv_decompose_fn;
    id<MTLComputePipelineState> glv_endomorphism_fn;
    id<MTLComputePipelineState> glv_precompute_cache_fn;
    id<MTLComputePipelineState> glv_apply_neg_flags_fn;
    id<MTLComputePipelineState> sort_histogram_fn;
    id<MTLComputePipelineState> sort_prefix_sum_fn;
    id<MTLComputePipelineState> sort_scatter_fn;
    id<MTLComputePipelineState> gather_sorted_fn;
    id<MTLComputePipelineState> reduce_gathered_fn;
    id<MTLComputePipelineState> reduce_gathered_win_fn;
    id<MTLComputePipelineState> compute_csm_fn;
    bool initialized = false;
    std::once_flag init_flag;
    bool init_result = false;

    // Pre-allocated buffers
    id<MTLBuffer> points_buffer = nil;
    id<MTLBuffer> sorted_indices_buffer = nil;
    id<MTLBuffer> all_offsets_buffer = nil;
    id<MTLBuffer> all_counts_buffer = nil;
    id<MTLBuffer> buckets_buffer = nil;
    id<MTLBuffer> partial_results_buffer = nil;  // 2x bucket count for half-reduce
    id<MTLBuffer> segment_results_buffer = nil;
    id<MTLBuffer> chunk_results_buffer = nil;    // intermediate for 2-level combine
    id<MTLBuffer> window_results_buffer = nil;
    id<MTLBuffer> count_sorted_map_buffer = nil;
    id<MTLBuffer> large_bucket_ids_buffer = nil;
    id<MTLBuffer> scatter_pos_buffer = nil;
    // gathered_points_buffer tested and removed: reduce is compute-bound (81ms vs 83ms sequential),
    // random access only adds ~2ms. Gather-in-scatter was net slower due to 14ms sort overhead.

    // GPU CSM output buffers (for compute_csm_fn kernel)
    id<MTLBuffer> n_large_buckets_out_buffer = nil;  // atomic uint: count of large buckets
    id<MTLBuffer> imbalance_flag_buffer = nil;        // atomic uint: set to 1 if imbalanced

    // Elastic MSM: pair match counting buffer

    // GLV buffers
    id<MTLBuffer> scalars_buffer = nil;
    id<MTLBuffer> k1_buffer = nil;
    id<MTLBuffer> k2_buffer = nil;
    id<MTLBuffer> neg1_buffer = nil;
    id<MTLBuffer> neg2_buffer = nil;

    // SRS cache: persistent buffer for SRS points, avoids re-copying identical data
    id<MTLBuffer> srs_buffer = nil;
    const void* cached_srs_ptr = nullptr;
    size_t cached_srs_count = 0;

    // Endomorphism cache: precomputed reduced + beta*x points (2n entries)
    // Avoids fp_mul in GLV endomorphism for repeated SRS
    id<MTLBuffer> endo_cache_buffer = nil;
    const void* cached_endo_srs_ptr = nullptr;
    size_t cached_endo_count = 0;

    // Pre-allocated CPU buffers (avoids per-MSM heap allocations)
    // non_mont_scalars_buf removed: fused copy+convert in run_msm eliminates need for intermediate buffer
    std::vector<std::vector<uint32_t>> sort_counts_buf;
    std::vector<std::vector<uint32_t>> sort_positions_buf;

    size_t max_points = 0;
    size_t max_buckets = 0;
    size_t max_windows = 0;
    size_t max_segments = 0;

    static MetalMSMContext& instance()
    {
        static MetalMSMContext ctx;
        return ctx;
    }

    bool init_if_needed()
    {
        std::call_once(init_flag, [this]() {
            @autoreleasepool {
                device = MTLCreateSystemDefaultDevice();
                if (!device) {
                    fprintf(stderr, "[Metal GPU] No Metal device found. GPU MSM disabled.\n");
                    return;
                }

                command_queue = [device newCommandQueue];
                if (!command_queue) {
                    fprintf(stderr, "[Metal GPU] Failed to create command queue. GPU MSM disabled.\n");
                    return;
                }

                // Try pre-compiled metallib first (~1ms), fall back to source compilation (~5ms cached)
                NSError* error = nil;
                id<MTLLibrary> library = load_precompiled_library();
                if (!library) {
                    NSString* shader_source = load_shader_source();
                    if (!shader_source) {
                        fprintf(stderr, "[Metal GPU] Failed to load shader source. GPU MSM disabled.\n");
                        return;
                    }
                    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                    options.fastMathEnabled = YES;
                    library = [device newLibraryWithSource:shader_source options:options error:&error];
                    if (!library) {
                        fprintf(stderr, "[Metal GPU] Shader compilation failed: %s. GPU MSM disabled.\n",
                                error ? [[error localizedDescription] UTF8String] : "unknown error");
                        return;
                    }
                }

                // Load all kernel functions with explicit error reporting.
                // A missing kernel must never silently disable all GPU MSM.
                auto load_fn = [&](const char* name) -> id<MTLFunction> {
                    id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
                    if (!fn) {
                        fprintf(stderr, "[Metal GPU] FATAL: kernel '%s' not found in shader. "
                                "GPU MSM disabled — all proving will use CPU fallback.\n", name);
                    }
                    return fn;
                };

                id<MTLFunction> reduce_fn        = load_fn("msm_reduce_sorted_buckets");
                id<MTLFunction> reduce_par_fn     = load_fn("msm_reduce_parallel");
                id<MTLFunction> reduce_half_fn_obj = load_fn("msm_reduce_sorted_buckets_half");
                id<MTLFunction> combine_halves_fn_obj = load_fn("msm_combine_bucket_halves");
                id<MTLFunction> reduce_quarter_fn_obj = load_fn("msm_reduce_sorted_buckets_quarter");
                id<MTLFunction> combine_quarter_fn_obj = load_fn("msm_combine_quarter_results");
                id<MTLFunction> reduce_large_fn  = load_fn("msm_reduce_large_bucket");
                id<MTLFunction> sum_fn            = load_fn("msm_bucket_sum_direct");
                id<MTLFunction> combine_fn        = load_fn("msm_combine_segments");
                id<MTLFunction> combine_l1_fn     = load_fn("msm_combine_segments_l1");
                id<MTLFunction> glv_dec_fn        = load_fn("glv_decompose");
                id<MTLFunction> glv_endo_fn       = load_fn("glv_endomorphism");
                id<MTLFunction> glv_precomp_fn    = load_fn("glv_precompute_cache");
                id<MTLFunction> glv_neg_fn        = load_fn("glv_apply_neg_flags");
                id<MTLFunction> sort_hist_fn      = load_fn("msm_sort_histogram");
                id<MTLFunction> sort_pfx_fn       = load_fn("msm_sort_prefix_sum");
                id<MTLFunction> sort_scat_fn      = load_fn("msm_sort_scatter");
                id<MTLFunction> gather_fn         = load_fn("msm_gather_sorted_points");
                id<MTLFunction> reduce_gath_fn    = load_fn("msm_reduce_gathered");
                id<MTLFunction> reduce_gath_win_fn = load_fn("msm_reduce_gathered_single_window");
                id<MTLFunction> compute_csm_fn_obj = load_fn("msm_compute_csm");
                if (!reduce_fn || !reduce_par_fn || !reduce_half_fn_obj || !combine_halves_fn_obj ||
                    !reduce_quarter_fn_obj || !combine_quarter_fn_obj ||
                    !reduce_large_fn || !sum_fn ||
                    !combine_fn || !combine_l1_fn || !glv_dec_fn || !glv_endo_fn ||
                    !glv_precomp_fn || !glv_neg_fn ||
                    !sort_hist_fn || !sort_pfx_fn || !sort_scat_fn ||
                    !gather_fn || !reduce_gath_fn || !reduce_gath_win_fn ||
                    !compute_csm_fn_obj) {
                    return;
                }

                auto make_pipeline = [&](const char* name, id<MTLFunction> fn) -> id<MTLComputePipelineState> {
                    id<MTLComputePipelineState> ps = [device newComputePipelineStateWithFunction:fn error:&error];
                    if (!ps) {
                        fprintf(stderr, "[Metal GPU] FATAL: pipeline creation failed for '%s': %s. "
                                "GPU MSM disabled.\n", name,
                                error ? [[error localizedDescription] UTF8String] : "unknown error");
                    }
                    return ps;
                };

                reduce_sorted_fn     = make_pipeline("msm_reduce_sorted_buckets", reduce_fn);
                reduce_parallel_fn   = make_pipeline("msm_reduce_parallel", reduce_par_fn);
                reduce_half_fn       = make_pipeline("msm_reduce_sorted_buckets_half", reduce_half_fn_obj);
                combine_halves_fn    = make_pipeline("msm_combine_bucket_halves", combine_halves_fn_obj);
                reduce_quarter_fn    = make_pipeline("msm_reduce_sorted_buckets_quarter", reduce_quarter_fn_obj);
                combine_quarter_fn   = make_pipeline("msm_combine_quarter_results", combine_quarter_fn_obj);
                reduce_large_bucket_fn = make_pipeline("msm_reduce_large_bucket", reduce_large_fn);
                bucket_sum_direct_fn = make_pipeline("msm_bucket_sum_direct", sum_fn);
                combine_segments_fn  = make_pipeline("msm_combine_segments", combine_fn);
                combine_segments_l1_fn = make_pipeline("msm_combine_segments_l1", combine_l1_fn);
                glv_decompose_fn     = make_pipeline("glv_decompose", glv_dec_fn);
                glv_endomorphism_fn  = make_pipeline("glv_endomorphism", glv_endo_fn);
                glv_precompute_cache_fn = make_pipeline("glv_precompute_cache", glv_precomp_fn);
                glv_apply_neg_flags_fn = make_pipeline("glv_apply_neg_flags", glv_neg_fn);
                sort_histogram_fn    = make_pipeline("msm_sort_histogram", sort_hist_fn);
                sort_prefix_sum_fn   = make_pipeline("msm_sort_prefix_sum", sort_pfx_fn);
                sort_scatter_fn      = make_pipeline("msm_sort_scatter", sort_scat_fn);
                gather_sorted_fn     = make_pipeline("msm_gather_sorted_points", gather_fn);
                reduce_gathered_fn   = make_pipeline("msm_reduce_gathered", reduce_gath_fn);
                reduce_gathered_win_fn = make_pipeline("msm_reduce_gathered_single_window", reduce_gath_win_fn);
                compute_csm_fn       = make_pipeline("msm_compute_csm", compute_csm_fn_obj);

                if (!reduce_sorted_fn || !reduce_parallel_fn || !reduce_half_fn || !combine_halves_fn ||
                    !reduce_quarter_fn || !combine_quarter_fn ||
                    !reduce_large_bucket_fn || !bucket_sum_direct_fn || !combine_segments_fn || !combine_segments_l1_fn ||
                    !glv_decompose_fn || !glv_endomorphism_fn ||
                    !glv_precompute_cache_fn || !glv_apply_neg_flags_fn ||
                    !sort_histogram_fn || !sort_prefix_sum_fn || !sort_scatter_fn ||
                    !gather_sorted_fn || !reduce_gathered_fn || !reduce_gathered_win_fn ||
                    !compute_csm_fn) {
                    return;
                }

                initialized = true;
                init_result = true;
            } // @autoreleasepool
        }); // call_once
        return init_result;
    }

    // Try loading pre-compiled .metallib (fast: ~1ms) before falling back to source compilation (~25ms)
    id<MTLLibrary> load_precompiled_library()
    {
        NSArray* metallib_paths = @[
            [[[NSProcessInfo processInfo].arguments[0] stringByDeletingLastPathComponent]
                stringByAppendingPathComponent:@"bn254.metallib"],
            @"bn254.metallib",
        ];

        const char* env_path = getenv("BB_METAL_SHADER_PATH");
        if (env_path) {
            NSString* path = [[NSString stringWithUTF8String:env_path] stringByReplacingOccurrencesOfString:@".metal" withString:@".metallib"];
            metallib_paths = [metallib_paths arrayByAddingObject:path];
        }

        for (NSString* path in metallib_paths) {
            NSError* error = nil;
            NSURL* url = [NSURL fileURLWithPath:path];
            id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
            if (lib) {
                return lib;
            }
        }
        return nil;
    }

    NSString* load_shader_source()
    {
        NSArray* search_paths = @[
            [[[NSProcessInfo processInfo].arguments[0] stringByDeletingLastPathComponent]
                stringByAppendingPathComponent:@"bn254.metal"],
            @"bn254.metal",
        ];

        for (NSString* path in search_paths) {
            NSString* source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
            if (source) {
                return source;
            }
        }

        const char* env_path = getenv("BB_METAL_SHADER_PATH");
        if (env_path) {
            NSString* path = [NSString stringWithUTF8String:env_path];
            NSString* source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
            if (source) {
                return source;
            }
        }

        return nil;
    }

    void ensure_buffers(size_t n, size_t n_buckets, size_t n_segments, size_t n_windows)
    {
        bool need_realloc = n > max_points || n_buckets > max_buckets || n_windows > max_windows ||
                            n_segments > max_segments;
        if (!need_realloc) {
            return;
        }

        size_t np = std::max(n, max_points);
        size_t nb = std::max(n_buckets, max_buckets);
        size_t nw = std::max(n_windows, max_windows);
        size_t ns = std::max(n_segments, max_segments);

        // GLV doubles the number of points (original + endomorphism)
        size_t np2 = np * 2;

        points_buffer = [device newBufferWithLength:sizeof(GpuPointAffine) * np2
                                           options:MTLResourceStorageModeShared];
        sorted_indices_buffer = [device newBufferWithLength:sizeof(uint32_t) * np2 * nw
                                                   options:MTLResourceStorageModeShared];
        all_offsets_buffer = [device newBufferWithLength:sizeof(uint32_t) * nb * nw
                                                options:MTLResourceStorageModeShared];
        all_counts_buffer = [device newBufferWithLength:sizeof(uint32_t) * nb * nw
                                               options:MTLResourceStorageModeShared];
        buckets_buffer = [device newBufferWithLength:sizeof(GpuPointProjective) * nb * nw
                                            options:MTLResourceStorageModeShared];
        partial_results_buffer = [device newBufferWithLength:sizeof(GpuPointProjective) * 2 * nb * nw
                                                    options:MTLResourceStorageModeShared];
        segment_results_buffer = [device newBufferWithLength:sizeof(GpuPointProjective) * ns * nw
                                                    options:MTLResourceStorageModeShared];
        // chunk_results: for 2-level combine, ceil(n_segments/256) chunks × n_windows
        {
            size_t n_chunks = (ns + 255) / 256;
            chunk_results_buffer = [device newBufferWithLength:sizeof(GpuPointProjective) * n_chunks * nw
                                                       options:MTLResourceStorageModeShared];
        }
        window_results_buffer = [device newBufferWithLength:sizeof(GpuPointProjective) * nw
                                                   options:MTLResourceStorageModeShared];
        count_sorted_map_buffer = [device newBufferWithLength:sizeof(uint32_t) * nb * nw
                                                      options:MTLResourceStorageModeShared];
        // Large bucket IDs: allocate for worst case (all buckets across all windows).
        // At large n (e.g. 4M points, avg 256/bucket), half the buckets can exceed
        // the threshold, producing ~150K large bucket IDs. Previous 8192 cap caused
        // out-of-bounds writes and GPU memory corruption.
        size_t large_buf_size = nb * nw;
        large_bucket_ids_buffer = [device newBufferWithLength:sizeof(uint32_t) * large_buf_size
                                                      options:MTLResourceStorageModeShared];
        scatter_pos_buffer = [device newBufferWithLength:sizeof(uint32_t) * nb * nw
                                                 options:MTLResourceStorageModeShared];
        // gathered_points_buffer removed: reduce is compute-bound, not memory-bound
        // GPU CSM output buffers
        // n_large_buckets_out_buffer: 3 × uint32 formatted as Metal indirect dispatch args {count, 1, 1}
        // The CSM kernel atomically increments [0]; [1] and [2] stay at 1 for indirect dispatch.
        n_large_buckets_out_buffer = [device newBufferWithLength:3 * sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
        {
            auto* args = static_cast<uint32_t*>([n_large_buckets_out_buffer contents]);
            args[0] = 0;
            args[1] = 1;
            args[2] = 1;
        }
        imbalance_flag_buffer = [device newBufferWithLength:sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];
        // sorted_points_buffer removed: fused gather+reduce reads sorted_indices + points directly

        // GLV buffers
        scalars_buffer = [device newBufferWithLength:sizeof(uint32_t) * 8 * np
                                             options:MTLResourceStorageModeShared];
        k1_buffer = [device newBufferWithLength:sizeof(uint32_t) * 8 * np
                                        options:MTLResourceStorageModeShared];
        k2_buffer = [device newBufferWithLength:sizeof(uint32_t) * 8 * np
                                        options:MTLResourceStorageModeShared];
        neg1_buffer = [device newBufferWithLength:sizeof(uint8_t) * np
                                          options:MTLResourceStorageModeShared];
        neg2_buffer = [device newBufferWithLength:sizeof(uint8_t) * np
                                          options:MTLResourceStorageModeShared];

        // Pre-fill CSM with identity mapping (csm[i] = i) so we can skip the CPU
        // CSM computation and merge Phase1+Phase3 into a single command buffer.
        // Identity CSM is optimal for uniform random scalars (the common case).
        {
            auto* csm = static_cast<uint32_t*>([count_sorted_map_buffer contents]);
            size_t total = nb * nw;
            for (size_t i = 0; i < total; i++) {
                csm[i] = static_cast<uint32_t>(i);
            }
        }

        max_points = np;
        max_buckets = nb;
        max_windows = nw;
        max_segments = ns;
    }

  public:
    // Pre-allocate GPU buffers and optionally cache SRS points to avoid first-MSM overhead.
    void prewarm(size_t n, const curve::BN254::AffineElement* srs_points = nullptr)
    {
        if (!init_if_needed())
            return;
        // Signed-digit GLV MSM with 16-bit windows: shader uses carry propagation
        const size_t n_buckets = (size_t(1) << 15) + 1; // 32769 (half + 1 for signed digits)
        const size_t n_windows = (128 + 15) / 16 + 1;   // 9 (8 data + 1 carry overflow)
        const size_t n_segments = 1024; // 2-level combine supports >256 segments
        ensure_buffers(n, n_buckets, n_segments, n_windows);

        // Pre-cache SRS on GPU to avoid copy on first MSM.
        // Skip if same pointer and count — SRS doesn't change between proofs.
        if (srs_points && n > 0 &&
            !(cached_srs_ptr == srs_points && cached_srs_count == n && srs_buffer)) {
            if (!srs_buffer || [srs_buffer length] < sizeof(GpuPointAffine) * n) {
                srs_buffer = [device newBufferWithLength:sizeof(GpuPointAffine) * n
                                                options:MTLResourceStorageModeShared];
            }
            std::memcpy(static_cast<GpuPointAffine*>([srs_buffer contents]),
                        srs_points, sizeof(GpuPointAffine) * n);
            cached_srs_ptr = srs_points;
            cached_srs_count = n;
        }

        // Warm the Metal command queue and pipeline states by dispatching a trivial kernel.
        // Without this, the first real MSM pays ~150ms of Metal runtime initialization.
        {
            id<MTLCommandBuffer> cb = [command_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:glv_decompose_fn];
            [enc setBuffer:scalars_buffer offset:0 atIndex:0];
            [enc setBuffer:k1_buffer offset:0 atIndex:1];
            [enc setBuffer:k2_buffer offset:0 atIndex:2];
            [enc setBuffer:neg1_buffer offset:0 atIndex:3];
            [enc setBuffer:neg2_buffer offset:0 atIndex:4];
            uint32_t one = 1;
            [enc setBytes:&one length:sizeof(uint32_t) atIndex:5];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
            [cb commit];
            // Don't wait — Metal command buffers execute in queue order, so the first real MSM
            // will naturally wait for this trivial dispatch to finish.
        }
    }

    /**
     * @brief Run MSM on Metal GPU
     *
     * @param points BN254 affine points (bb::g1::affine_element format, 4x64-bit Montgomery limbs per coord)
     * @param scalars BN254 scalar field elements (4x64-bit, NON-Montgomery form)
     * @param num_points Number of point-scalar pairs
     * @return GpuPointProjective result (to be converted back to bb::g1::element)
     */
    bool run_msm(const curve::BN254::AffineElement* points,
                 const curve::BN254::ScalarField* scalars,
                 size_t num_points,
                 curve::BN254::Element& result_out,
                 bool /*skip_imbalance_check*/ = false,
                 bool from_montgomery = false)
    {
        @autoreleasepool {
            const size_t n = num_points;
            // Window size for GLV 128-bit scalars. With 2n points after GLV doubling,
            // optimal window trades bucket count vs passes over data.
            // w=16: 32769 buckets, 9 windows (8 data + 1 carry), ~26 pts/bucket average for 428K pts.
            // NOTE: wire polynomials have repeated scalars (max_count ~35K), causing 1043 large buckets
            // regardless of window_bits. Changing w=16→17 showed no improvement.
            uint32_t window_bits;
            if (n <= 256)
                window_bits = 8;
            else if (n <= 4096)
                window_bits = 10;
            else if (n <= 32768)
                window_bits = 12;
            else
                window_bits = 16;

            // Signed-digit MSM: shader converts digits to [-half, +half] with carry propagation.
            // Shader uses n_data_windows = n_windows - 1 for data; last window is carry overflow.
            // GLV scalars are ~127 bits so the carry window is always empty in practice.
            const size_t n_windows = (128 + window_bits - 1) / window_bits + 1; // 9 (8 data + 1 carry)
            const size_t n_buckets = (size_t(1) << (window_bits - 1)) + 1;
            // For large n, use 1024 segments for better GPU occupancy in bucket_sum.
            // 2-level combine handles >256 segments: L1 reduces 256-chunk groups, L2 reduces chunks.
            const size_t n_segments = (n > 32768) ? 1024 : std::min(size_t(256), std::max(size_t(1), n_buckets / 2));
            // GLV: MSM operates on 2n points (original + endomorphism)
            const size_t n2 = n * 2;

            ensure_buffers(n, n_buckets, n_segments, n_windows);

            // ---- Phase 0: Copy scalars + cache SRS points ----
            static_assert(sizeof(curve::BN254::AffineElement) == sizeof(GpuPointAffine),
                          "AffineElement and GpuPointAffine must have same size");

            // SRS point caching: skip copy if same SRS data and sufficient size.
            // When pointer changes (e.g. ZK tail commits with start_index>0), only copy n points
            // to avoid reading past end of the subspan.
            if (points != cached_srs_ptr || n > cached_srs_count) {
                size_t copy_count = (points == cached_srs_ptr) ? std::max(n, cached_srs_count) : n;
                if (!srs_buffer || [srs_buffer length] < sizeof(GpuPointAffine) * copy_count) {
                    srs_buffer = [device newBufferWithLength:sizeof(GpuPointAffine) * copy_count
                                                    options:MTLResourceStorageModeShared];
                }
                std::memcpy(static_cast<GpuPointAffine*>([srs_buffer contents]),
                            points, sizeof(GpuPointAffine) * copy_count);
                cached_srs_ptr = points;
                cached_srs_count = copy_count;
            }

            // Copy scalars to GPU buffer. When from_montgomery=true, convert during copy
            // to avoid a separate intermediate buffer (fuses two passes into one).
            auto* gpu_scalars_dst = static_cast<curve::BN254::ScalarField*>([scalars_buffer contents]);
            if (from_montgomery) {
                const size_t chunk_size = 8192;
                size_t n_chunks = (n + chunk_size - 1) / chunk_size;
                dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t chunk) {
                    size_t begin = chunk * chunk_size;
                    size_t end = std::min(begin + chunk_size, n);
                    for (size_t i = begin; i < end; i++) {
                        gpu_scalars_dst[i] = scalars[i];
                        gpu_scalars_dst[i].self_from_montgomery_form();
                    }
                });
            } else {
                std::memcpy(gpu_scalars_dst, scalars, sizeof(curve::BN254::ScalarField) * n);
            }
            // Phase 1+2: GPU GLV + counting sort + CSM
            // Merging GLV, sort, and count-sorted mapping into one command buffer
            // eliminates CPU/GPU round-trips. Metal auto-inserts barriers between
            // compute encoders for data dependencies.
            {
                static int gpu_profile_level = []() {
                    const char* env = std::getenv("BB_GPU_PROFILE");
                    if (!env) return 0;
                    int v = atoi(env);
                    return (v >= 2) ? 2 : 1;
                }();
                bool gpu_profile = (gpu_profile_level >= 1);
                bool gpu_profile_phases = (gpu_profile_level >= 2);
                auto gpu_time = [](const char* /*label*/) -> double { return CFAbsoluteTimeGetCurrent(); };

                // Per-phase timing: commit current CB, record GPU time, start new CB
                struct PhaseTime { const char* name; double gpu_ms; };
                std::vector<PhaseTime> phase_times;
                auto phase_barrier = [&](id<MTLCommandBuffer> __strong& cb, const char* phase_name) {
                    if (!gpu_profile_phases) return;
                    [cb commit];
                    [cb waitUntilCompleted];
                    if ([cb error]) {
                        fprintf(stderr, "[GPU_ERR] %s: %s\n", phase_name, [[cb error] localizedDescription].UTF8String);
                    }
                    double gs = [cb GPUStartTime];
                    double ge = [cb GPUEndTime];
                    phase_times.push_back({phase_name, (ge - gs) * 1000.0});
                    cb = [command_queue commandBuffer];
                };

                uint32_t n_u32 = static_cast<uint32_t>(n);
                uint32_t wb_u32 = static_cast<uint32_t>(window_bits);
                uint32_t nb_u32 = static_cast<uint32_t>(n_buckets);
                uint32_t nw_u32 = static_cast<uint32_t>(n_windows);
                uint32_t n2_u32 = static_cast<uint32_t>(n2);
                MTLSize grid_n = MTLSizeMake(n, 1, 1);
                MTLSize tg_n = MTLSizeMake(std::min(size_t(256), n), 1, 1);

                // All GLV + sort kernels in a single command buffer.
                // Metal auto-inserts barriers between compute encoders for data dependencies.
                // Keeping in one CB allows GPU to pipeline kernels without CPU roundtrip overhead.
                id<MTLCommandBuffer> cb_prep = [command_queue commandBuffer];

                // GLV Kernel 1: Decompose scalars → k1, k2, neg1, neg2
                id<MTLComputeCommandEncoder> enc_dec = [cb_prep computeCommandEncoder];
                [enc_dec setComputePipelineState:glv_decompose_fn];
                [enc_dec setBuffer:scalars_buffer offset:0 atIndex:0];
                [enc_dec setBuffer:k1_buffer offset:0 atIndex:1];
                [enc_dec setBuffer:k2_buffer offset:0 atIndex:2];
                [enc_dec setBuffer:neg1_buffer offset:0 atIndex:3];
                [enc_dec setBuffer:neg2_buffer offset:0 atIndex:4];
                [enc_dec setBytes:&n_u32 length:sizeof(uint32_t) atIndex:5];
                [enc_dec dispatchThreads:grid_n threadsPerThreadgroup:tg_n];
                [enc_dec endEncoding];

                // GLV Kernel 2: Endomorphism (cached or full).
                // Cache layout: [original[0..n-1] | endo[0..n-1]], stride = n.
                // Cache is valid only for the SAME n: if n changes, the endo stride changes,
                // so we must rebuild. Strict equality check ensures correctness.
                bool endo_cached = (points == cached_endo_srs_ptr && n == cached_endo_count);
                if (!endo_cached) {
                    // First time for this SRS or n changed: precompute cache
                    if (!endo_cache_buffer || [endo_cache_buffer length] < sizeof(GpuPointAffine) * 2 * n) {
                        endo_cache_buffer = [device newBufferWithLength:sizeof(GpuPointAffine) * 2 * n
                                                               options:MTLResourceStorageModeShared];
                    }
                    id<MTLComputeCommandEncoder> enc_precomp = [cb_prep computeCommandEncoder];
                    [enc_precomp setComputePipelineState:glv_precompute_cache_fn];
                    [enc_precomp setBuffer:srs_buffer offset:0 atIndex:0];
                    [enc_precomp setBuffer:endo_cache_buffer offset:0 atIndex:1];
                    [enc_precomp setBytes:&n_u32 length:sizeof(uint32_t) atIndex:2];
                    [enc_precomp dispatchThreads:grid_n threadsPerThreadgroup:tg_n];
                    [enc_precomp endEncoding];
                    cached_endo_srs_ptr = points;
                    cached_endo_count = n;
                }
                // Apply neg flags from cache (always needed — neg flags change per MSM)
                id<MTLComputeCommandEncoder> enc_neg = [cb_prep computeCommandEncoder];
                [enc_neg setComputePipelineState:glv_apply_neg_flags_fn];
                [enc_neg setBuffer:endo_cache_buffer offset:0 atIndex:0];
                [enc_neg setBuffer:points_buffer offset:0 atIndex:1];
                [enc_neg setBuffer:neg1_buffer offset:0 atIndex:2];
                [enc_neg setBuffer:neg2_buffer offset:0 atIndex:3];
                [enc_neg setBytes:&n_u32 length:sizeof(uint32_t) atIndex:4];
                [enc_neg dispatchThreads:grid_n threadsPerThreadgroup:tg_n];
                [enc_neg endEncoding];

                phase_barrier(cb_prep, "glv");

                id<MTLBlitCommandEncoder> blit_sort = [cb_prep blitCommandEncoder];
                [blit_sort fillBuffer:all_counts_buffer
                                range:NSMakeRange(0, sizeof(uint32_t) * n_buckets * n_windows)
                                value:0];
                // sorted_indices zero-fill removed: bucket-0 has count=0 (histogram
                // skips d==0), so reduce_gathered returns early at the count==0 check
                // and never reads bucket-0 entries. All other positions are written
                // by the scatter kernel before being read. Saves ~1.4ms per 1M MSM.
                [blit_sort endEncoding];

                id<MTLComputeCommandEncoder> enc_hist = [cb_prep computeCommandEncoder];
                [enc_hist setComputePipelineState:sort_histogram_fn];
                [enc_hist setBuffer:k1_buffer offset:0 atIndex:0];
                [enc_hist setBuffer:k2_buffer offset:0 atIndex:1];
                [enc_hist setBuffer:all_counts_buffer offset:0 atIndex:2];
                [enc_hist setBytes:&n_u32 length:sizeof(uint32_t) atIndex:3];
                [enc_hist setBytes:&wb_u32 length:sizeof(uint32_t) atIndex:4];
                [enc_hist setBytes:&nb_u32 length:sizeof(uint32_t) atIndex:5];
                [enc_hist setBytes:&nw_u32 length:sizeof(uint32_t) atIndex:6];
                [enc_hist dispatchThreads:grid_n threadsPerThreadgroup:tg_n];
                [enc_hist endEncoding];

                id<MTLComputeCommandEncoder> enc_pfx = [cb_prep computeCommandEncoder];
                [enc_pfx setComputePipelineState:sort_prefix_sum_fn];
                [enc_pfx setBuffer:all_counts_buffer offset:0 atIndex:0];
                [enc_pfx setBuffer:all_offsets_buffer offset:0 atIndex:1];
                [enc_pfx setBuffer:scatter_pos_buffer offset:0 atIndex:2];
                [enc_pfx setBytes:&nb_u32 length:sizeof(uint32_t) atIndex:3];
                [enc_pfx dispatchThreadgroups:MTLSizeMake(n_windows, 1, 1)
                      threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                [enc_pfx endEncoding];

                id<MTLComputeCommandEncoder> enc_scat = [cb_prep computeCommandEncoder];
                [enc_scat setComputePipelineState:sort_scatter_fn];
                [enc_scat setBuffer:k1_buffer offset:0 atIndex:0];
                [enc_scat setBuffer:k2_buffer offset:0 atIndex:1];
                [enc_scat setBuffer:scatter_pos_buffer offset:0 atIndex:2];
                [enc_scat setBuffer:sorted_indices_buffer offset:0 atIndex:3];
                [enc_scat setBytes:&n_u32 length:sizeof(uint32_t) atIndex:4];
                [enc_scat setBytes:&wb_u32 length:sizeof(uint32_t) atIndex:5];
                [enc_scat setBytes:&nb_u32 length:sizeof(uint32_t) atIndex:6];
                [enc_scat setBytes:&nw_u32 length:sizeof(uint32_t) atIndex:7];
                [enc_scat setBytes:&n2_u32 length:sizeof(uint32_t) atIndex:8];
                MTLSize grid_scat = MTLSizeMake(n, 1, 1);
                MTLSize tg_scat = MTLSizeMake(std::min(size_t(256), n), 1, 1);
                [enc_scat dispatchThreads:grid_scat threadsPerThreadgroup:tg_scat];
                [enc_scat endEncoding];

                phase_barrier(cb_prep, "sort");

                // Scale threshold with n to avoid classifying too many buckets as "large".
                // At n=4M, avg count/bucket ≈ 256 so threshold=256 would send half to large path.
                // Use max(256, 4× average) to keep large buckets to statistical outliers.
                uint32_t avg_per_bucket = static_cast<uint32_t>(n2 / n_buckets + 1);
                uint32_t large_thresh_u32 = std::max(uint32_t(256), avg_per_bucket * 4);

                // Phase 2: GPU Count-Sorted Mapping (CSM)
                // Replaces CPU CSM loop: the GPU kernel computes CSM, detects imbalance,
                // and identifies large buckets in a single dispatch per window.
                {
                    // Zero the atomic output buffers before GPU CSM dispatch.
                    // Only zero first 4 bytes of n_large_buckets_out (the count); bytes 4-11 stay {1,1}
                    // for indirect dispatch args format {threadgroups_x, 1, 1}.
                    id<MTLBlitCommandEncoder> blit_csm = [cb_prep blitCommandEncoder];
                    [blit_csm fillBuffer:n_large_buckets_out_buffer range:NSMakeRange(0, sizeof(uint32_t)) value:0];
                    [blit_csm fillBuffer:imbalance_flag_buffer range:NSMakeRange(0, sizeof(uint32_t)) value:0];
                    [blit_csm endEncoding];

                    uint32_t full_data_windows_u32 = static_cast<uint32_t>(128 / window_bits);

                    id<MTLComputeCommandEncoder> enc_csm = [cb_prep computeCommandEncoder];
                    [enc_csm setComputePipelineState:compute_csm_fn];
                    [enc_csm setBuffer:all_counts_buffer offset:0 atIndex:0];
                    [enc_csm setBuffer:count_sorted_map_buffer offset:0 atIndex:1];
                    [enc_csm setBuffer:large_bucket_ids_buffer offset:0 atIndex:2];
                    [enc_csm setBuffer:n_large_buckets_out_buffer offset:0 atIndex:3];
                    [enc_csm setBuffer:imbalance_flag_buffer offset:0 atIndex:4];
                    [enc_csm setBytes:&nb_u32 length:sizeof(uint32_t) atIndex:5];
                    [enc_csm setBytes:&nw_u32 length:sizeof(uint32_t) atIndex:6];
                    [enc_csm setBytes:&n2_u32 length:sizeof(uint32_t) atIndex:7];
                    [enc_csm setBytes:&full_data_windows_u32 length:sizeof(uint32_t) atIndex:8];
                    [enc_csm setBytes:&large_thresh_u32 length:sizeof(uint32_t) atIndex:9];
                    uint32_t large_buf_cap_u32 = static_cast<uint32_t>(n_buckets * n_windows);
                    [enc_csm setBytes:&large_buf_cap_u32 length:sizeof(uint32_t) atIndex:10];
                    // One threadgroup per window, 256 threads each
                    [enc_csm dispatchThreadgroups:MTLSizeMake(n_windows, 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc_csm endEncoding];
                }

                phase_barrier(cb_prep, "csm");

                // --- Phase 3+4 kernels merged into same command buffer ---
                // Metal auto-inserts barriers between compute encoders for data dependencies.
                // Eliminates one GPU-CPU roundtrip (waitUntilCompleted) per MSM call.

                MsmParams params = { static_cast<uint32_t>(n2), window_bits, static_cast<uint32_t>(n_buckets) };
                uint32_t n_segs = static_cast<uint32_t>(n_segments);
                uint32_t n_wins = static_cast<uint32_t>(n_windows);
                size_t total_segments = n_segments * n_windows;
                size_t total_buckets = n_buckets * n_windows;

                // Buckets buffer zero-fill removed: reduce_gathered writes every bucket
                // position — identity for bucket-0/count-0/large, accumulated for normal.
                // Large buckets are then overwritten by reduce_large_bucket. Saves ~0.6ms.

                // Fused gather+reduce: reads sorted_indices + points directly
                {
                    uint32_t nw_u32_r = static_cast<uint32_t>(n_windows);
                    id<MTLComputeCommandEncoder> enc_r = [cb_prep computeCommandEncoder];
                    [enc_r setComputePipelineState:reduce_gathered_fn];
                    [enc_r setBuffer:points_buffer offset:0 atIndex:0];
                    [enc_r setBuffer:buckets_buffer offset:0 atIndex:1];
                    [enc_r setBuffer:all_offsets_buffer offset:0 atIndex:2];
                    [enc_r setBuffer:all_counts_buffer offset:0 atIndex:3];
                    [enc_r setBytes:&params length:sizeof(MsmParams) atIndex:4];
                    [enc_r setBytes:&nw_u32_r length:sizeof(uint32_t) atIndex:5];
                    [enc_r setBuffer:sorted_indices_buffer offset:0 atIndex:6];
                    [enc_r setBuffer:count_sorted_map_buffer offset:0 atIndex:7];
                    [enc_r setBytes:&large_thresh_u32 length:sizeof(uint32_t) atIndex:8];
                    MTLSize grid = MTLSizeMake(total_buckets, 1, 1);
                    MTLSize tg = MTLSizeMake(std::min(size_t(256), total_buckets), 1, 1);
                    [enc_r dispatchThreads:grid threadsPerThreadgroup:tg];
                    [enc_r endEncoding];
                }

                // Large bucket reduce via indirect dispatch — GPU reads threadgroup count
                // from n_large_buckets_out_buffer (formatted as {count, 1, 1}).
                // If count is 0, Metal dispatches zero threadgroups (no-op).
                {
                    uint32_t n2_per_win = static_cast<uint32_t>(n2);
                    id<MTLComputeCommandEncoder> enc = [cb_prep computeCommandEncoder];
                    [enc setComputePipelineState:reduce_large_bucket_fn];
                    [enc setBuffer:points_buffer offset:0 atIndex:0];
                    [enc setBuffer:buckets_buffer offset:0 atIndex:1];
                    [enc setBuffer:all_offsets_buffer offset:0 atIndex:2];
                    [enc setBuffer:all_counts_buffer offset:0 atIndex:3];
                    [enc setBuffer:sorted_indices_buffer offset:0 atIndex:4];
                    [enc setBuffer:large_bucket_ids_buffer offset:0 atIndex:5];
                    [enc setBytes:&n2_per_win length:sizeof(uint32_t) atIndex:6];
                    [enc setBytes:&nb_u32 length:sizeof(uint32_t) atIndex:7];
                    [enc dispatchThreadgroupsWithIndirectBuffer:n_large_buckets_out_buffer
                                          indirectBufferOffset:0
                                         threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }

                phase_barrier(cb_prep, "reduce");

                // bucket_sum: Horner's sum per segment, parallelized across segments.
                {
                    id<MTLComputeCommandEncoder> enc = [cb_prep computeCommandEncoder];
                    [enc setComputePipelineState:bucket_sum_direct_fn];
                    [enc setBuffer:buckets_buffer offset:0 atIndex:0];
                    [enc setBuffer:segment_results_buffer offset:0 atIndex:1];
                    [enc setBytes:&params length:sizeof(MsmParams) atIndex:2];
                    [enc setBytes:&n_segs length:sizeof(uint32_t) atIndex:3];
                    [enc setBytes:&n_wins length:sizeof(uint32_t) atIndex:4];
                    MTLSize grid = MTLSizeMake(total_segments, 1, 1);
                    MTLSize tg = MTLSizeMake(std::min(size_t(256), total_segments), 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    [enc endEncoding];
                }

                phase_barrier(cb_prep, "bucket_sum");

                // combine_segments: two-pass for n_segments > 256
                if (n_segments <= 256) {
                    id<MTLComputeCommandEncoder> enc = [cb_prep computeCommandEncoder];
                    [enc setComputePipelineState:combine_segments_fn];
                    [enc setBuffer:segment_results_buffer offset:0 atIndex:0];
                    [enc setBuffer:window_results_buffer offset:0 atIndex:1];
                    [enc setBytes:&n_segs length:sizeof(uint32_t) atIndex:2];
                    NSUInteger tg_size = std::min(size_t(256), n_segments);
                    NSUInteger tg_pow2 = 1;
                    while (tg_pow2 < tg_size) tg_pow2 <<= 1;
                    [enc dispatchThreadgroups:MTLSizeMake(n_windows, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(tg_pow2, 1, 1)];
                    [enc endEncoding];
                } else {
                    constexpr uint32_t CHUNK_SIZE = 256;
                    uint32_t chunk_sz = CHUNK_SIZE;
                    size_t n_chunks = (n_segments + CHUNK_SIZE - 1) / CHUNK_SIZE;
                    size_t n_total_chunks = n_chunks * n_windows;
                    {
                        id<MTLComputeCommandEncoder> enc = [cb_prep computeCommandEncoder];
                        [enc setComputePipelineState:combine_segments_l1_fn];
                        [enc setBuffer:segment_results_buffer offset:0 atIndex:0];
                        [enc setBuffer:chunk_results_buffer offset:0 atIndex:1];
                        [enc setBytes:&n_segs length:sizeof(uint32_t) atIndex:2];
                        [enc setBytes:&chunk_sz length:sizeof(uint32_t) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake(n_total_chunks, 1, 1)
                             threadsPerThreadgroup:MTLSizeMake(CHUNK_SIZE, 1, 1)];
                        [enc endEncoding];
                    }
                    {
                        uint32_t n_chunks_u32 = static_cast<uint32_t>(n_chunks);
                        id<MTLComputeCommandEncoder> enc = [cb_prep computeCommandEncoder];
                        [enc setComputePipelineState:combine_segments_fn];
                        [enc setBuffer:chunk_results_buffer offset:0 atIndex:0];
                        [enc setBuffer:window_results_buffer offset:0 atIndex:1];
                        [enc setBytes:&n_chunks_u32 length:sizeof(uint32_t) atIndex:2];
                        NSUInteger tg_pow2 = 1;
                        while (tg_pow2 < n_chunks) tg_pow2 <<= 1;
                        [enc dispatchThreadgroups:MTLSizeMake(n_windows, 1, 1)
                             threadsPerThreadgroup:MTLSizeMake(tg_pow2, 1, 1)];
                        [enc endEncoding];
                    }
                }

                // Single commit+wait for the entire pipeline (GLV+sort+CSM+reduce+bucket_sum+combine)
                double t_pre_commit = gpu_time("pre_commit");
                [cb_prep commit];
                [cb_prep waitUntilCompleted];
                double t_post_wait = gpu_time("post_wait");
                if ([cb_prep error]) {
                    fprintf(stderr, "[GPU_ERR] cb error: %s\n", [[cb_prep error] localizedDescription].UTF8String);
                    return false;
                }
                if (gpu_profile) {
                    if (gpu_profile_phases) {
                        // Per-phase mode: last CB is "combine" phase
                        double gs = [cb_prep GPUStartTime];
                        double ge = [cb_prep GPUEndTime];
                        phase_times.push_back({"combine", (ge - gs) * 1000.0});
                        double total_gpu = 0;
                        fprintf(stderr, "[GPU_MSM n=%zu phases]", n);
                        for (auto& pt : phase_times) {
                            fprintf(stderr, " %s=%.2fms", pt.name, pt.gpu_ms);
                            total_gpu += pt.gpu_ms;
                        }
                        fprintf(stderr, " total=%.2fms wall=%.2fms\n", total_gpu, (t_post_wait - t_pre_commit) * 1000.0);
                    } else {
                        double gpu_start = [cb_prep GPUStartTime];
                        double gpu_end = [cb_prep GPUEndTime];
                        fprintf(stderr, "[GPU_MSM n=%zu] wall=%.2fms gpu=%.2fms\n",
                                n, (t_post_wait - t_pre_commit) * 1000.0,
                                (gpu_end - gpu_start) * 1000.0);
                    }
                }

            }

            // Check imbalance after all GPU work completes.
            // If imbalanced, the phase3 work was wasted but the result is discarded.
            // Imbalance is rare (pathological scalar distributions only).
            uint32_t imbalance_val = *static_cast<uint32_t*>([imbalance_flag_buffer contents]);
            if (imbalance_val != 0) {
                if (n >= 65536) fprintf(stderr, "[MSM %zu] FAIL: GPU CSM detected bucket imbalance\n", n);
                return false;
            }

            // Phase 5: Read window results and combine with Horner's method (CPU)
            auto* win_results = static_cast<const GpuPointProjective*>([window_results_buffer contents]);

            static_assert(sizeof(GpuPointProjective) == sizeof(curve::BN254::Element),
                          "GpuPointProjective and Element must have same size");

            std::vector<curve::BN254::Element> window_elems(n_windows);
            std::memcpy(window_elems.data(), win_results, sizeof(curve::BN254::Element) * n_windows);

            // GPU uses z=0 for identity, but barretenberg uses x=modulus sentinel.
            // Convert before Horner combination so self_dbl/operator+= work correctly.
            for (size_t w = 0; w < n_windows; w++) {
                if (window_elems[w].z.is_zero()) {
                    window_elems[w] = curve::BN254::Element::infinity();
                }
            }

            // Horner's method: result = w[last]; for w = last-1..0: result = result * 2^bits + w[i]
            curve::BN254::Element result = window_elems[n_windows - 1];
            for (size_t w = n_windows - 1; w > 0; w--) {
                for (uint32_t b = 0; b < window_bits; b++) {
                    result.self_dbl();
                }
                result += window_elems[w - 1];
            }

            result_out = result;
            return true;
        } // @autoreleasepool
    }

    bool run_msm_direct(const curve::BN254::AffineElement* points,
                        const curve::BN254::ScalarField* scalars,
                        size_t num_points,
                        uint32_t max_scalar_bits,
                        curve::BN254::Element& result_out)
    {
        @autoreleasepool {
            const size_t n = num_points;

            if (max_scalar_bits == 0) {
                result_out = curve::BN254::Element::infinity();
                return true;
            }

            const uint32_t window_bits = 10;
            const size_t n_windows = (max_scalar_bits + window_bits - 1) / window_bits;
            const size_t n_buckets = size_t(1) << window_bits;
            const size_t n_segments = std::min(size_t(256), std::max(size_t(1), n_buckets / 2));

            ensure_buffers(n, n_buckets, n_segments, n_windows);

            // Phase 0: Copy scalars + cache SRS points

            auto* gpu_points = static_cast<GpuPointAffine*>([points_buffer contents]);
            std::memcpy(gpu_points, points, sizeof(GpuPointAffine) * n);

            // Phase 1+2: GPU GLV + counting sort

            auto* all_offsets = static_cast<uint32_t*>([all_offsets_buffer contents]);
            auto* all_counts = static_cast<uint32_t*>([all_counts_buffer contents]);
            auto* sorted_idx = static_cast<uint32_t*>([sorted_indices_buffer contents]);

            const uint32_t bucket_mask = (1u << window_bits) - 1;

            std::vector<std::vector<uint32_t>> per_window_counts(n_windows, std::vector<uint32_t>(n_buckets, 0));
            std::vector<std::vector<uint32_t>> per_window_positions(n_windows, std::vector<uint32_t>(n_buckets, 0));

            auto* counts_ptr = per_window_counts.data();
            auto* positions_ptr = per_window_positions.data();
            const auto* scalar_data = scalars;
            // Multi-limb digit extraction: for 10-bit windows on 254-bit scalars,
            // we need to read from the correct 64-bit limb and handle cross-limb boundaries.
            dispatch_apply(n_windows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t w) {
                size_t w_off = w * n_buckets;
                size_t idx_base = w * n;
                auto& counts = counts_ptr[w];
                auto& positions = positions_ptr[w];

                uint32_t shift = static_cast<uint32_t>(w * window_bits);

                // Multi-limb digit extraction for 10-bit windows on up to 254-bit scalars
                const uint32_t limb_idx = shift / 64;
                const uint32_t bit_off = shift % 64;
                const bool cross_limb = (bit_off > 54 && limb_idx + 1 < 4);

                for (size_t i = 0; i < n; i++) {
                    uint64_t lo = scalar_data[i].data[limb_idx] >> bit_off;
                    uint64_t hi = cross_limb ? scalar_data[i].data[limb_idx + 1] << (64 - bit_off) : 0;
                    uint32_t digit = static_cast<uint32_t>(lo | hi) & bucket_mask;
                    counts[digit]++;
                }

                uint32_t running_offset = 0;
                for (size_t b = 0; b < n_buckets; b++) {
                    all_offsets[w_off + b] = running_offset;
                    all_counts[w_off + b] = counts[b];
                    positions[b] = running_offset;
                    running_offset += counts[b];
                }

                for (size_t i = 0; i < n; i++) {
                    uint64_t lo = scalar_data[i].data[limb_idx] >> bit_off;
                    uint64_t hi = cross_limb ? scalar_data[i].data[limb_idx + 1] << (64 - bit_off) : 0;
                    uint32_t digit = static_cast<uint32_t>(lo | hi) & bucket_mask;
                    if (digit != 0) {
                        sorted_idx[idx_base + positions[digit]] = static_cast<uint32_t>(i);
                        positions[digit]++;
                    }
                }
            });

            {
                // Batch MSM path: use identity CSM (cache-friendly) since batch MSMs
                // are typically smaller and have uniform bucket distributions
                auto* csm = static_cast<uint32_t*>([count_sorted_map_buffer contents]);
                size_t total_csm = n_buckets * n_windows;
                for (size_t i = 0; i < total_csm; i++) {
                    csm[i] = static_cast<uint32_t>(i);
                }
            }

            size_t total_buckets = n_buckets * n_windows;
            MsmParams params = { static_cast<uint32_t>(n), window_bits, static_cast<uint32_t>(n_buckets) };
            uint32_t n_wins_batch = static_cast<uint32_t>(n_windows);
            uint32_t n_segs = static_cast<uint32_t>(n_segments);
            uint32_t n_wins = static_cast<uint32_t>(n_windows);
            size_t total_segments = n_segments * n_windows;

            {
                id<MTLCommandBuffer> cb1 = [command_queue commandBuffer];

                id<MTLBlitCommandEncoder> blit1 = [cb1 blitCommandEncoder];
                [blit1 fillBuffer:buckets_buffer
                            range:NSMakeRange(0, sizeof(GpuPointProjective) * total_buckets)
                            value:0];
                [blit1 fillBuffer:segment_results_buffer
                            range:NSMakeRange(0, sizeof(GpuPointProjective) * total_segments)
                            value:0];
                [blit1 fillBuffer:window_results_buffer
                            range:NSMakeRange(0, sizeof(GpuPointProjective) * n_windows)
                            value:0];
                [blit1 endEncoding];

                // Fused gather+reduce: reads sorted_indices + points directly
                id<MTLComputeCommandEncoder> enc1 = [cb1 computeCommandEncoder];
                [enc1 setComputePipelineState:reduce_gathered_fn];
                [enc1 setBuffer:points_buffer offset:0 atIndex:0];
                [enc1 setBuffer:buckets_buffer offset:0 atIndex:1];
                [enc1 setBuffer:all_offsets_buffer offset:0 atIndex:2];
                [enc1 setBuffer:all_counts_buffer offset:0 atIndex:3];
                [enc1 setBytes:&params length:sizeof(MsmParams) atIndex:4];
                [enc1 setBytes:&n_wins_batch length:sizeof(uint32_t) atIndex:5];
                [enc1 setBuffer:sorted_indices_buffer offset:0 atIndex:6];
                [enc1 setBuffer:count_sorted_map_buffer offset:0 atIndex:7];
                MTLSize grid1 = MTLSizeMake(total_buckets, 1, 1);
                MTLSize tg1 = MTLSizeMake(std::min(size_t(256), total_buckets), 1, 1);
                [enc1 dispatchThreads:grid1 threadsPerThreadgroup:tg1];
                [enc1 endEncoding];

                id<MTLComputeCommandEncoder> enc2 = [cb1 computeCommandEncoder];
                [enc2 setComputePipelineState:bucket_sum_direct_fn];
                [enc2 setBuffer:buckets_buffer offset:0 atIndex:0];
                [enc2 setBuffer:segment_results_buffer offset:0 atIndex:1];
                [enc2 setBytes:&params length:sizeof(MsmParams) atIndex:2];
                [enc2 setBytes:&n_segs length:sizeof(uint32_t) atIndex:3];
                [enc2 setBytes:&n_wins length:sizeof(uint32_t) atIndex:4];
                MTLSize grid2 = MTLSizeMake(total_segments, 1, 1);
                MTLSize tg2 = MTLSizeMake(std::min(size_t(256), total_segments), 1, 1);
                [enc2 dispatchThreads:grid2 threadsPerThreadgroup:tg2];
                [enc2 endEncoding];

                id<MTLComputeCommandEncoder> enc3 = [cb1 computeCommandEncoder];
                [enc3 setComputePipelineState:combine_segments_fn];
                [enc3 setBuffer:segment_results_buffer offset:0 atIndex:0];
                [enc3 setBuffer:window_results_buffer offset:0 atIndex:1];
                [enc3 setBytes:&n_segs length:sizeof(uint32_t) atIndex:2];
                NSUInteger tg_size = std::min(size_t(256), n_segments);
                NSUInteger tg_pow2 = 1;
                while (tg_pow2 < tg_size)
                    tg_pow2 <<= 1;
                [enc3 dispatchThreadgroups:MTLSizeMake(n_windows, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(tg_pow2, 1, 1)];
                [enc3 endEncoding];

                [cb1 commit];
                [cb1 waitUntilCompleted];
                if ([cb1 error]) {
                    return false;
                }
            }

            // Phase 3+4: GPU reduce + bucket_sum + combine

            auto* win_results = static_cast<const GpuPointProjective*>([window_results_buffer contents]);

            std::vector<curve::BN254::Element> window_elems(n_windows);
            std::memcpy(window_elems.data(), win_results, sizeof(curve::BN254::Element) * n_windows);

            // GPU uses z=0 for identity; convert to barretenberg sentinel form
            for (size_t w = 0; w < n_windows; w++) {
                if (window_elems[w].z.is_zero()) {
                    window_elems[w] = curve::BN254::Element::infinity();
                }
            }

            curve::BN254::Element result = window_elems[n_windows - 1];
            for (size_t w = n_windows - 1; w > 0; w--) {
                for (uint32_t b = 0; b < window_bits; b++) {
                    result.self_dbl();
                }
                result += window_elems[w - 1];
            }

            result_out = result;
            return true;
        } // @autoreleasepool
    }
};

// ======================== Public API ========================

bool metal_available()
{
    static int cached = -1;
    if (cached == -1) {
        if (std::getenv("BB_NO_GPU")) {
            cached = 0;
        } else {
            cached = MetalMSMContext::instance().init_if_needed() ? 1 : 0;
        }
    }
    return cached == 1;
}

void metal_init_async()
{
    // Fire-and-forget: start Metal device + shader compilation on a background thread.
    // By the time prewarm() is called, init_if_needed() (guarded by std::call_once) will
    // either return immediately (already done) or block briefly until the bg thread finishes.
    static std::once_flag async_flag;
    std::call_once(async_flag, []() {
        std::thread([]() {
            MetalMSMContext::instance().init_if_needed();
        }).detach();
    });
}

void metal_prewarm(size_t num_points, const curve::BN254::AffineElement* srs_points)
{
    MetalMSMContext::instance().prewarm(num_points, srs_points);
}

curve::BN254::AffineElement metal_pippenger(PolynomialSpan<const curve::BN254::ScalarField> scalars,
                                            std::span<const curve::BN254::AffineElement> points,
                                            bool skip_imbalance_check)
{
    auto& ctx = MetalMSMContext::instance();
    BB_ASSERT(ctx.init_if_needed());

    const size_t num_points = scalars.size();
    if (num_points == 0) {
        return curve::BN254::AffineElement::infinity();
    }
    const size_t start = scalars.start_index;

    // Use fused copy+convert path: scalars are converted from Montgomery form
    // directly into the GPU shared buffer, avoiding the intermediate non_mont_scalars_buf.
    curve::BN254::Element result;
    bool ok = ctx.run_msm(&points[start], &scalars[start], num_points, result, skip_imbalance_check,
                          /*from_montgomery=*/true);

    if (!ok || result.is_point_at_infinity() || result.z.is_zero()) {
        return curve::BN254::AffineElement::infinity();
    }

    return result;
}

curve::BN254::AffineElement metal_pippenger_raw(std::span<const curve::BN254::ScalarField> scalars,
                                                std::span<const curve::BN254::AffineElement> points,
                                                bool skip_imbalance_check)
{
    auto& ctx = MetalMSMContext::instance();
    BB_ASSERT(ctx.init_if_needed());

    const size_t num_points = scalars.size();
    if (num_points == 0) {
        return curve::BN254::AffineElement::infinity();
    }

    // Use fused copy+convert path in run_msm: scalars are converted from Montgomery
    // directly into the GPU shared buffer, eliminating the intermediate non_mont_scalars_buf copy.
    curve::BN254::Element result;
    bool ok = ctx.run_msm(points.data(), scalars.data(), num_points, result, skip_imbalance_check,
                          /*from_montgomery=*/true);

    if (!ok || result.is_point_at_infinity() || result.z.is_zero()) {
        return curve::BN254::AffineElement::infinity();
    }

    return result;
}

curve::BN254::AffineElement metal_pippenger_preconverted(std::span<const curve::BN254::ScalarField> scalars,
                                                         std::span<const curve::BN254::AffineElement> points,
                                                         bool skip_imbalance_check)
{
    auto& ctx = MetalMSMContext::instance();
    BB_ASSERT(ctx.init_if_needed());

    const size_t num_points = scalars.size();
    if (num_points == 0) {
        return curve::BN254::AffineElement::infinity();
    }

    // Scalars are already in non-Montgomery form.
    // Skip dispatch_apply max_scalar_bits scan to avoid GCD thread contention with CPU Pippenger.
    // Preconverted path is used in GPU+CPU mixed batches (fold_commit) where scalars are full
    // field elements (>20 bits), so run_msm_direct is never applicable.
    curve::BN254::Element result;
    bool ok = ctx.run_msm(points.data(), scalars.data(), num_points, result, skip_imbalance_check);

    if (!ok || result.is_point_at_infinity() || result.z.is_zero()) {
        return curve::BN254::AffineElement::infinity();
    }

    return result;
}

} // namespace bb::scalar_multiplication::metal

#else // !__APPLE__

namespace bb::scalar_multiplication::metal {

bool metal_available()
{
    return false;
}

void metal_init_async() {}
void metal_prewarm(size_t /*num_points*/, const curve::BN254::AffineElement* /*srs_points*/) {}

curve::BN254::AffineElement metal_pippenger_preconverted(std::span<const curve::BN254::ScalarField> /*scalars*/,
                                                         std::span<const curve::BN254::AffineElement> /*points*/,
                                                         bool /*skip_imbalance_check*/)
{
    return curve::BN254::AffineElement::infinity();
}

} // namespace bb::scalar_multiplication::metal

#endif
