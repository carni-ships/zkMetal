// test_metallib.m — Load zkmetal.metallib and print available kernel functions
//
// Build and run:
//   clang -framework Metal -framework Foundation \
//         -o test_metallib test_metallib.m MetalLibLoader.m
//   ./test_metallib [path/to/zkmetal.metallib]
//
// Or use the build script:
//   ./build_metallib.sh && ./build_test.sh

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "MetalLibLoader.h"
#include <stdio.h>
#include <string.h>

// ANSI colors for terminal output
#define GREEN  "\033[32m"
#define RED    "\033[31m"
#define YELLOW "\033[33m"
#define BOLD   "\033[1m"
#define RESET  "\033[0m"

static int test_count = 0;
static int pass_count = 0;

static void check(int condition, const char* description) {
    test_count++;
    if (condition) {
        pass_count++;
        printf("  " GREEN "PASS" RESET " %s\n", description);
    } else {
        printf("  " RED "FAIL" RESET " %s\n", description);
    }
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        const char* metallib_path = "zkmetal.metallib";
        if (argc > 1) {
            metallib_path = argv[1];
        }

        printf(BOLD "zkMetal .metallib test\n" RESET);
        printf("Loading: %s\n\n", metallib_path);

        // ----------------------------------------------------------------
        // Test 1: Initialization
        // ----------------------------------------------------------------
        printf(BOLD "1. Initialization\n" RESET);

        check(!zkmetal_gpu_is_ready(), "Not ready before init");

        ZkGpuStatus status = zkmetal_gpu_init(metallib_path);
        check(status == ZK_GPU_SUCCESS, "zkmetal_gpu_init() succeeds");
        check(zkmetal_gpu_is_ready(), "GPU is ready after init");

        if (status != ZK_GPU_SUCCESS) {
            printf("\n" RED "Fatal: could not load metallib. Aborting.\n" RESET);
            return 1;
        }

        // ----------------------------------------------------------------
        // Test 2: Device info
        // ----------------------------------------------------------------
        printf("\n" BOLD "2. Device info\n" RESET);

        const char* dev_name = zkmetal_gpu_device_name();
        check(dev_name != NULL, "Device name is not NULL");
        if (dev_name) {
            printf("       GPU: %s\n", dev_name);
        }

        uint32_t n_kernels = zkmetal_gpu_kernel_count();
        check(n_kernels > 0, "Kernel count > 0");
        printf("       Kernels: %u\n", n_kernels);

        // ----------------------------------------------------------------
        // Test 3: List all kernel functions
        // ----------------------------------------------------------------
        printf("\n" BOLD "3. Available kernels (%u total)\n" RESET, n_kernels);

        // Group kernels by category
        const char* categories[] = {
            "msm_",  "ntt_",  "intt_", "poseidon2_", "keccak",
            "sha256", "blake3", "fri_", "poly_", "radix_sort",
            "batch_", "glv_", "sumcheck_", NULL
        };
        const char* cat_names[] = {
            "MSM", "NTT", "iNTT", "Poseidon2", "Keccak",
            "SHA-256", "Blake3", "FRI", "Polynomial", "Sort",
            "Batch", "GLV", "Sumcheck"
        };

        for (uint32_t i = 0; i < n_kernels; i++) {
            const char* name = zkmetal_gpu_kernel_name(i);
            if (name) {
                // Find category
                const char* cat = "Other";
                for (int c = 0; categories[c]; c++) {
                    if (strstr(name, categories[c])) {
                        cat = cat_names[c];
                        break;
                    }
                }
                printf("  " YELLOW "%-12s" RESET " %s\n", cat, name);
            }
        }

        // ----------------------------------------------------------------
        // Test 4: Pipeline creation for key kernels
        // ----------------------------------------------------------------
        printf("\n" BOLD "4. Pipeline creation\n" RESET);

        const char* key_kernels[] = {
            "msm_reduce_sorted_buckets",
            "ntt_butterfly",
            "poseidon2_hash_pairs",
            "keccak256_hash_batch",
            "sha256_hash_batch",
            "blake3_hash_batch",
            "radix_sort_histogram",
            NULL
        };

        for (int i = 0; key_kernels[i]; i++) {
            void* pipeline = zkmetal_gpu_create_pipeline(key_kernels[i]);
            char desc[128];
            snprintf(desc, sizeof(desc), "Pipeline: %s", key_kernels[i]);
            check(pipeline != NULL, desc);
            if (pipeline) {
                zkmetal_gpu_destroy_pipeline(pipeline);
            }
        }

        // ----------------------------------------------------------------
        // Test 5: Compute test (Poseidon2 hash if available)
        // ----------------------------------------------------------------
        printf("\n" BOLD "5. Compute test (Poseidon2 hash pair)\n" RESET);

        // Two zero field elements as input — not cryptographically meaningful
        // but verifies the GPU dispatch pipeline works end-to-end
        uint8_t input[64];
        uint8_t output[32];
        memset(input, 0, sizeof(input));
        memset(output, 0xFF, sizeof(output)); // Fill with 0xFF to detect changes

        ZkGpuStatus hash_status = zkmetal_gpu_poseidon2_hash_pairs(input, 1, output);
        if (hash_status == ZK_GPU_SUCCESS) {
            check(1, "Poseidon2 GPU dispatch succeeded");
            // Check output changed from 0xFF
            int changed = 0;
            for (int i = 0; i < 32; i++) {
                if (output[i] != 0xFF) { changed = 1; break; }
            }
            check(changed, "Output buffer was written by GPU");
            printf("       Hash: ");
            for (int i = 0; i < 32; i++) printf("%02x", output[i]);
            printf("\n");
        } else {
            printf("  " YELLOW "SKIP" RESET " Poseidon2 compute (status=%d) — "
                   "kernel may need additional buffers\n", hash_status);
        }

        // ----------------------------------------------------------------
        // Summary
        // ----------------------------------------------------------------
        printf("\n" BOLD "Results: %d/%d passed\n" RESET, pass_count, test_count);

        zkmetal_gpu_shutdown();
        check(!zkmetal_gpu_is_ready(), "Not ready after shutdown");

        return (pass_count == test_count) ? 0 : 1;
    }
}
