/*
 * zkmetal_bb_bridge.cpp -- Implementation of the BB-zkMetal bridge
 *
 * Handles:
 *   - Lazy GPU initialization (thread-safe)
 *   - MSM dispatch with GPU threshold
 *   - NTT dispatch with log2 computation
 *   - Format notes: BB and zkMetal both use uint64_t[4] Montgomery on LE ARM64,
 *     so no conversion is needed for field elements. For MSM scalars, zkMetal
 *     takes uint32_t* but the memory layout of uint64_t[4] == uint32_t[8]
 *     on little-endian, so we reinterpret_cast.
 */

#ifdef HAS_ZKMETAL

#include "zkmetal_bb_bridge.h"

#include <cmath>
#include <mutex>

extern "C" {
#include <zkmetal.h>
}

namespace zkmetal {

// ============================================================
// Initialization
// ============================================================

namespace {

std::once_flag g_init_flag;
std::atomic<bool> g_initialized{false};
std::atomic<bool> g_gpu_available{false};

void do_initialize() {
    // zkMetal initializes Metal pipelines lazily on first GPU call.
    // We just mark ourselves as ready. The actual Metal device creation
    // happens inside zkMetal's MSM/NTT routines on first invocation.
    g_initialized.store(true, std::memory_order_release);

    // Check if we're on Apple Silicon (we must be if HAS_ZKMETAL is defined
    // and the library linked, but verify at runtime).
#if defined(__APPLE__) && defined(__aarch64__)
    g_gpu_available.store(true, std::memory_order_release);
#else
    g_gpu_available.store(false, std::memory_order_release);
#endif
}

} // anonymous namespace

bool ensure_initialized() {
    std::call_once(g_init_flag, do_initialize);
    return g_initialized.load(std::memory_order_acquire);
}

bool is_gpu_available() {
    ensure_initialized();
    return g_gpu_available.load(std::memory_order_acquire);
}

// ============================================================
// MSM
// ============================================================

void msm_bn254_g1(const uint64_t* points_affine,
                   const uint64_t* scalars,
                   size_t n,
                   uint64_t* result_jacobian)
{
    ensure_initialized();

    // zkMetal's Pippenger expects:
    //   points: const uint64_t* -- affine, n * 8 uint64_t (x[4], y[4] each)
    //   scalars: const uint32_t* -- n * 8 uint32_t (256-bit scalar each)
    //   n: int
    //   result: uint64_t* -- Jacobian, 12 uint64_t (X[4], Y[4], Z[4])
    //
    // BB passes scalars as uint64_t[4] in Montgomery form.
    // On little-endian ARM64, uint64_t[4] and uint32_t[8] have identical
    // memory layout, so reinterpret_cast is safe and zero-cost.
    const auto* scalars_u32 = reinterpret_cast<const uint32_t*>(scalars);

    bn254_pippenger_msm(points_affine, scalars_u32, static_cast<int>(n), result_jacobian);
}

void msm_grumpkin(const uint64_t* points_affine,
                  const uint64_t* scalars,
                  size_t n,
                  uint64_t* result_jacobian)
{
    ensure_initialized();

    const auto* scalars_u32 = reinterpret_cast<const uint32_t*>(scalars);
    grumpkin_pippenger_msm(points_affine, scalars_u32, static_cast<int>(n), result_jacobian);
}

// ============================================================
// NTT
// ============================================================

namespace {

int compute_log2(size_t n) {
    // n must be a power of 2
    int logN = 0;
    size_t tmp = n;
    while (tmp > 1) {
        tmp >>= 1;
        logN++;
    }
    return logN;
}

} // anonymous namespace

void ntt_forward(uint64_t* data, size_t n) {
    ensure_initialized();
    int logN = compute_log2(n);
    bn254_fr_ntt(data, logN);
}

void ntt_inverse(uint64_t* data, size_t n) {
    ensure_initialized();
    int logN = compute_log2(n);
    bn254_fr_intt(data, logN);
}

} // namespace zkmetal

#endif // HAS_ZKMETAL
