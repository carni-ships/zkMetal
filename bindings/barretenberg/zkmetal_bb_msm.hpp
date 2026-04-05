/*
 * zkmetal_bb_msm.hpp -- Drop-in MSM replacement for Barretenberg
 *
 * Include this in BB's scalar_multiplication code to route large MSMs
 * through zkMetal's GPU-accelerated Pippenger implementation.
 *
 * Usage:
 *   #include "zkmetal_bb_msm.hpp"
 *   // Then in your MSM function:
 *   if (zkmetal::msm::try_gpu_msm<Curve>(points, scalars, n, result))
 *       return result;
 *   // else fall through to CPU implementation
 */

#pragma once

#ifdef HAS_ZKMETAL

#include "zkmetal_bb_bridge.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace zkmetal {
namespace msm {

/// Tag types for curve dispatch
struct BN254Tag {};
struct GrumpkinTag {};

/// Trait to detect curve type from BB's template parameter.
/// Specialize this for your curve types. Default: unsupported.
template <typename Curve>
struct CurveTraits {
    static constexpr bool supported = false;
};

/*
 * To wire up with BB's actual curve types, add specializations like:
 *
 *   template<> struct CurveTraits<bb::curve::BN254> {
 *       static constexpr bool supported = true;
 *       using tag = BN254Tag;
 *   };
 *
 *   template<> struct CurveTraits<bb::curve::Grumpkin> {
 *       static constexpr bool supported = true;
 *       using tag = GrumpkinTag;
 *   };
 */

/// Internal dispatch by tag
namespace detail {

inline void gpu_msm(BN254Tag,
                     const uint64_t* points,
                     const uint64_t* scalars,
                     size_t n,
                     uint64_t* result)
{
    zkmetal::msm_bn254_g1(points, scalars, n, result);
}

inline void gpu_msm(GrumpkinTag,
                     const uint64_t* points,
                     const uint64_t* scalars,
                     size_t n,
                     uint64_t* result)
{
    zkmetal::msm_grumpkin(points, scalars, n, result);
}

} // namespace detail

/**
 * try_gpu_msm -- Attempt GPU-accelerated MSM.
 *
 * @tparam Curve  BB curve type (must have CurveTraits specialization)
 * @param points  Affine points array (8 uint64_t each for 256-bit curves)
 * @param scalars Scalar array (4 uint64_t each, Montgomery form)
 * @param n       Number of point-scalar pairs
 * @param result  Output Jacobian point (12 uint64_t)
 * @return true   if handled by GPU, false if caller should use CPU fallback
 *
 * Returns false when:
 *   - Curve is not supported
 *   - n < MSM_GPU_THRESHOLD (CPU is faster for small MSMs)
 *   - GPU is not available
 */
template <typename Curve>
inline bool try_gpu_msm(const void* points,
                         const void* scalars,
                         size_t n,
                         void* result)
{
    if constexpr (!CurveTraits<Curve>::supported) {
        return false;
    } else {
        if (n < MSM_GPU_THRESHOLD || !is_gpu_available()) {
            return false;
        }

        detail::gpu_msm(
            typename CurveTraits<Curve>::tag{},
            static_cast<const uint64_t*>(points),
            static_cast<const uint64_t*>(scalars),
            n,
            static_cast<uint64_t*>(result)
        );
        return true;
    }
}

/**
 * try_gpu_msm_raw -- Non-templated version for direct use.
 *
 * @param curve_id  0 = BN254, 1 = Grumpkin
 * @param points    Affine points
 * @param scalars   Scalars (Montgomery form)
 * @param n         Count
 * @param result    Jacobian output
 * @return true if GPU handled, false for CPU fallback
 */
inline bool try_gpu_msm_raw(int curve_id,
                              const uint64_t* points,
                              const uint64_t* scalars,
                              size_t n,
                              uint64_t* result)
{
    if (n < MSM_GPU_THRESHOLD || !is_gpu_available()) {
        return false;
    }

    switch (curve_id) {
        case 0: // BN254
            zkmetal::msm_bn254_g1(points, scalars, n, result);
            return true;
        case 1: // Grumpkin
            zkmetal::msm_grumpkin(points, scalars, n, result);
            return true;
        default:
            return false;
    }
}

} // namespace msm
} // namespace zkmetal

#endif // HAS_ZKMETAL
