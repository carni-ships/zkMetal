// MSM GPU kernels: Pippenger's bucket method
// Phase 1: Reduce sorted points per bucket
// Phase 2: Weighted bucket sum per segment
// Phase 3: Tree reduction of segment results

#include "../geometry/bn254_curve.metal"

struct MsmParams {
    uint n_points;
    uint window_bits;
    uint n_buckets;
};

// Phase 1: Reduce pre-sorted points per bucket (batched across windows)
// Uses count-sorted mapping for SIMD uniformity
kernel void msm_reduce_sorted_buckets(
    device const PointAffine* points           [[buffer(0)]],
    device PointProjective* buckets            [[buffer(1)]],
    device const uint* bucket_offsets          [[buffer(2)]],
    device const uint* bucket_counts           [[buffer(3)]],
    constant MsmParams& params                 [[buffer(4)]],
    constant uint& n_windows                   [[buffer(5)]],
    device const uint* sorted_indices          [[buffer(6)]],
    device const uint* count_sorted_map        [[buffer(7)]],
    uint tid                                   [[thread_position_in_grid]]
) {
    uint total = params.n_buckets * n_windows;
    if (tid >= total) return;

    uint orig_pos = count_sorted_map[tid];
    // count_sorted_map packs window in upper 16 bits, bucket in lower 16 bits
    uint orig_bucket = orig_pos & 0xFFFFu;
    uint orig_window = orig_pos >> 16u;
    uint flat_idx = orig_window * params.n_buckets + orig_bucket;

    if (orig_bucket == 0) {
        buckets[flat_idx] = point_identity();
        return;
    }

    uint count = bucket_counts[flat_idx];
    if (count == 0) {
        buckets[flat_idx] = point_identity();
        return;
    }

    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[flat_idx];
    uint raw_idx0 = sorted_indices[base + offset];
    PointAffine pt0 = points[raw_idx0 & 0x7FFFFFFFu];
    if (raw_idx0 & 0x80000000u) pt0.y = fp_neg(pt0.y);
    PointProjective acc = point_from_affine(pt0);
    for (uint i = 1; i < count; i++) {
        uint raw_idx = sorted_indices[base + offset + i];
        PointAffine pt = points[raw_idx & 0x7FFFFFFFu];
        if (raw_idx & 0x80000000u) pt.y = fp_neg(pt.y);
        acc = point_add_mixed(acc, pt);
    }
    buckets[flat_idx] = acc;
}

// SIMD shuffle helper for PointProjective
inline PointProjective simd_shuffle_down_point(PointProjective p, uint offset) {
    PointProjective r;
    for (int k = 0; k < 8; k++) {
        r.x.v[k] = simd_shuffle_down(p.x.v[k], offset);
        r.y.v[k] = simd_shuffle_down(p.y.v[k], offset);
        r.z.v[k] = simd_shuffle_down(p.z.v[k], offset);
    }
    return r;
}

// Phase 1b: Cooperative reduce — one SIMD group (32 threads) per bucket.
// Each thread accumulates its portion, then SIMD tree reduction merges.
// For buckets with 1-32 points: 1 point per thread max, then reduce.
// For buckets with >32 points: each thread loops with stride 32.
kernel void msm_reduce_cooperative(
    device const PointAffine* points           [[buffer(0)]],
    device PointProjective* buckets            [[buffer(1)]],
    device const uint* bucket_offsets          [[buffer(2)]],
    device const uint* bucket_counts           [[buffer(3)]],
    constant MsmParams& params                 [[buffer(4)]],
    constant uint& n_windows                   [[buffer(5)]],
    device const uint* sorted_indices          [[buffer(6)]],
    device const uint* count_sorted_map        [[buffer(7)]],
    uint tgid                                  [[threadgroup_position_in_grid]],
    uint lid                                   [[thread_index_in_threadgroup]]
) {
    uint total = params.n_buckets * n_windows;
    if (tgid >= total) return;

    uint orig_pos = count_sorted_map[tgid];
    // count_sorted_map packs window in upper 16 bits, bucket in lower 16 bits
    uint orig_bucket = orig_pos & 0xFFFFu;
    uint orig_window = orig_pos >> 16u;
    uint flat_idx = orig_window * params.n_buckets + orig_bucket;

    // Identity for bucket 0 or empty buckets
    if (orig_bucket == 0 || bucket_counts[flat_idx] == 0) {
        if (lid == 0) {
            buckets[flat_idx] = point_identity();
        }
        return;
    }

    uint count = bucket_counts[flat_idx];
    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[flat_idx];

    // Each thread accumulates points at stride 32
    PointProjective acc = point_identity();
    for (uint i = lid; i < count; i += 32) {
        uint raw_idx = sorted_indices[base + offset + i];
        PointAffine pt = points[raw_idx & 0x7FFFFFFFu];
        if (raw_idx & 0x80000000u) pt.y = fp_neg(pt.y);
        if (point_is_identity(acc)) {
            acc = point_from_affine(pt);
        } else {
            acc = point_add_mixed(acc, pt);
        }
    }

    // SIMD tree reduction (5 levels for 32 lanes)
    for (uint off = 16; off > 0; off >>= 1) {
        PointProjective other = simd_shuffle_down_point(acc, off);
        if (lid < off) {
            if (point_is_identity(acc)) {
                acc = other;
            } else if (!point_is_identity(other)) {
                acc = point_add(acc, other);
            }
        }
    }

    if (lid == 0) {
        buckets[flat_idx] = acc;
    }
}

// Phase 2: Direct weighted bucket sum per segment
kernel void msm_bucket_sum_direct(
    device const PointProjective* buckets       [[buffer(0)]],
    device PointProjective* segment_results     [[buffer(1)]],
    constant MsmParams& params                  [[buffer(2)]],
    constant uint& n_segments                   [[buffer(3)]],
    constant uint& n_windows                    [[buffer(4)]],
    uint tid                                    [[thread_position_in_grid]]
) {
    uint total = n_segments * n_windows;
    if (tid >= total) return;
    uint window_idx = tid / n_segments;
    uint seg_idx = tid % n_segments;

    uint n_buckets = params.n_buckets;
    uint seg_size = (n_buckets + n_segments - 1) / n_segments;
    uint bucket_base = window_idx * n_buckets;

    int hi_s = int(n_buckets) - int(seg_idx * seg_size);
    int lo_raw_s = int((seg_idx + 1) * seg_size);
    int lo_s = (lo_raw_s >= int(n_buckets)) ? 1 : (int(n_buckets) - lo_raw_s);
    if (lo_s < 1) lo_s = 1;
    if (hi_s <= lo_s) {
        segment_results[tid] = point_identity();
        return;
    }

    PointProjective running = point_identity();
    PointProjective sum = point_identity();

    uint hi = uint(hi_s);
    uint lo = uint(lo_s);
    for (uint i = hi - 1; i >= lo; i--) {
        PointProjective bucket = buckets[bucket_base + i];
        if (!point_is_identity(bucket)) {
            if (point_is_identity(running)) {
                running = bucket;
            } else {
                running = point_add(running, bucket);
            }
        }
        if (!point_is_identity(running)) {
            if (point_is_identity(sum)) {
                sum = running;
            } else {
                sum = point_add(sum, running);
            }
        }
        if (i == lo) break;
    }

    uint weight = lo - 1;
    if (weight > 0 && !point_is_identity(running)) {
        PointProjective weighted = point_identity();
        PointProjective base = running;
        uint k = weight;
        while (k > 0) {
            if (k & 1u) {
                if (point_is_identity(weighted)) {
                    weighted = base;
                } else {
                    weighted = point_add(weighted, base);
                }
            }
            base = point_double(base);
            k >>= 1;
        }
        if (point_is_identity(sum)) {
            sum = weighted;
        } else {
            sum = point_add(sum, weighted);
        }
    }

    segment_results[tid] = sum;
}

// Phase 3: Serial reduction of segment results per window (one thread per window)
// Avoids threadgroup memory patterns that trigger Metal compiler miscompilation
kernel void msm_combine_segments(
    device const PointProjective* segment_results [[buffer(0)]],
    device PointProjective* window_results        [[buffer(1)]],
    constant uint& n_segments                     [[buffer(2)]],
    uint tid                                      [[thread_position_in_grid]]
) {
    uint base = tid * n_segments;
    PointProjective sum = point_identity();
    for (uint s = 0; s < n_segments; s++) {
        PointProjective seg = segment_results[base + s];
        if (!point_is_identity(seg)) {
            if (point_is_identity(sum)) {
                sum = seg;
            } else {
                sum = point_add(sum, seg);
            }
        }
    }
    window_results[tid] = sum;
}

// Horner's method to combine window results into final MSM result
kernel void msm_horner_combine(
    device const PointProjective* window_results [[buffer(0)]],
    device PointProjective* final_result         [[buffer(1)]],
    constant uint& n_windows                     [[buffer(2)]],
    constant uint& window_bits                   [[buffer(3)]],
    uint tid                                     [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    PointProjective result = window_results[n_windows - 1];
    for (int w = int(n_windows) - 2; w >= 0; w--) {
        for (uint b = 0; b < window_bits; b++) {
            result = point_double(result);
        }
        PointProjective wr = window_results[w];
        if (!point_is_identity(wr)) {
            if (point_is_identity(result)) {
                result = wr;
            } else {
                result = point_add(result, wr);
            }
        }
    }
    final_result[0] = result;
}

// Signed-digit scalar recoding: extract window digits with carry propagation
kernel void signed_digit_extract(
    device const uint* scalars          [[buffer(0)]],
    device uint* digits                 [[buffer(1)]],
    constant uint& n_points             [[buffer(2)]],
    constant uint& window_bits          [[buffer(3)]],
    constant uint& n_windows            [[buffer(4)]],
    uint gid                            [[thread_position_in_grid]]
) {
    if (gid >= n_points) return;

    const device uint* sp = scalars + gid * 8;
    uint mask = (1u << window_bits) - 1u;
    uint half_bk = 1u << (window_bits - 1u);
    uint full_bk = 1u << window_bits;
    uint carry = 0;

    for (uint w = 0; w < n_windows; w++) {
        uint bit_off = w * window_bits;
        uint limb_idx = bit_off / 32u;
        uint bit_pos = bit_off % 32u;

        uint idx = 0;
        if (limb_idx < 8u) {
            idx = sp[limb_idx] >> bit_pos;
            if (bit_pos + window_bits > 32u && limb_idx + 1u < 8u) {
                idx |= sp[limb_idx + 1u] << (32u - bit_pos);
            }
            idx &= mask;
        }

        uint digit = idx + carry;
        carry = 0;
        if (digit > half_bk) {
            digit = full_bk - digit;
            carry = 1;
            digits[w * n_points + gid] = digit | 0x80000000u;
        } else {
            digits[w * n_points + gid] = digit;
        }
    }
}
