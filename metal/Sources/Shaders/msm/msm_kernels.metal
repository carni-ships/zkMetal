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
    uint nb = 1u << params.window_bits;
    uint total = params.n_buckets * n_windows;
    if (tid >= total) return;

    uint orig_pos = count_sorted_map[tid];
    uint orig_bucket = orig_pos & (nb - 1u);

    if (orig_bucket == 0) {
        buckets[orig_pos] = point_identity();
        return;
    }

    uint count = bucket_counts[orig_pos];
    if (count == 0) {
        buckets[orig_pos] = point_identity();
        return;
    }

    uint orig_window = orig_pos >> params.window_bits;
    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[orig_pos];
    PointProjective acc = point_from_affine(points[sorted_indices[base + offset]]);
    for (uint i = 1; i < count; i++) {
        acc = point_add_mixed(acc, points[sorted_indices[base + offset + i]]);
    }
    buckets[orig_pos] = acc;
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
    uint nb = 1u << params.window_bits;
    uint total = params.n_buckets * n_windows;
    if (tgid >= total) return;

    uint orig_pos = count_sorted_map[tgid];
    uint orig_bucket = orig_pos & (nb - 1u);

    // Identity for bucket 0 or empty buckets
    if (orig_bucket == 0 || bucket_counts[orig_pos] == 0) {
        if (lid == 0) {
            buckets[orig_pos] = point_identity();
        }
        return;
    }

    uint count = bucket_counts[orig_pos];
    uint orig_window = orig_pos >> params.window_bits;
    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[orig_pos];

    // Each thread accumulates points at stride 32
    PointProjective acc = point_identity();
    for (uint i = lid; i < count; i += 32) {
        PointAffine pt = points[sorted_indices[base + offset + i]];
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
        buckets[orig_pos] = acc;
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

    uint n_buckets = 1u << params.window_bits;
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

// Phase 3: Tree reduction of segment results per window
kernel void msm_combine_segments(
    device const PointProjective* segment_results [[buffer(0)]],
    device PointProjective* window_results        [[buffer(1)]],
    constant uint& n_segments                     [[buffer(2)]],
    uint tgid                                     [[threadgroup_position_in_grid]],
    uint lid                                      [[thread_index_in_threadgroup]],
    uint tg_size                                  [[threads_per_threadgroup]]
) {
    threadgroup PointProjective shared_buf[256];

    uint base = tgid * n_segments;
    uint idx0 = lid * 2;
    uint idx1 = lid * 2 + 1;

    PointProjective v;
    if (idx0 >= n_segments) {
        v = point_identity();
    } else if (idx1 >= n_segments) {
        v = segment_results[base + idx0];
    } else {
        PointProjective a = segment_results[base + idx0];
        PointProjective b = segment_results[base + idx1];
        if (point_is_identity(a)) { v = b; }
        else if (point_is_identity(b)) { v = a; }
        else { v = point_add(a, b); }
    }
    shared_buf[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            PointProjective a = shared_buf[lid];
            PointProjective b = shared_buf[lid + stride];
            if (point_is_identity(a)) {
                shared_buf[lid] = b;
            } else if (!point_is_identity(b)) {
                shared_buf[lid] = point_add(a, b);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        window_results[tgid] = shared_buf[0];
    }
}
