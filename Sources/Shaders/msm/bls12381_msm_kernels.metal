// BLS12-381 MSM GPU kernels: Pippenger's bucket method
// Adapted from BLS12-377 MSM kernels for BLS12-381 base field Fp.

#include "../geometry/bls12381_curve.metal"

struct Msm381Params {
    uint n_points;
    uint window_bits;
    uint n_buckets;
};

// Phase 1: Reduce pre-sorted points per bucket (batched across windows)
kernel void msm381_reduce_sorted_buckets(
    device const Point381Affine* points    [[buffer(0)]],
    device Point381Projective* buckets     [[buffer(1)]],
    device const uint* bucket_offsets      [[buffer(2)]],
    device const uint* bucket_counts       [[buffer(3)]],
    constant Msm381Params& params          [[buffer(4)]],
    constant uint& n_windows               [[buffer(5)]],
    device const uint* sorted_indices      [[buffer(6)]],
    device const uint* count_sorted_map    [[buffer(7)]],
    uint tid                               [[thread_position_in_grid]]
) {
    uint total = params.n_buckets * n_windows;
    if (tid >= total) return;

    uint orig_pos = count_sorted_map[tid];
    uint orig_bucket = orig_pos & 0xFFFFu;
    uint orig_window = orig_pos >> 16u;
    uint flat_idx = orig_window * params.n_buckets + orig_bucket;

    if (orig_bucket == 0) {
        buckets[flat_idx] = point381_identity();
        return;
    }

    uint count = bucket_counts[flat_idx];
    if (count == 0) {
        buckets[flat_idx] = point381_identity();
        return;
    }

    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[flat_idx];
    uint raw_idx0 = sorted_indices[base + offset];
    Point381Affine pt0 = points[raw_idx0 & 0x7FFFFFFFu];
    if (raw_idx0 & 0x80000000u) pt0.y = fp381_neg(pt0.y);
    Point381Projective acc = point381_from_affine(pt0);
    for (uint i = 1; i < count; i++) {
        uint raw_idx = sorted_indices[base + offset + i];
        Point381Affine pt = points[raw_idx & 0x7FFFFFFFu];
        if (raw_idx & 0x80000000u) pt.y = fp381_neg(pt.y);
        acc = point381_add_mixed(acc, pt);
    }
    buckets[flat_idx] = acc;
}

// SIMD shuffle helper for Point381Projective (12 limbs per field element)
inline Point381Projective simd_shuffle_down_point381(Point381Projective p, uint offset) {
    Point381Projective r;
    for (int k = 0; k < FP381_LIMBS; k++) {
        r.x.v[k] = simd_shuffle_down(p.x.v[k], offset);
        r.y.v[k] = simd_shuffle_down(p.y.v[k], offset);
        r.z.v[k] = simd_shuffle_down(p.z.v[k], offset);
    }
    return r;
}

// Phase 1b: Cooperative reduce -- one SIMD group (32 threads) per bucket
kernel void msm381_reduce_cooperative(
    device const Point381Affine* points    [[buffer(0)]],
    device Point381Projective* buckets     [[buffer(1)]],
    device const uint* bucket_offsets      [[buffer(2)]],
    device const uint* bucket_counts       [[buffer(3)]],
    constant Msm381Params& params          [[buffer(4)]],
    constant uint& n_windows               [[buffer(5)]],
    device const uint* sorted_indices      [[buffer(6)]],
    device const uint* count_sorted_map    [[buffer(7)]],
    uint tgid                              [[threadgroup_position_in_grid]],
    uint lid                               [[thread_index_in_threadgroup]]
) {
    uint total = params.n_buckets * n_windows;
    if (tgid >= total) return;

    uint orig_pos = count_sorted_map[tgid];
    uint orig_bucket = orig_pos & 0xFFFFu;
    uint orig_window = orig_pos >> 16u;
    uint flat_idx = orig_window * params.n_buckets + orig_bucket;

    if (orig_bucket == 0 || bucket_counts[flat_idx] == 0) {
        if (lid == 0) buckets[flat_idx] = point381_identity();
        return;
    }

    uint count = bucket_counts[flat_idx];
    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[flat_idx];

    Point381Projective acc = point381_identity();
    for (uint i = lid; i < count; i += 32) {
        uint raw_idx = sorted_indices[base + offset + i];
        Point381Affine pt = points[raw_idx & 0x7FFFFFFFu];
        if (raw_idx & 0x80000000u) pt.y = fp381_neg(pt.y);
        if (point381_is_identity(acc)) {
            acc = point381_from_affine(pt);
        } else {
            acc = point381_add_mixed(acc, pt);
        }
    }

    for (uint off = 16; off > 0; off >>= 1) {
        Point381Projective other = simd_shuffle_down_point381(acc, off);
        if (lid < off) {
            if (point381_is_identity(acc)) {
                acc = other;
            } else if (!point381_is_identity(other)) {
                acc = point381_add(acc, other);
            }
        }
    }

    if (lid == 0) buckets[flat_idx] = acc;
}

// Phase 2: Direct weighted bucket sum per segment
kernel void msm381_bucket_sum_direct(
    device const Point381Projective* buckets   [[buffer(0)]],
    device Point381Projective* segment_results [[buffer(1)]],
    constant Msm381Params& params              [[buffer(2)]],
    constant uint& n_segments                  [[buffer(3)]],
    constant uint& n_windows                   [[buffer(4)]],
    uint tid                                   [[thread_position_in_grid]]
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
        segment_results[tid] = point381_identity();
        return;
    }

    Point381Projective running = point381_identity();
    Point381Projective sum = point381_identity();

    uint hi = uint(hi_s);
    uint lo = uint(lo_s);
    for (uint i = hi - 1; i >= lo; i--) {
        Point381Projective bucket = buckets[bucket_base + i];
        if (!point381_is_identity(bucket)) {
            if (point381_is_identity(running)) {
                running = bucket;
            } else {
                running = point381_add(running, bucket);
            }
        }
        if (!point381_is_identity(running)) {
            if (point381_is_identity(sum)) {
                sum = running;
            } else {
                sum = point381_add(sum, running);
            }
        }
        if (i == lo) break;
    }

    uint weight = lo - 1;
    if (weight > 0 && !point381_is_identity(running)) {
        Point381Projective weighted = point381_identity();
        Point381Projective base = running;
        uint k = weight;
        while (k > 0) {
            if (k & 1u) {
                if (point381_is_identity(weighted)) {
                    weighted = base;
                } else {
                    weighted = point381_add(weighted, base);
                }
            }
            base = point381_double(base);
            k >>= 1;
        }
        if (point381_is_identity(sum)) {
            sum = weighted;
        } else {
            sum = point381_add(sum, weighted);
        }
    }

    segment_results[tid] = sum;
}

// Phase 2: SIMD tree-reduced bucket sum per segment
// Replaces sequential point additions with SIMD shuffle tree reduction.
// Pattern mirrors msm381_reduce_cooperative: stride-loop then SIMD shuffle tree.
kernel void msm381_bucket_sum_cooperative(
    device const Point381Projective* buckets   [[buffer(0)]],
    device Point381Projective* segment_results [[buffer(1)]],
    constant Msm381Params& params              [[buffer(2)]],
    constant uint& n_segments                  [[buffer(3)]],
    constant uint& n_windows                   [[buffer(4)]],
    uint tid                                   [[thread_position_in_grid]],
    uint tgid                                  [[threadgroup_position_in_grid]],
    uint lid                                   [[thread_index_in_threadgroup]],
    uint tg_size                               [[threads_per_threadgroup]]
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

    uint hi = uint(hi_s);
    uint lo = uint(lo_s);
    uint count = (hi > lo) ? (hi - lo) : 0;

    // Phase 1: Accumulate running + sum with strided threads
    Point381Projective acc_running = point381_identity();
    Point381Projective acc_sum = point381_identity();

    for (uint c = lid; c < count; c += tg_size) {
        uint bucket_idx = hi - 1 - c;
        Point381Projective bucket = buckets[bucket_base + bucket_idx];
        if (!point381_is_identity(bucket)) {
            if (point381_is_identity(acc_running)) {
                acc_running = bucket;
            } else {
                acc_running = point381_add(acc_running, bucket);
            }
        }
        if (!point381_is_identity(acc_running)) {
            if (point381_is_identity(acc_sum)) {
                acc_sum = acc_running;
            } else {
                acc_sum = point381_add(acc_sum, acc_running);
            }
        }
    }

    // Phase 2: SIMD shuffle tree reduction within threadgroup
    threadgroup Point381Projective shared_running[64];
    threadgroup Point381Projective shared_sum[64];

    shared_running[lid] = acc_running;
    shared_sum[lid] = acc_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint off = tg_size >> 1; off > 16; off >>= 1) {
        if (lid < off) {
            Point381Projective a = shared_running[lid];
            Point381Projective b = shared_running[lid + off];
            if (point381_is_identity(a)) {
                shared_running[lid] = b;
            } else if (!point381_is_identity(b)) {
                shared_running[lid] = point381_add(a, b);
            }

            Point381Projective c = shared_sum[lid];
            Point381Projective d = shared_sum[lid + off];
            if (point381_is_identity(c)) {
                shared_sum[lid] = d;
            } else if (!point381_is_identity(d)) {
                shared_sum[lid] = point381_add(c, d);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final warp-level reduction (SIMD shuffles)
    if (lid < 32) {
        for (uint off = 16; off > 0; off >>= 1) {
            Point381Projective a = simd_shuffle_down_point381(acc_running, off);
            if (!point381_is_identity(acc_running) && !point381_is_identity(a)) {
                acc_running = point381_add(acc_running, a);
            } else if (point381_is_identity(acc_running)) {
                acc_running = a;
            }

            Point381Projective b = simd_shuffle_down_point381(acc_sum, off);
            if (!point381_is_identity(acc_sum) && !point381_is_identity(b)) {
                acc_sum = point381_add(acc_sum, b);
            } else if (point381_is_identity(acc_sum)) {
                acc_sum = b;
            }
        }
    }

    // Phase 3: Weighted bucket + write result
    if (lid == 0) {
        uint weight = lo - 1;
        Point381Projective final_sum = acc_sum;

        if (weight > 0 && !point381_is_identity(acc_running)) {
            Point381Projective weighted = point381_identity();
            Point381Projective base = acc_running;
            uint k = weight;
            while (k > 0) {
                if (k & 1u) {
                    if (point381_is_identity(weighted)) {
                        weighted = base;
                    } else {
                        weighted = point381_add(weighted, base);
                    }
                }
                base = point381_double(base);
                k >>= 1;
            }
            if (point381_is_identity(final_sum)) {
                final_sum = weighted;
            } else {
                final_sum = point381_add(final_sum, weighted);
            }
        }
        segment_results[tid] = final_sum;
    }
}

// Phase 3: Serial reduction of segment results per window
kernel void msm381_combine_segments(
    device const Point381Projective* segment_results [[buffer(0)]],
    device Point381Projective* window_results        [[buffer(1)]],
    constant uint& n_segments                        [[buffer(2)]],
    uint tid                                         [[thread_position_in_grid]]
) {
    uint base = tid * n_segments;
    Point381Projective sum = point381_identity();
    for (uint s = 0; s < n_segments; s++) {
        Point381Projective seg = segment_results[base + s];
        if (!point381_is_identity(seg)) {
            if (point381_is_identity(sum)) {
                sum = seg;
            } else {
                sum = point381_add(sum, seg);
            }
        }
    }
    window_results[tid] = sum;
}

// Horner's method to combine window results
kernel void msm381_horner_combine(
    device const Point381Projective* window_results [[buffer(0)]],
    device Point381Projective* final_result         [[buffer(1)]],
    constant uint& n_windows                        [[buffer(2)]],
    constant uint& window_bits                      [[buffer(3)]],
    uint tid                                        [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    Point381Projective result = window_results[n_windows - 1];
    for (int w = int(n_windows) - 2; w >= 0; w--) {
        for (uint b = 0; b < window_bits; b++) {
            result = point381_double(result);
        }
        Point381Projective wr = window_results[w];
        if (!point381_is_identity(wr)) {
            if (point381_is_identity(result)) {
                result = wr;
            } else {
                result = point381_add(result, wr);
            }
        }
    }
    final_result[0] = result;
}

// Signed-digit scalar recoding for Fr381 (8x32-bit scalars)
kernel void msm381_signed_digit_extract(
    device const uint* scalars         [[buffer(0)]],
    device uint* digits                [[buffer(1)]],
    constant uint& n_points            [[buffer(2)]],
    constant uint& window_bits         [[buffer(3)]],
    constant uint& n_windows           [[buffer(4)]],
    uint gid                           [[thread_position_in_grid]]
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

// GPU counting sort: histogram phase
kernel void msm381_sort_histogram(
    device const uint* digits          [[buffer(0)]],
    device atomic_uint* counts         [[buffer(1)]],
    constant uint& n_points            [[buffer(2)]],
    constant uint& n_buckets           [[buffer(3)]],
    constant uint& n_windows           [[buffer(4)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= n_points * n_windows) return;
    uint w = gid / n_points;
    uint i = gid % n_points;
    uint digit = digits[w * n_points + i] & 0x7FFFFFFFu;
    atomic_fetch_add_explicit(&counts[w * n_buckets + digit], 1u, memory_order_relaxed);
}

// GPU counting sort: scatter phase
kernel void msm381_sort_scatter(
    device const uint* digits          [[buffer(0)]],
    device uint* sorted_indices        [[buffer(1)]],
    device atomic_uint* positions      [[buffer(2)]],
    constant uint& n_points            [[buffer(3)]],
    constant uint& n_buckets           [[buffer(4)]],
    constant uint& n_windows           [[buffer(5)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= n_points * n_windows) return;
    uint w = gid / n_points;
    uint i = gid % n_points;
    uint raw = digits[w * n_points + i];
    uint digit = raw & 0x7FFFFFFFu;
    if (digit == 0) return;
    uint pos = atomic_fetch_add_explicit(&positions[w * n_buckets + digit], 1u, memory_order_relaxed);
    uint idx = i;
    if (raw & 0x80000000u) idx |= 0x80000000u;
    sorted_indices[w * n_points + pos] = idx;
}

// GPU counting sort: build count-sorted map
kernel void msm381_build_csm(
    device const uint* counts          [[buffer(0)]],
    device uint* csm                   [[buffer(1)]],
    device uint* offsets               [[buffer(2)]],
    constant uint& n_buckets           [[buffer(3)]],
    constant uint& n_windows           [[buffer(4)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= n_windows) return;
    uint w = gid;
    uint wOff = w * n_buckets;

    uint max_count = 0;
    for (uint i = 0; i < n_buckets; i++) {
        uint c = counts[wOff + i];
        if (c > max_count) max_count = c;
    }

    for (uint i = 0; i <= max_count && i < n_buckets; i++) {
        csm[wOff + i] = 0;
    }
    for (uint i = 0; i < n_buckets; i++) {
        uint c = counts[wOff + i];
        csm[wOff + c]++;
    }
    uint running = 0;
    for (uint c = max_count; ; c--) {
        uint cnt = csm[wOff + c];
        csm[wOff + c] = running;
        running += cnt;
        if (c == 0) break;
    }
    for (uint i = 0; i <= max_count && i < n_buckets; i++) {
        offsets[wOff + i] = csm[wOff + i];
    }
    for (uint i = 0; i < n_buckets; i++) {
        uint c = counts[wOff + i];
        uint dest = offsets[wOff + c];
        offsets[wOff + c] = dest + 1;
        csm[wOff + dest] = (w << 16u) | i;
    }
}
