// Grumpkin MSM GPU kernels: Pippenger's bucket method
// Adapted from Vesta MSM kernels for Grumpkin curve (BN254 Fr base field).

#include "../geometry/grumpkin_curve.metal"

struct GrumpkinMsmParams {
    uint n_points;
    uint window_bits;
    uint n_buckets;
};

// Phase 1: Reduce pre-sorted points per bucket
kernel void grumpkin_msm_reduce_sorted_buckets(
    device const GrumpkinPointAffine* points      [[buffer(0)]],
    device GrumpkinPointProjective* buckets       [[buffer(1)]],
    device const uint* bucket_offsets             [[buffer(2)]],
    device const uint* bucket_counts              [[buffer(3)]],
    constant GrumpkinMsmParams& params            [[buffer(4)]],
    constant uint& n_windows                      [[buffer(5)]],
    device const uint* sorted_indices             [[buffer(6)]],
    device const uint* count_sorted_map           [[buffer(7)]],
    uint tid                                      [[thread_position_in_grid]]
) {
    uint total = params.n_buckets * n_windows;
    if (tid >= total) return;

    uint orig_pos = count_sorted_map[tid];
    uint orig_bucket = orig_pos & 0xFFFFu;
    uint orig_window = orig_pos >> 16u;
    uint flat_idx = orig_window * params.n_buckets + orig_bucket;

    if (orig_bucket == 0) {
        buckets[flat_idx] = grumpkin_point_identity();
        return;
    }

    uint count = bucket_counts[flat_idx];
    if (count == 0) {
        buckets[flat_idx] = grumpkin_point_identity();
        return;
    }

    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[flat_idx];
    uint raw_idx0 = sorted_indices[base + offset];
    GrumpkinPointAffine pt0 = points[raw_idx0 & 0x7FFFFFFFu];
    if (raw_idx0 & 0x80000000u) pt0.y = fr_neg(pt0.y);
    GrumpkinPointProjective acc = grumpkin_point_from_affine(pt0);
    for (uint i = 1; i < count; i++) {
        uint raw_idx = sorted_indices[base + offset + i];
        GrumpkinPointAffine pt = points[raw_idx & 0x7FFFFFFFu];
        if (raw_idx & 0x80000000u) pt.y = fr_neg(pt.y);
        acc = grumpkin_point_add_mixed_unsafe(acc, pt);
    }
    buckets[flat_idx] = acc;
}

// SIMD shuffle helper
inline GrumpkinPointProjective simd_shuffle_down_grumpkin(GrumpkinPointProjective p, uint offset) {
    GrumpkinPointProjective r;
    for (int k = 0; k < 8; k++) {
        r.x.v[k] = simd_shuffle_down(p.x.v[k], offset);
        r.y.v[k] = simd_shuffle_down(p.y.v[k], offset);
        r.z.v[k] = simd_shuffle_down(p.z.v[k], offset);
    }
    return r;
}

// Phase 1b: Cooperative reduce
kernel void grumpkin_msm_reduce_cooperative(
    device const GrumpkinPointAffine* points      [[buffer(0)]],
    device GrumpkinPointProjective* buckets       [[buffer(1)]],
    device const uint* bucket_offsets             [[buffer(2)]],
    device const uint* bucket_counts              [[buffer(3)]],
    constant GrumpkinMsmParams& params            [[buffer(4)]],
    constant uint& n_windows                      [[buffer(5)]],
    device const uint* sorted_indices             [[buffer(6)]],
    device const uint* count_sorted_map           [[buffer(7)]],
    uint tgid                                     [[threadgroup_position_in_grid]],
    uint lid                                      [[thread_index_in_threadgroup]]
) {
    uint total = params.n_buckets * n_windows;
    if (tgid >= total) return;

    uint orig_pos = count_sorted_map[tgid];
    uint orig_bucket = orig_pos & 0xFFFFu;
    uint orig_window = orig_pos >> 16u;
    uint flat_idx = orig_window * params.n_buckets + orig_bucket;

    if (orig_bucket == 0 || bucket_counts[flat_idx] == 0) {
        if (lid == 0) buckets[flat_idx] = grumpkin_point_identity();
        return;
    }

    uint count = bucket_counts[flat_idx];
    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[flat_idx];

    GrumpkinPointProjective acc = grumpkin_point_identity();
    for (uint i = lid; i < count; i += 32) {
        uint raw_idx = sorted_indices[base + offset + i];
        GrumpkinPointAffine pt = points[raw_idx & 0x7FFFFFFFu];
        if (raw_idx & 0x80000000u) pt.y = fr_neg(pt.y);
        if (grumpkin_point_is_identity(acc)) {
            acc = grumpkin_point_from_affine(pt);
        } else {
            acc = grumpkin_point_add_mixed(acc, pt);
        }
    }

    for (uint off = 16; off > 0; off >>= 1) {
        GrumpkinPointProjective other = simd_shuffle_down_grumpkin(acc, off);
        if (lid < off) {
            if (grumpkin_point_is_identity(acc)) {
                acc = other;
            } else if (!grumpkin_point_is_identity(other)) {
                acc = grumpkin_point_add(acc, other);
            }
        }
    }

    if (lid == 0) buckets[flat_idx] = acc;
}

// Phase 2: Direct weighted bucket sum per segment
kernel void grumpkin_msm_bucket_sum_direct(
    device const GrumpkinPointProjective* buckets   [[buffer(0)]],
    device GrumpkinPointProjective* segment_results [[buffer(1)]],
    constant GrumpkinMsmParams& params              [[buffer(2)]],
    constant uint& n_segments                       [[buffer(3)]],
    constant uint& n_windows                        [[buffer(4)]],
    uint tid                                        [[thread_position_in_grid]]
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
        segment_results[tid] = grumpkin_point_identity();
        return;
    }

    GrumpkinPointProjective running = grumpkin_point_identity();
    GrumpkinPointProjective sum = grumpkin_point_identity();

    uint hi = uint(hi_s);
    uint lo = uint(lo_s);
    for (uint i = hi - 1; i >= lo; i--) {
        GrumpkinPointProjective bucket = buckets[bucket_base + i];
        if (!grumpkin_point_is_identity(bucket)) {
            if (grumpkin_point_is_identity(running)) {
                running = bucket;
            } else {
                running = grumpkin_point_add(running, bucket);
            }
        }
        if (!grumpkin_point_is_identity(running)) {
            if (grumpkin_point_is_identity(sum)) {
                sum = running;
            } else {
                sum = grumpkin_point_add(sum, running);
            }
        }
        if (i == lo) break;
    }

    uint weight = lo - 1;
    if (weight > 0 && !grumpkin_point_is_identity(running)) {
        GrumpkinPointProjective weighted = grumpkin_point_identity();
        GrumpkinPointProjective base = running;
        uint k = weight;
        while (k > 0) {
            if (k & 1u) {
                if (grumpkin_point_is_identity(weighted)) {
                    weighted = base;
                } else {
                    weighted = grumpkin_point_add(weighted, base);
                }
            }
            base = grumpkin_point_double(base);
            k >>= 1;
        }
        if (grumpkin_point_is_identity(sum)) {
            sum = weighted;
        } else {
            sum = grumpkin_point_add(sum, weighted);
        }
    }

    segment_results[tid] = sum;
}

// Phase 3: Combine segments
kernel void grumpkin_msm_combine_segments(
    device const GrumpkinPointProjective* segment_results [[buffer(0)]],
    device GrumpkinPointProjective* window_results        [[buffer(1)]],
    constant uint& n_segments                             [[buffer(2)]],
    uint tid                                              [[thread_position_in_grid]]
) {
    uint base = tid * n_segments;
    GrumpkinPointProjective sum = grumpkin_point_identity();
    for (uint s = 0; s < n_segments; s++) {
        GrumpkinPointProjective seg = segment_results[base + s];
        if (!grumpkin_point_is_identity(seg)) {
            if (grumpkin_point_is_identity(sum)) {
                sum = seg;
            } else {
                sum = grumpkin_point_add(sum, seg);
            }
        }
    }
    window_results[tid] = sum;
}

// Horner combine
kernel void grumpkin_msm_horner_combine(
    device const GrumpkinPointProjective* window_results [[buffer(0)]],
    device GrumpkinPointProjective* final_result         [[buffer(1)]],
    constant uint& n_windows                             [[buffer(2)]],
    constant uint& window_bits                           [[buffer(3)]],
    uint tid                                             [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    GrumpkinPointProjective result = window_results[n_windows - 1];
    for (int w = int(n_windows) - 2; w >= 0; w--) {
        for (uint b = 0; b < window_bits; b++) {
            result = grumpkin_point_double(result);
        }
        GrumpkinPointProjective wr = window_results[w];
        if (!grumpkin_point_is_identity(wr)) {
            if (grumpkin_point_is_identity(result)) {
                result = wr;
            } else {
                result = grumpkin_point_add(result, wr);
            }
        }
    }
    final_result[0] = result;
}

// Signed-digit scalar recoding
kernel void grumpkin_signed_digit_extract(
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

// GPU counting sort kernels
kernel void grumpkin_sort_histogram(
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

kernel void grumpkin_sort_scatter(
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

kernel void grumpkin_build_csm(
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
