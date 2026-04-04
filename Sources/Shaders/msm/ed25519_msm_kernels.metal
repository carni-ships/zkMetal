// Ed25519 MSM GPU kernels: Pippenger's bucket method
// Adapted for twisted Edwards curve with extended coordinates.

#include "../geometry/ed25519_curve.metal"

struct EdMsmParams {
    uint n_points;
    uint window_bits;
    uint n_buckets;
};

// Phase 1: Reduce pre-sorted points per bucket (batched across windows)
kernel void ed_msm_reduce_sorted_buckets(
    device const EdPointAffine* points      [[buffer(0)]],
    device EdPointExtended* buckets         [[buffer(1)]],
    device const uint* bucket_offsets       [[buffer(2)]],
    device const uint* bucket_counts        [[buffer(3)]],
    constant EdMsmParams& params            [[buffer(4)]],
    constant uint& n_windows               [[buffer(5)]],
    device const uint* sorted_indices       [[buffer(6)]],
    device const uint* count_sorted_map     [[buffer(7)]],
    constant EdFp& d2                      [[buffer(8)]],
    uint tid                               [[thread_position_in_grid]]
) {
    uint total = params.n_buckets * n_windows;
    if (tid >= total) return;

    uint orig_pos = count_sorted_map[tid];
    uint orig_bucket = orig_pos & 0xFFFFu;
    uint orig_window = orig_pos >> 16u;
    uint flat_idx = orig_window * params.n_buckets + orig_bucket;

    if (orig_bucket == 0) {
        buckets[flat_idx] = ed_point_identity();
        return;
    }

    uint count = bucket_counts[flat_idx];
    if (count == 0) {
        buckets[flat_idx] = ed_point_identity();
        return;
    }

    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[flat_idx];
    uint raw_idx0 = sorted_indices[base + offset];
    EdPointAffine pt0 = points[raw_idx0 & 0x7FFFFFFFu];
    if (raw_idx0 & 0x80000000u) pt0.x = ed_neg(pt0.x);  // negate x for Edwards
    EdPointExtended acc = ed_point_from_affine(pt0);
    for (uint i = 1; i < count; i++) {
        uint raw_idx = sorted_indices[base + offset + i];
        EdPointAffine pt = points[raw_idx & 0x7FFFFFFFu];
        if (raw_idx & 0x80000000u) pt.x = ed_neg(pt.x);
        acc = ed_point_add_mixed(acc, pt, d2);
    }
    buckets[flat_idx] = acc;
}

// Phase 2: Bucket sum (running sum accumulation per window segment)
kernel void ed_msm_bucket_sum_direct(
    device const EdPointExtended* buckets   [[buffer(0)]],
    device EdPointExtended* segment_results [[buffer(1)]],
    constant EdMsmParams& params            [[buffer(2)]],
    constant uint& n_segments              [[buffer(3)]],
    constant uint& n_windows_batch         [[buffer(4)]],
    constant EdFp& d2                      [[buffer(5)]],
    uint gid                               [[thread_position_in_grid]]
) {
    uint total_segs = n_segments * n_windows_batch;
    if (gid >= total_segs) return;

    uint win = gid / n_segments;
    uint seg = gid % n_segments;
    uint nb = params.n_buckets;

    uint seg_size = (nb - 1 + n_segments - 1) / n_segments;
    uint start = seg * seg_size + 1;
    uint end = min(start + seg_size, nb);
    if (start >= nb) {
        segment_results[gid] = ed_point_identity();
        return;
    }

    uint base = win * nb;
    EdPointExtended running = ed_point_identity();
    EdPointExtended partial = ed_point_identity();

    for (uint b = end - 1; b >= start; b--) {
        EdPointExtended bkt = buckets[base + b];
        if (!ed_point_is_identity(bkt)) {
            running = ed_point_add(running, bkt, d2);
        }
        partial = ed_point_add(partial, running, d2);
        if (b == start) break;
    }

    segment_results[gid] = partial;
}

// Phase 3: Combine segments per window
kernel void ed_msm_combine_segments(
    device const EdPointExtended* segment_results [[buffer(0)]],
    device EdPointExtended* window_results       [[buffer(1)]],
    constant uint& n_segments                    [[buffer(2)]],
    constant EdFp& d2                            [[buffer(3)]],
    uint gid                                     [[thread_position_in_grid]]
) {
    uint base = gid * n_segments;
    EdPointExtended result = ed_point_identity();
    for (uint i = 0; i < n_segments; i++) {
        EdPointExtended seg = segment_results[base + i];
        if (!ed_point_is_identity(seg)) {
            result = ed_point_add(result, seg, d2);
        }
    }
    window_results[gid] = result;
}

// Signed-digit extraction kernel
kernel void ed_msm_signed_digit_extract(
    device const uint* scalars     [[buffer(0)]],
    device uint* signed_digits     [[buffer(1)]],
    constant uint& n_points        [[buffer(2)]],
    constant uint& window_bits     [[buffer(3)]],
    constant uint& n_windows       [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= n_points) return;

    device const uint* sc = scalars + gid * 8;
    uint mask = (1u << window_bits) - 1u;
    uint half = 1u << (window_bits - 1u);
    uint full = 1u << window_bits;
    uint carry = 0;

    for (uint w = 0; w < n_windows; w++) {
        uint bit_off = w * window_bits;
        uint limb_idx = bit_off / 32u;
        uint bit_pos = bit_off % 32u;

        uint idx = 0;
        if (limb_idx < 8u) {
            idx = sc[limb_idx] >> bit_pos;
            if (bit_pos + window_bits > 32u && limb_idx + 1u < 8u) {
                idx |= sc[limb_idx + 1u] << (32u - bit_pos);
            }
            idx &= mask;
        }

        uint digit = idx + carry;
        carry = 0;
        if (digit > half) {
            digit = full - digit;
            carry = 1;
            signed_digits[w * n_points + gid] = digit | 0x80000000u;
        } else {
            signed_digits[w * n_points + gid] = digit;
        }
    }
}
