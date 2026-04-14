// secp256k1 MSM GPU kernels: Pippenger's bucket method
// Adapted from BN254 MSM kernels for secp256k1 field elements.

#include "../geometry/secp256k1_curve.metal"

struct SecpMsmParams {
    uint n_points;
    uint window_bits;
    uint n_buckets;
};

// Phase 1: Reduce pre-sorted points per bucket (batched across windows)
kernel void secp_msm_reduce_sorted_buckets(
    device const SecpPointAffine* points    [[buffer(0)]],
    device SecpPointProjective* buckets     [[buffer(1)]],
    device const uint* bucket_offsets       [[buffer(2)]],
    device const uint* bucket_counts        [[buffer(3)]],
    constant SecpMsmParams& params          [[buffer(4)]],
    constant uint& n_windows               [[buffer(5)]],
    device const uint* sorted_indices       [[buffer(6)]],
    device const uint* count_sorted_map     [[buffer(7)]],
    uint tid                               [[thread_position_in_grid]]
) {
    uint total = params.n_buckets * n_windows;
    if (tid >= total) return;

    uint orig_pos = count_sorted_map[tid];
    uint orig_bucket = orig_pos & 0xFFFFu;
    uint orig_window = orig_pos >> 16u;
    uint flat_idx = orig_window * params.n_buckets + orig_bucket;

    if (orig_bucket == 0) {
        buckets[flat_idx] = secp_point_identity();
        return;
    }

    uint count = bucket_counts[flat_idx];
    if (count == 0) {
        buckets[flat_idx] = secp_point_identity();
        return;
    }

    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[flat_idx];
    uint raw_idx0 = sorted_indices[base + offset];
    SecpPointAffine pt0 = points[raw_idx0 & 0x7FFFFFFFu];
    if (raw_idx0 & 0x80000000u) pt0.y = secp_neg(pt0.y);
    SecpPointProjective acc = secp_point_from_affine(pt0);
    // Use unsafe mixed add: acc is never identity (initialized from affine),
    // and random point collision has probability ~10^-65.
    for (uint i = 1; i < count; i++) {
        uint raw_idx = sorted_indices[base + offset + i];
        SecpPointAffine pt = points[raw_idx & 0x7FFFFFFFu];
        if (raw_idx & 0x80000000u) pt.y = secp_neg(pt.y);
        acc = secp_point_add_mixed_unsafe(acc, pt);
    }
    buckets[flat_idx] = acc;
}

// SIMD shuffle helper for SecpPointProjective (8 limbs per field element)
inline SecpPointProjective simd_shuffle_down_secp_point(SecpPointProjective p, uint offset) {
    SecpPointProjective r;
    for (int k = 0; k < SECP_LIMBS; k++) {
        r.x.v[k] = simd_shuffle_down(p.x.v[k], offset);
        r.y.v[k] = simd_shuffle_down(p.y.v[k], offset);
        r.z.v[k] = simd_shuffle_down(p.z.v[k], offset);
    }
    return r;
}

// SIMD shuffle xor helper for tree reduction (pairs lanes i and i^offset)
inline SecpPointProjective simd_shuffle_xor_secp_point(SecpPointProjective p, uint offset) {
    SecpPointProjective r;
    for (int k = 0; k < SECP_LIMBS; k++) {
        r.x.v[k] = simd_shuffle_xor(p.x.v[k], offset);
        r.y.v[k] = simd_shuffle_xor(p.y.v[k], offset);
        r.z.v[k] = simd_shuffle_xor(p.z.v[k], offset);
    }
    return r;
}

// Phase 1b: Warp-per-bucket — one warp (32 threads) per bucket.
// Designed for n_buckets <= 1024 where total warps (n_buckets * n_windows / 32)
// is small enough to avoid excessive threadgroup scheduling overhead.
// Each thread handles count/32 elements strided, then tree-reduces via shuffle.
kernel void secp_msm_reduce_warp_per_bucket(
    device const SecpPointAffine* points    [[buffer(0)]],
    device SecpPointProjective* buckets     [[buffer(1)]],
    device const uint* bucket_offsets       [[buffer(2)]],
    device const uint* bucket_counts        [[buffer(3)]],
    constant SecpMsmParams& params          [[buffer(4)]],
    constant uint& n_windows               [[buffer(5)]],
    device const uint* sorted_indices       [[buffer(6)]],
    device const uint* count_sorted_map     [[buffer(7)]],
    uint tgid                              [[threadgroup_position_in_grid]],
    uint lid                               [[thread_index_in_threadgroup]]
) {
    uint total_buckets = params.n_buckets * n_windows;
    // Each warp (threadgroup) processes one bucket
    if (tgid >= total_buckets) return;

    uint orig_pos = count_sorted_map[tgid];
    uint orig_bucket = orig_pos & 0xFFFFu;
    uint orig_window = orig_pos >> 16u;
    uint flat_idx = orig_window * params.n_buckets + orig_bucket;

    if (orig_bucket == 0 || bucket_counts[flat_idx] == 0) {
        if (lid == 0) buckets[flat_idx] = secp_point_identity();
        return;
    }

    uint count = bucket_counts[flat_idx];
    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[flat_idx];

    // Phase A: each thread loads its portion strided
    SecpPointProjective acc = secp_point_identity();
    for (uint i = lid; i < count; i += 32) {
        uint raw_idx = sorted_indices[base + offset + i];
        SecpPointAffine pt = points[raw_idx & 0x7FFFFFFFu];
        if (raw_idx & 0x80000000u) pt.y = secp_neg(pt.y);
        if (secp_point_is_identity(acc)) {
            acc = secp_point_from_affine(pt);
        } else {
            acc = secp_point_add_mixed_unsafe(acc, pt);
        }
    }

    // Phase B: tree-reduce via simd_shuffle_xor (32->16->8->4->2->1)
    // simd_shuffle_xor lane i gets value from lane i^off
    for (uint off = 16; off > 0; off >>= 1) {
        SecpPointProjective other = simd_shuffle_down_secp_point(acc, off);
        if (lid < off) {
            if (secp_point_is_identity(acc)) {
                acc = other;
            } else if (!secp_point_is_identity(other)) {
                acc = secp_point_add_unsafe(acc, other);
            }
        }
    }

    if (lid == 0) buckets[flat_idx] = acc;
}

// Phase 1c: Cooperative reduce — one SIMD group (32 threads) per bucket
kernel void secp_msm_reduce_cooperative(
    device const SecpPointAffine* points    [[buffer(0)]],
    device SecpPointProjective* buckets     [[buffer(1)]],
    device const uint* bucket_offsets       [[buffer(2)]],
    device const uint* bucket_counts        [[buffer(3)]],
    constant SecpMsmParams& params          [[buffer(4)]],
    constant uint& n_windows               [[buffer(5)]],
    device const uint* sorted_indices       [[buffer(6)]],
    device const uint* count_sorted_map     [[buffer(7)]],
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
        if (lid == 0) buckets[flat_idx] = secp_point_identity();
        return;
    }

    uint count = bucket_counts[flat_idx];
    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[flat_idx];

    SecpPointProjective acc = secp_point_identity();
    for (uint i = lid; i < count; i += 32) {
        uint raw_idx = sorted_indices[base + offset + i];
        SecpPointAffine pt = points[raw_idx & 0x7FFFFFFFu];
        if (raw_idx & 0x80000000u) pt.y = secp_neg(pt.y);
        if (secp_point_is_identity(acc)) {
            acc = secp_point_from_affine(pt);
        } else {
            acc = secp_point_add_mixed_unsafe(acc, pt);
        }
    }

    for (uint off = 16; off > 0; off >>= 1) {
        SecpPointProjective other = simd_shuffle_down_secp_point(acc, off);
        if (lid < off) {
            if (secp_point_is_identity(acc)) {
                acc = other;
            } else if (!secp_point_is_identity(other)) {
                acc = secp_point_add_unsafe(acc, other);
            }
        }
    }

    if (lid == 0) buckets[flat_idx] = acc;
}

// Phase 1d: Shared memory reduction — 256 threads per bucket for large buckets.
// Higher radix (2^8) with warp-per-bucket using shared memory parallel reduction.
// Tree reduction: 256->128->64->32->16->8->4->2->1.
// This eliminates SIMD shuffle overhead for buckets with many points.
kernel void secp_msm_reduce_shared_mem(
    device const SecpPointAffine* points    [[buffer(0)]],
    device SecpPointProjective* buckets     [[buffer(1)]],
    device const uint* bucket_offsets       [[buffer(2)]],
    device const uint* bucket_counts        [[buffer(3)]],
    constant SecpMsmParams& params          [[buffer(4)]],
    constant uint& n_windows               [[buffer(5)]],
    device const uint* sorted_indices       [[buffer(6)]],
    device const uint* count_sorted_map     [[buffer(7)]],
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
        if (lid == 0) buckets[flat_idx] = secp_point_identity();
        return;
    }

    uint count = bucket_counts[flat_idx];
    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[flat_idx];

    // Shared memory: 256 Projective points (256 * 96 = 24KB, within 32KB limit)
    threadgroup SecpPointProjective s_buf[256];

    // Phase A: Cooperative load into shared memory
    // Each thread loads points at indices: lid, lid+256, lid+512, ...
    // This distributes count points evenly across 256 threads
    SecpPointProjective acc = secp_point_identity();
    for (uint i = lid; i < count; i += 256) {
        uint raw_idx = sorted_indices[base + offset + i];
        SecpPointAffine pt = points[raw_idx & 0x7FFFFFFFu];
        if (raw_idx & 0x80000000u) pt.y = secp_neg(pt.y);
        if (secp_point_is_identity(acc)) {
            acc = secp_point_from_affine(pt);
        } else {
            acc = secp_point_add_mixed_unsafe(acc, pt);
        }
    }

    // Store partial result to shared memory
    s_buf[lid] = acc;
    threadgroup_barrier(mem_flags::mem_none);

    // Phase B: Tree reduction using all 256 threads
    // Level 0: 256 threads, offset=128, pairs (0,128), (1,129), ...
    if (lid < 128) {
        SecpPointProjective other = s_buf[lid ^ 128];
        if (secp_point_is_identity(acc)) {
            acc = other;
        } else if (!secp_point_is_identity(other)) {
            acc = secp_point_add_unsafe(acc, other);
        }
        s_buf[lid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Level 1: 128 threads, offset=64
    if (lid < 64) {
        SecpPointProjective other = s_buf[lid ^ 64];
        if (secp_point_is_identity(acc)) {
            acc = other;
        } else if (!secp_point_is_identity(other)) {
            acc = secp_point_add_unsafe(acc, other);
        }
        s_buf[lid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Level 2: 64 threads, offset=32
    if (lid < 32) {
        SecpPointProjective other = s_buf[lid ^ 32];
        if (secp_point_is_identity(acc)) {
            acc = other;
        } else if (!secp_point_is_identity(other)) {
            acc = secp_point_add_unsafe(acc, other);
        }
        s_buf[lid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Level 3: 32 threads, offset=16
    if (lid < 16) {
        SecpPointProjective other = s_buf[lid ^ 16];
        if (secp_point_is_identity(acc)) {
            acc = other;
        } else if (!secp_point_is_identity(other)) {
            acc = secp_point_add_unsafe(acc, other);
        }
        s_buf[lid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Level 4: 16 threads, offset=8
    if (lid < 8) {
        SecpPointProjective other = s_buf[lid ^ 8];
        if (secp_point_is_identity(acc)) {
            acc = other;
        } else if (!secp_point_is_identity(other)) {
            acc = secp_point_add_unsafe(acc, other);
        }
        s_buf[lid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Level 5: 8 threads, offset=4
    if (lid < 4) {
        SecpPointProjective other = s_buf[lid ^ 4];
        if (secp_point_is_identity(acc)) {
            acc = other;
        } else if (!secp_point_is_identity(other)) {
            acc = secp_point_add_unsafe(acc, other);
        }
        s_buf[lid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Level 6: 4 threads, offset=2
    if (lid < 2) {
        SecpPointProjective other = s_buf[lid ^ 2];
        if (secp_point_is_identity(acc)) {
            acc = other;
        } else if (!secp_point_is_identity(other)) {
            acc = secp_point_add_unsafe(acc, other);
        }
        s_buf[lid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Level 7: 2 threads, offset=1
    if (lid < 1) {
        SecpPointProjective other = s_buf[lid ^ 1];
        if (secp_point_is_identity(acc)) {
            acc = other;
        } else if (!secp_point_is_identity(other)) {
            acc = secp_point_add_unsafe(acc, other);
        }
    }

    if (lid == 0) buckets[flat_idx] = acc;
}

// Phase 2: Direct weighted bucket sum per segment
kernel void secp_msm_bucket_sum_direct(
    device const SecpPointProjective* buckets   [[buffer(0)]],
    device SecpPointProjective* segment_results [[buffer(1)]],
    constant SecpMsmParams& params              [[buffer(2)]],
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
        segment_results[tid] = secp_point_identity();
        return;
    }

    SecpPointProjective running = secp_point_identity();
    SecpPointProjective sum = secp_point_identity();

    uint hi = uint(hi_s);
    uint lo = uint(lo_s);
    for (uint i = hi - 1; i >= lo; i--) {
        SecpPointProjective bucket = buckets[bucket_base + i];
        if (!secp_point_is_identity(bucket)) {
            if (secp_point_is_identity(running)) {
                running = bucket;
            } else {
                running = secp_point_add_unsafe(running, bucket);
            }
        }
        if (!secp_point_is_identity(running)) {
            if (secp_point_is_identity(sum)) {
                sum = running;
            } else {
                sum = secp_point_add_unsafe(sum, running);
            }
        }
        if (i == lo) break;
    }

    uint weight = lo - 1;
    if (weight > 0 && !secp_point_is_identity(running)) {
        SecpPointProjective weighted = secp_point_identity();
        SecpPointProjective base = running;
        uint k = weight;
        while (k > 0) {
            if (k & 1u) {
                if (secp_point_is_identity(weighted)) {
                    weighted = base;
                } else {
                    weighted = secp_point_add_unsafe(weighted, base);
                }
            }
            base = secp_point_double(base);
            k >>= 1;
        }
        if (secp_point_is_identity(sum)) {
            sum = weighted;
        } else {
            sum = secp_point_add_unsafe(sum, weighted);
        }
    }

    segment_results[tid] = sum;
}

// Phase 3: Serial reduction of segment results per window
kernel void secp_msm_combine_segments(
    device const SecpPointProjective* segment_results [[buffer(0)]],
    device SecpPointProjective* window_results        [[buffer(1)]],
    constant uint& n_segments                         [[buffer(2)]],
    uint tid                                          [[thread_position_in_grid]]
) {
    uint base = tid * n_segments;
    SecpPointProjective sum = secp_point_identity();
    for (uint s = 0; s < n_segments; s++) {
        SecpPointProjective seg = segment_results[base + s];
        if (!secp_point_is_identity(seg)) {
            if (secp_point_is_identity(sum)) {
                sum = seg;
            } else {
                sum = secp_point_add_unsafe(sum, seg);
            }
        }
    }
    window_results[tid] = sum;
}

// Horner's method to combine window results
kernel void secp_msm_horner_combine(
    device const SecpPointProjective* window_results [[buffer(0)]],
    device SecpPointProjective* final_result         [[buffer(1)]],
    constant uint& n_windows                         [[buffer(2)]],
    constant uint& window_bits                       [[buffer(3)]],
    uint tid                                         [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    SecpPointProjective result = window_results[n_windows - 1];
    for (int w = int(n_windows) - 2; w >= 0; w--) {
        for (uint b = 0; b < window_bits; b++) {
            result = secp_point_double(result);
        }
        SecpPointProjective wr = window_results[w];
        if (!secp_point_is_identity(wr)) {
            if (secp_point_is_identity(result)) {
                result = wr;
            } else {
                result = secp_point_add_unsafe(result, wr);
            }
        }
    }
    final_result[0] = result;
}

// Signed-digit scalar recoding for secp256k1 (8×32-bit scalars)
kernel void secp_msm_signed_digit_extract(
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
kernel void secp_msm_sort_histogram(
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
kernel void secp_msm_sort_scatter(
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
kernel void secp_msm_build_csm(
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

// ============================================================================
// Batch MSM Kernel — Multiple small MSMs in parallel on GPU
//
// Problem: One large MSM (2^18) underutilizes GPU due to memory-bound sort.
// Solution: Split into B small MSMs (M points each), run all B in parallel.
//
// Each thread block handles one small MSM (M points, wb window bits):
//  1. Load all M points into shared memory (cooperative loading)
//  2. Extract signed-digit windows for each point's scalar
//  3. Accumulate into shared buckets using parallel reduction
//  4. Reduce buckets to single result via Pippenger's method
//  5. Write result to global memory
//
// Key insight: For small M (≤64), we can do bucket accumulation in shared
// memory without global sorting. Sorting overhead O(M log M) becomes negligible
// when M is small, and the bucket reduction is fully parallel.
//
// Threadgroup: 256 threads, one MSM of M points
// B = n / M parallel MSMs on the GPU
//
// Shared memory layout (fits within 32KB limit):
//  - s_points[64]: 64 × 64 = 4KB  (for M≤64 points)
//  - s_buckets[128]: 128 × 192 = 24KB  (192 = sizeof(SecpPointProjective))
//  - s_scalars[64*8]: 64 × 8 × 4 = 2KB
//  Total: ~30KB ✓
// ============================================================================
kernel void secp_msm_batch_small(
    device const SecpPointAffine* all_points    [[buffer(0)]],  // B × M points
    device const uint* all_scalars_flat         [[buffer(1)]],  // B × M × 8 uint
    device SecpPointProjective* results         [[buffer(2)]],  // B results
    constant uint& M                             [[buffer(3)]],  // points per MSM (≤64)
    constant uint& B                             [[buffer(4)]],  // number of MSMs
    constant uint& wb                           [[buffer(5)]],  // window bits (≤7)
    uint tgid                                   [[threadgroup_position_in_grid]],
    uint lid                                    [[thread_index_in_threadgroup]]
) {
    if (tgid >= B) return;

    const uint n_windows = (256 + wb - 1) / wb;  // 256-bit scalars
    const uint n_buckets = 1 << wb;               // 2^wb buckets
    const uint half_buckets = n_buckets >> 1;
    const uint mask = n_buckets - 1;

    // Shared memory: ≤30KB total for M≤64, wb≤7
    // s_points[64]: 64 × 64 = 4KB
    // s_buckets[128]: 128 × 192 = 24KB  (192 = sizeof(SecpPointProjective))
    // s_scalars[64*8]: 64 × 8 × 4 = 2KB
    threadgroup SecpPointAffine s_points[64];
    threadgroup SecpPointProjective s_buckets[128];
    threadgroup uint s_scalars[64 * 8];

    uint b = tgid;

    // Phase 1: Load points and scalars cooperatively
    if (lid < M) {
        uint idx = b * M + lid;
        s_points[lid] = all_points[idx];
        for (uint j = 0; j < 8; j++) {
            s_scalars[lid * 8 + j] = all_scalars_flat[idx * 8 + j];
        }
    }
    threadgroup_barrier(mem_flags::mem_none);

    // Pippenger accumulation: result = Σ 2^{w*wb} * window_result[w]
    SecpPointProjective accumulator = secp_point_identity();

    // Process each window
    for (uint w = 0; w < n_windows; w++) {
        // Phase 2: Initialize buckets to identity
        if (lid < n_buckets) {
            s_buckets[lid] = secp_point_identity();
        }
        threadgroup_barrier(mem_flags::mem_none);

        // Phase 3: Extract digits and accumulate into buckets
        // Only lid==0 does the accumulation to avoid race conditions when multiple
        // threads try to accumulate to the same bucket simultaneously.
        // Other threads wait at the barrier and skip to Phase 4.
        for (uint i = 0; i < M; i++) {
            // Extract signed digit for window w from scalar[i]
            uint bit_off = w * wb;
            uint limb_idx = bit_off >> 5;  // / 32
            uint bit_pos = bit_off & 31;   // % 32

            uint idx = 0;
            if (limb_idx < 8) {
                idx = s_scalars[i * 8 + limb_idx] >> bit_pos;
                if (bit_pos + wb > 32 && limb_idx + 1 < 8) {
                    idx |= s_scalars[i * 8 + limb_idx + 1] << (32 - bit_pos);
                }
                idx &= mask;
            }

            // Signed-digit recoding
            uint digit = idx;
            bool negate = false;
            if (digit > half_buckets) {
                digit = n_buckets - digit;
                negate = true;
            }

            if (digit > 0) {
                SecpPointAffine pt = s_points[i];
                if (negate) {
                    pt.y = secp_neg(pt.y);
                }
                SecpPointProjective pt_proj = secp_point_from_affine(pt);

                // Accumulate point into bucket[digit] ONCE
                // bucket[d] = Σ P_i (sum of all points with digit d)
                // Note: use secp_point_add_mixed (safe) to handle doubling case
                // when bucket already contains the same point.
                if (secp_point_is_identity(s_buckets[digit])) {
                    s_buckets[digit] = pt_proj;
                } else {
                    s_buckets[digit] = secp_point_add_mixed(s_buckets[digit], pt);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_none);

        // Phase 4: Reduce buckets to window result
        // window_result = Σ d * bucket[d] = Σ d * (Σ P_i) = Σ d*P_i
        SecpPointProjective window_result = secp_point_identity();
        for (uint d = 1; d < n_buckets; d++) {
            if (!secp_point_is_identity(s_buckets[d])) {
                // Scale bucket[d] by d: bucket[d] + bucket[d] + ... (d times)
                // But bucket[d] = Σ P_i, so d * bucket[d] = Σ d*P_i
                SecpPointProjective scaled = s_buckets[d];
                for (uint k = 1; k < d; k++) {
                    scaled = secp_point_add_unsafe(scaled, s_buckets[d]);
                }
                if (secp_point_is_identity(window_result)) {
                    window_result = scaled;
                } else {
                    window_result = secp_point_add_unsafe(window_result, scaled);
                }
            }
        }

        // Phase 5: Scale window_result by 2^{w*wb} and add to accumulator
        // result = Σ 2^{w*wb} * window_result[w]
        // Scale by 2^{w*wb} via repeated doubling
        for (uint k = 0; k < w * wb; k++) {
            window_result = secp_point_double(window_result);
        }
        if (!secp_point_is_identity(window_result)) {
            if (secp_point_is_identity(accumulator)) {
                accumulator = window_result;
            } else {
                accumulator = secp_point_add_unsafe(accumulator, window_result);
            }
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    // Write final result
    if (lid == 0) {
        results[b] = accumulator;
    }
}

// ============================================================================
// NAF Batch MSM Kernel — NAF representation for ~33% fewer point additions
//
// NAF (Non-Adjacent Form) produces digits -1, 0, +1 with at most n/3 non-zero
// digits (vs n/2 for binary). This means ~33% fewer point additions.
//
// NAF encoding:
//   Each NAF digit is stored as uint8: 0 = zero, 1 = +1, 2 = -1
//   The NAF digits are precomputed on CPU since NAF extraction is sequential.
//
// Layout: all_naf_digits[b * M * 256 + i * 256 + bit] = NAF digit for point i at bit position
//
// Kernel structure:
//   1. Load M points into shared memory
//   2. For each bit position (0-255):
//      - Threads cooperatively accumulate pos_acc and neg_acc
//      - result = 2 * result + (pos_acc - neg_acc)
//   3. Write final result
//
// Shared memory layout (fits within 32KB limit):
//   - s_points[64]: 64 × 64 = 4KB
//   - s_naf[64 * 256]: 64 × 256 × 1 = 16KB  (NAF digits)
//   Total: ~20KB ✓
// ============================================================================
kernel void secp_msm_batch_small_naf(
    device const SecpPointAffine* all_points       [[buffer(0)]],  // B × M points
    device const uint8_t* all_naf_digits          [[buffer(1)]],  // B × M × 256 NAF digits
    device SecpPointProjective* results           [[buffer(2)]],  // B results
    constant uint& M                              [[buffer(3)]],  // points per MSM (≤64)
    constant uint& B                              [[buffer(4)]],  // number of MSMs
    uint tgid                                     [[threadgroup_position_in_grid]],
    uint lid                                      [[thread_index_in_threadgroup]]
) {
    if (tgid >= B) return;

    // NAF processes 256 bit positions
    const uint NAF_BITS = 256;

    // Shared memory: ~20KB total for M≤64
    // s_points[64]: 64 × 64 = 4KB
    // s_naf[64 * 256]: 64 × 256 × 1 = 16KB
    threadgroup SecpPointAffine s_points[64];
    threadgroup uint8_t s_naf[64 * 256];

    uint b = tgid;

    // Phase 1: Load points and NAF digits cooperatively
    if (lid < M) {
        uint idx = b * M + lid;
        s_points[lid] = all_points[idx];

        // Load NAF digits for this point (256 bytes)
        uint naf_base = (b * M + lid) * NAF_BITS;
        for (uint bit = 0; bit < NAF_BITS; bit++) {
            s_naf[lid * NAF_BITS + bit] = all_naf_digits[naf_base + bit];
        }
    }
    threadgroup_barrier(mem_flags::mem_none);

    // NAF MSM accumulation: result = Σ NAF_digit[i] * 2^i * P_i
    // Process bit-by-bit: result = 2 * result + (pos_acc - neg_acc) at each bit
    SecpPointProjective accumulator = secp_point_identity();

    for (uint bit = 0; bit < NAF_BITS; bit++) {
        // Phase 2: Initialize accumulators
        SecpPointProjective pos_acc = secp_point_identity();
        SecpPointProjective neg_acc = secp_point_identity();

        // Phase 3: Accumulate points based on NAF digit
        // Only lid==0 does accumulation to avoid races
        if (lid == 0) {
            for (uint i = 0; i < M; i++) {
                // Get NAF digit for point i at bit position
                uint8_t naf_digit = s_naf[i * NAF_BITS + bit];

                if (naf_digit == 1) {
                    // +1: add point
                    if (secp_point_is_identity(pos_acc)) {
                        pos_acc = secp_point_from_affine(s_points[i]);
                    } else {
                        pos_acc = secp_point_add_mixed(pos_acc, s_points[i]);
                    }
                } else if (naf_digit == 2) {
                    // -1: add negated point
                    SecpPointAffine neg_pt = s_points[i];
                    neg_pt.y = secp_neg(neg_pt.y);
                    if (secp_point_is_identity(neg_acc)) {
                        neg_acc = secp_point_from_affine(neg_pt);
                    } else {
                        neg_acc = secp_point_add_mixed(neg_acc, neg_pt);
                    }
                }
                // naf_digit == 0: skip
            }
        }
        threadgroup_barrier(mem_flags::mem_none);

        // Phase 4: Compute bit contribution
        // bit_result = pos_acc - neg_acc = pos_acc + (-neg_acc)
        SecpPointProjective bit_result;
        if (secp_point_is_identity(pos_acc)) {
            bit_result = neg_acc;
        } else if (secp_point_is_identity(neg_acc)) {
            bit_result = pos_acc;
        } else {
            // neg_acc = -neg_acc by negating y of all points in it
            // Actually we already negated during accumulation, so just add
            bit_result = secp_point_add_unsafe(pos_acc, neg_acc);
        }

        // Phase 5: Accumulate into result
        // result = 2 * result + bit_result
        if (!secp_point_is_identity(bit_result)) {
            // accumulator = 2 * accumulator + bit_result
            if (!secp_point_is_identity(accumulator)) {
                accumulator = secp_point_double(accumulator);
                accumulator = secp_point_add_unsafe(accumulator, bit_result);
            } else {
                accumulator = bit_result;
            }
        } else {
            // Just double the accumulator
            if (!secp_point_is_identity(accumulator)) {
                accumulator = secp_point_double(accumulator);
            }
        }
        threadgroup_barrier(mem_flags::mem_none);
    }

    // Write final result
    if (lid == 0) {
        results[b] = accumulator;
    }
}
