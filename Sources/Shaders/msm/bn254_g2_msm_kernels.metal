// BN254 G2 MSM GPU kernels: Pippenger's bucket method for G2 points
// G2 points live on the twist curve over Fp2: y^2 = x^3 + 3/(9+u)
// Each coordinate is an Fp2 element (pair of Fp), so 4x wider than G1.

#include "../fields/bn254_fp.metal"

// ============================================================================
// Fp2 = Fp[u]/(u^2 + 1) — extension field arithmetic
// ============================================================================

struct G2Fp2 {
    Fp c0;
    Fp c1;
};

G2Fp2 g2fp2_zero() {
    G2Fp2 r; r.c0 = fp_zero(); r.c1 = fp_zero(); return r;
}

G2Fp2 g2fp2_one() {
    G2Fp2 r; r.c0 = fp_one(); r.c1 = fp_zero(); return r;
}

bool g2fp2_is_zero(G2Fp2 a) {
    return fp_is_zero(a.c0) && fp_is_zero(a.c1);
}

G2Fp2 g2fp2_add(G2Fp2 a, G2Fp2 b) {
    G2Fp2 r; r.c0 = fp_add(a.c0, b.c0); r.c1 = fp_add(a.c1, b.c1); return r;
}

G2Fp2 g2fp2_sub(G2Fp2 a, G2Fp2 b) {
    G2Fp2 r; r.c0 = fp_sub(a.c0, b.c0); r.c1 = fp_sub(a.c1, b.c1); return r;
}

G2Fp2 g2fp2_neg(G2Fp2 a) {
    G2Fp2 r; r.c0 = fp_neg(a.c0); r.c1 = fp_neg(a.c1); return r;
}

G2Fp2 g2fp2_double(G2Fp2 a) {
    G2Fp2 r; r.c0 = fp_double(a.c0); r.c1 = fp_double(a.c1); return r;
}

// (a0+a1*u)(b0+b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
// Karatsuba: c1 = (a0+a1)(b0+b1) - a0*b0 - a1*b1
G2Fp2 g2fp2_mul(G2Fp2 a, G2Fp2 b) {
    Fp t0 = fp_mul_karatsuba(a.c0, b.c0);
    Fp t1 = fp_mul_karatsuba(a.c1, b.c1);
    G2Fp2 r;
    r.c0 = fp_sub(t0, t1);
    r.c1 = fp_sub(fp_mul_karatsuba(fp_add(a.c0, a.c1), fp_add(b.c0, b.c1)), fp_add(t0, t1));
    return r;
}

// (a0+a1*u)^2 = (a0+a1)(a0-a1) + 2*a0*a1*u
G2Fp2 g2fp2_sqr(G2Fp2 a) {
    Fp t0 = fp_mul_karatsuba(a.c0, a.c1);
    G2Fp2 r;
    r.c0 = fp_mul_karatsuba(fp_add(a.c0, a.c1), fp_sub(a.c0, a.c1));
    r.c1 = fp_double(t0);
    return r;
}

// ============================================================================
// G2 point types (Jacobian projective with Fp2 coordinates)
// ============================================================================

struct G2PointAffine {
    G2Fp2 x;
    G2Fp2 y;
};

struct G2PointProjective {
    G2Fp2 x;
    G2Fp2 y;
    G2Fp2 z;
};

G2PointProjective g2_point_identity() {
    G2PointProjective p;
    p.x = g2fp2_one();
    p.y = g2fp2_one();
    p.z = g2fp2_zero();
    return p;
}

bool g2_point_is_identity(G2PointProjective p) {
    return g2fp2_is_zero(p.z);
}

G2PointProjective g2_point_from_affine(G2PointAffine a) {
    G2PointProjective p;
    p.x = a.x;
    p.y = a.y;
    p.z = g2fp2_one();
    return p;
}

// Point doubling: 4M + 6S + 7add (a=0 for BN254 twist: b' = 3/(9+u))
G2PointProjective g2_point_double(G2PointProjective p) {
    if (g2_point_is_identity(p)) return p;

    G2Fp2 a = g2fp2_sqr(p.x);
    G2Fp2 b = g2fp2_sqr(p.y);
    G2Fp2 c = g2fp2_sqr(b);

    G2Fp2 d = g2fp2_sub(g2fp2_sqr(g2fp2_add(p.x, b)), g2fp2_add(a, c));
    d = g2fp2_double(d);

    G2Fp2 e = g2fp2_add(g2fp2_double(a), a); // 3*X^2 (a=0 for twist)
    G2Fp2 f = g2fp2_sqr(e);

    G2PointProjective r;
    r.x = g2fp2_sub(f, g2fp2_double(d));
    r.y = g2fp2_sub(g2fp2_mul(e, g2fp2_sub(d, r.x)), g2fp2_double(g2fp2_double(g2fp2_double(c))));
    G2Fp2 yz = g2fp2_add(p.y, p.z);
    r.z = g2fp2_sub(g2fp2_sqr(yz), g2fp2_add(b, g2fp2_sqr(p.z)));
    return r;
}

// Mixed addition: projective + affine (Q.z = 1)
G2PointProjective g2_point_add_mixed(G2PointProjective p, G2PointAffine q) {
    if (g2_point_is_identity(p)) return g2_point_from_affine(q);

    G2Fp2 z1z1 = g2fp2_sqr(p.z);
    G2Fp2 u2 = g2fp2_mul(q.x, z1z1);
    G2Fp2 h = g2fp2_sub(u2, p.x);

    if (g2fp2_is_zero(h)) {
        G2Fp2 s2 = g2fp2_mul(q.y, g2fp2_mul(p.z, z1z1));
        G2Fp2 rr = g2fp2_double(g2fp2_sub(s2, p.y));
        if (g2fp2_is_zero(rr)) return g2_point_double(p);
        return g2_point_identity();
    }

    G2Fp2 s2 = g2fp2_mul(q.y, g2fp2_mul(p.z, z1z1));
    G2PointProjective result;
    result.z = g2fp2_double(g2fp2_mul(p.z, h));
    G2Fp2 hh = g2fp2_sqr(h);
    G2Fp2 i = g2fp2_double(g2fp2_double(hh));
    G2Fp2 v = g2fp2_mul(p.x, i);
    G2Fp2 j = g2fp2_mul(h, i);
    G2Fp2 rr = g2fp2_double(g2fp2_sub(s2, p.y));
    result.x = g2fp2_sub(g2fp2_sub(g2fp2_sqr(rr), j), g2fp2_double(v));
    result.y = g2fp2_sub(g2fp2_mul(rr, g2fp2_sub(v, result.x)),
                         g2fp2_double(g2fp2_mul(p.y, j)));
    return result;
}

// Unsafe mixed addition: no identity/doubling checks
G2PointProjective g2_point_add_mixed_unsafe(G2PointProjective p, G2PointAffine q) {
    G2Fp2 z1z1 = g2fp2_sqr(p.z);
    G2Fp2 u2 = g2fp2_mul(q.x, z1z1);
    G2Fp2 s2 = g2fp2_mul(q.y, g2fp2_mul(p.z, z1z1));
    G2Fp2 h = g2fp2_sub(u2, p.x);
    G2PointProjective result;
    result.z = g2fp2_double(g2fp2_mul(p.z, h));
    G2Fp2 hh = g2fp2_sqr(h);
    G2Fp2 i = g2fp2_double(g2fp2_double(hh));
    G2Fp2 v = g2fp2_mul(p.x, i);
    G2Fp2 j = g2fp2_mul(h, i);
    G2Fp2 rr = g2fp2_double(g2fp2_sub(s2, p.y));
    result.x = g2fp2_sub(g2fp2_sub(g2fp2_sqr(rr), j), g2fp2_double(v));
    result.y = g2fp2_sub(g2fp2_mul(rr, g2fp2_sub(v, result.x)),
                         g2fp2_double(g2fp2_mul(p.y, j)));
    return result;
}

// Full addition: projective + projective
G2PointProjective g2_point_add(G2PointProjective p, G2PointProjective q) {
    if (g2_point_is_identity(p)) return q;
    if (g2_point_is_identity(q)) return p;

    G2Fp2 z1z1 = g2fp2_sqr(p.z);
    G2Fp2 z2z2 = g2fp2_sqr(q.z);
    G2Fp2 u1 = g2fp2_mul(p.x, z2z2);
    G2Fp2 u2 = g2fp2_mul(q.x, z1z1);
    G2Fp2 s1 = g2fp2_mul(p.y, g2fp2_mul(q.z, z2z2));
    G2Fp2 s2 = g2fp2_mul(q.y, g2fp2_mul(p.z, z1z1));

    G2Fp2 h = g2fp2_sub(u2, u1);
    G2Fp2 rr = g2fp2_double(g2fp2_sub(s2, s1));

    if (g2fp2_is_zero(h)) {
        if (g2fp2_is_zero(rr)) return g2_point_double(p);
        return g2_point_identity();
    }

    G2PointProjective result;
    result.z = g2fp2_mul(g2fp2_double(g2fp2_mul(p.z, q.z)), h);

    G2Fp2 dh = g2fp2_double(h);
    G2Fp2 ii = g2fp2_sqr(dh);
    G2Fp2 v = g2fp2_mul(u1, ii);
    G2Fp2 j = g2fp2_mul(h, ii);

    result.x = g2fp2_sub(g2fp2_sub(g2fp2_sqr(rr), j), g2fp2_double(v));
    result.y = g2fp2_sub(g2fp2_mul(rr, g2fp2_sub(v, result.x)),
                         g2fp2_double(g2fp2_mul(s1, j)));
    return result;
}

// ============================================================================
// MSM Kernels
// ============================================================================

struct G2MsmParams {
    uint n_points;
    uint window_bits;
    uint n_buckets;
};

// Phase 1: Reduce pre-sorted points per bucket (batched across windows)
kernel void g2_msm_reduce_sorted_buckets(
    device const G2PointAffine* points           [[buffer(0)]],
    device G2PointProjective* buckets            [[buffer(1)]],
    device const uint* bucket_offsets            [[buffer(2)]],
    device const uint* bucket_counts             [[buffer(3)]],
    constant G2MsmParams& params                 [[buffer(4)]],
    constant uint& n_windows                     [[buffer(5)]],
    device const uint* sorted_indices            [[buffer(6)]],
    device const uint* count_sorted_map          [[buffer(7)]],
    uint tid                                     [[thread_position_in_grid]]
) {
    uint total = params.n_buckets * n_windows;
    if (tid >= total) return;

    uint orig_pos = count_sorted_map[tid];
    uint orig_bucket = orig_pos & 0xFFFFu;
    uint orig_window = orig_pos >> 16u;
    uint flat_idx = orig_window * params.n_buckets + orig_bucket;

    if (orig_bucket == 0) {
        buckets[flat_idx] = g2_point_identity();
        return;
    }

    uint count = bucket_counts[flat_idx];
    if (count == 0) {
        buckets[flat_idx] = g2_point_identity();
        return;
    }

    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[flat_idx];
    uint raw_idx0 = sorted_indices[base + offset];
    G2PointAffine pt0 = points[raw_idx0 & 0x7FFFFFFFu];
    if (raw_idx0 & 0x80000000u) pt0.y = g2fp2_neg(pt0.y);
    G2PointProjective acc = g2_point_from_affine(pt0);
    for (uint i = 1; i < count; i++) {
        uint raw_idx = sorted_indices[base + offset + i];
        G2PointAffine pt = points[raw_idx & 0x7FFFFFFFu];
        if (raw_idx & 0x80000000u) pt.y = g2fp2_neg(pt.y);
        acc = g2_point_add_mixed(acc, pt);
    }
    buckets[flat_idx] = acc;
}

// Phase 2: Direct weighted bucket sum per segment
kernel void g2_msm_bucket_sum_direct(
    device const G2PointProjective* buckets       [[buffer(0)]],
    device G2PointProjective* segment_results     [[buffer(1)]],
    constant G2MsmParams& params                  [[buffer(2)]],
    constant uint& n_segments                     [[buffer(3)]],
    constant uint& n_windows                      [[buffer(4)]],
    uint tid                                      [[thread_position_in_grid]]
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
        segment_results[tid] = g2_point_identity();
        return;
    }

    G2PointProjective running = g2_point_identity();
    G2PointProjective sum = g2_point_identity();

    uint hi = uint(hi_s);
    uint lo = uint(lo_s);
    for (uint i = hi - 1; i >= lo; i--) {
        G2PointProjective bucket = buckets[bucket_base + i];
        if (!g2_point_is_identity(bucket)) {
            if (g2_point_is_identity(running)) {
                running = bucket;
            } else {
                running = g2_point_add(running, bucket);
            }
        }
        if (!g2_point_is_identity(running)) {
            if (g2_point_is_identity(sum)) {
                sum = running;
            } else {
                sum = g2_point_add(sum, running);
            }
        }
        if (i == lo) break;
    }

    uint weight = lo - 1;
    if (weight > 0 && !g2_point_is_identity(running)) {
        G2PointProjective weighted = g2_point_identity();
        G2PointProjective base = running;
        uint k = weight;
        while (k > 0) {
            if (k & 1u) {
                if (g2_point_is_identity(weighted)) {
                    weighted = base;
                } else {
                    weighted = g2_point_add(weighted, base);
                }
            }
            base = g2_point_double(base);
            k >>= 1;
        }
        if (g2_point_is_identity(sum)) {
            sum = weighted;
        } else {
            sum = g2_point_add(sum, weighted);
        }
    }

    segment_results[tid] = sum;
}

// Phase 3: Serial reduction of segment results per window
kernel void g2_msm_combine_segments(
    device const G2PointProjective* segment_results [[buffer(0)]],
    device G2PointProjective* window_results        [[buffer(1)]],
    constant uint& n_segments                       [[buffer(2)]],
    uint tid                                        [[thread_position_in_grid]]
) {
    uint base = tid * n_segments;
    G2PointProjective sum = g2_point_identity();
    for (uint s = 0; s < n_segments; s++) {
        G2PointProjective seg = segment_results[base + s];
        if (!g2_point_is_identity(seg)) {
            if (g2_point_is_identity(sum)) {
                sum = seg;
            } else {
                sum = g2_point_add(sum, seg);
            }
        }
    }
    window_results[tid] = sum;
}

// Horner's method to combine window results
kernel void g2_msm_horner_combine(
    device const G2PointProjective* window_results [[buffer(0)]],
    device G2PointProjective* final_result         [[buffer(1)]],
    constant uint& n_windows                       [[buffer(2)]],
    constant uint& window_bits                     [[buffer(3)]],
    uint tid                                       [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    G2PointProjective result = window_results[n_windows - 1];
    for (int w = int(n_windows) - 2; w >= 0; w--) {
        for (uint b = 0; b < window_bits; b++) {
            result = g2_point_double(result);
        }
        G2PointProjective wr = window_results[w];
        if (!g2_point_is_identity(wr)) {
            if (g2_point_is_identity(result)) {
                result = wr;
            } else {
                result = g2_point_add(result, wr);
            }
        }
    }
    final_result[0] = result;
}

// Signed-digit scalar recoding (same as G1 — scalars are Fr elements)
kernel void g2_signed_digit_extract(
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

// GPU counting sort kernels (identical logic to G1, different prefix)
kernel void g2_sort_histogram(
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

kernel void g2_sort_scatter(
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

kernel void g2_build_csm(
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
