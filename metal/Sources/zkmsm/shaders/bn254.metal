// BN254 Multi-Scalar Multiplication on Metal GPU
//
// Implements Pippenger's bucket method for MSM on the BN254 curve.
// BN254 field prime: p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
// BN254 curve: y^2 = x^3 + 3
//
// Field elements are represented as 4x64-bit limbs in Montgomery form.
// Point coordinates use projective (X, Y, Z) representation.

#include <metal_stdlib>
using namespace metal;

// --- BN254 Field Constants (Montgomery form) ---
// p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
// Each limb is 64 bits. p = limb[0] + limb[1]*2^64 + limb[2]*2^128 + limb[3]*2^192

// Using uint4 as a pair of uint2 to represent 256-bit values
// We pack 4x64-bit limbs as 8x32-bit values

struct Fp {
    uint v[8]; // 256-bit value as 8x32-bit limbs (little-endian)
};

struct PointAffine {
    Fp x;
    Fp y;
};

struct PointProjective {
    Fp x;
    Fp y;
    Fp z;
};

// BN254 prime modulus p (little-endian 32-bit limbs)
constant uint P[8] = {
    0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91,
    0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
};

// Montgomery parameter: R^2 mod p (for converting to Montgomery form)
constant uint R2[8] = {
    0x538afa89, 0xf32cfc5b, 0xd44501fb, 0xb5e71911,
    0x0a417ff6, 0x47ab1eff, 0xcab8351f, 0x06d89f71
};

// Montgomery inverse: -p^(-1) mod 2^32
constant uint INV = 0xe4866389u;

// --- 256-bit Arithmetic ---

// Add two 256-bit numbers, return result and carry
Fp fp_add_raw(Fp a, Fp b, thread uint &carry) {
    Fp r;
    ulong c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c += ulong(a.v[i]) + ulong(b.v[i]);
        r.v[i] = uint(c & 0xFFFFFFFF);
        c >>= 32;
    }
    carry = uint(c);
    return r;
}

// Subtract b from a, return result and borrow
Fp fp_sub_raw(Fp a, Fp b, thread uint &borrow) {
    Fp r;
    long c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        c += long(a.v[i]) - long(b.v[i]);
        r.v[i] = uint(c & 0xFFFFFFFF);
        c >>= 32;
    }
    borrow = (c < 0) ? 1u : 0u;
    return r;
}

// Compare a >= b
bool fp_gte(Fp a, Fp b) {
    for (int i = 7; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true; // equal
}

// Load modulus P into Fp (inline constants)
Fp fp_modulus() {
    Fp r;
    r.v[0] = 0xd87cfd47u; r.v[1] = 0x3c208c16u; r.v[2] = 0x6871ca8du; r.v[3] = 0x97816a91u;
    r.v[4] = 0x8181585du; r.v[5] = 0xb85045b6u; r.v[6] = 0xe131a029u; r.v[7] = 0x30644e72u;
    return r;
}

// Modular addition: (a + b) mod p
Fp fp_add(Fp a, Fp b) {
    uint carry;
    Fp r = fp_add_raw(a, b, carry);
    Fp p = fp_modulus();
    if (carry || fp_gte(r, p)) {
        uint borrow;
        r = fp_sub_raw(r, p, borrow);
    }
    return r;
}

// Modular subtraction: (a - b) mod p
Fp fp_sub(Fp a, Fp b) {
    uint borrow;
    Fp r = fp_sub_raw(a, b, borrow);
    if (borrow) {
        uint carry;
        r = fp_add_raw(r, fp_modulus(), carry);
    }
    return r;
}

// Zero check
bool fp_is_zero(Fp a) {
    for (int i = 0; i < 8; i++) {
        if (a.v[i] != 0) return false;
    }
    return true;
}

// Return zero Fp
Fp fp_zero() {
    Fp r;
    for (int i = 0; i < 8; i++) r.v[i] = 0;
    return r;
}

// Modular negation: p - a (returns 0 if a == 0)
Fp fp_neg(Fp a) {
    if (fp_is_zero(a)) return a;
    uint borrow;
    return fp_sub_raw(fp_modulus(), a, borrow);
}

// Return Fp(1) in Montgomery form
Fp fp_one() {
    // R mod p = 2^256 mod p (little-endian 32-bit limbs)
    Fp r;
    r.v[0] = 0xc58f0d9d; r.v[1] = 0xd35d438d; r.v[2] = 0xf5c70b3d;
    r.v[3] = 0x0a78eb28; r.v[4] = 0x7879462c; r.v[5] = 0x666ea36f;
    r.v[6] = 0x9a07df2f; r.v[7] = 0x0e0a77c1;
    return r;
}

// --- Montgomery Multiplication ---
// Computes (a * b * R^-1) mod p using CIOS method

Fp fp_mul(Fp a, Fp b) {
    // CIOS (Coarsely Integrated Operand Scanning) Montgomery multiplication
    // Working in 32-bit limbs for Metal GPU compatibility
    uint t[10]; // n+2 limbs to handle carries safely
    #pragma unroll
    for (int i = 0; i < 10; i++) t[i] = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        // Step 1: t += a[i] * b
        ulong carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            carry += ulong(t[j]) + ulong(a.v[i]) * ulong(b.v[j]);
            t[j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ulong ext = ulong(t[8]) + carry;
        t[8] = uint(ext & 0xFFFFFFFF);
        t[9] = uint(ext >> 32);

        // Step 2: Montgomery reduction — t += m * P, then shift right by 32
        uint m = t[0] * INV;
        carry = ulong(t[0]) + ulong(m) * ulong(P[0]);
        carry >>= 32; // t[0] becomes 0 by construction (that's the point of INV)
        #pragma unroll
        for (int j = 1; j < 8; j++) {
            carry += ulong(t[j]) + ulong(m) * ulong(P[j]);
            t[j - 1] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ext = ulong(t[8]) + carry;
        t[7] = uint(ext & 0xFFFFFFFF);
        t[8] = t[9] + uint(ext >> 32);
        t[9] = 0;
    }

    Fp r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i];

    // Final reduction: if r >= p, subtract p
    if (t[8] != 0 || fp_gte(r, fp_modulus())) {
        uint borrow;
        r = fp_sub_raw(r, fp_modulus(), borrow);
    }
    return r;
}

// Modular squaring — CIOS style (same structure as fp_mul for better unrolling)
Fp fp_sqr(Fp a) {
    uint t[10];
    #pragma unroll
    for (int i = 0; i < 10; i++) t[i] = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        ulong carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            carry += ulong(t[j]) + ulong(a.v[i]) * ulong(a.v[j]);
            t[j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ulong ext = ulong(t[8]) + carry;
        t[8] = uint(ext & 0xFFFFFFFF);
        t[9] = uint(ext >> 32);

        uint m = t[0] * INV;
        carry = ulong(t[0]) + ulong(m) * ulong(P[0]);
        carry >>= 32;
        #pragma unroll
        for (int j = 1; j < 8; j++) {
            carry += ulong(t[j]) + ulong(m) * ulong(P[j]);
            t[j - 1] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ext = ulong(t[8]) + carry;
        t[7] = uint(ext & 0xFFFFFFFF);
        t[8] = t[9] + uint(ext >> 32);
        t[9] = 0;
    }

    Fp r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i];
    if (t[8] != 0 || fp_gte(r, fp_modulus())) {
        uint borrow;
        r = fp_sub_raw(r, fp_modulus(), borrow);
    }
    return r;
}

// Double in field: 2*a mod p
Fp fp_double(Fp a) {
    return fp_add(a, a);
}

// --- Projective Point Operations ---
// BN254: y^2 = x^3 + 3
// Using Jacobian projective coordinates: (X, Y, Z) represents (X/Z^2, Y/Z^3)

// Point at infinity (identity)
PointProjective point_identity() {
    PointProjective p;
    p.x = fp_one();
    p.y = fp_one();
    p.z = fp_zero();
    return p;
}

bool point_is_identity(PointProjective p) {
    return fp_is_zero(p.z);
}

// Convert affine to projective
PointProjective point_from_affine(PointAffine a) {
    PointProjective p;
    p.x = a.x;
    p.y = a.y;
    p.z = fp_one();
    return p;
}

// Point doubling — noinline, flattened, 2YZ
__attribute__((noinline))
PointProjective point_double(PointProjective p) {
    if (point_is_identity(p)) return p;

    Fp t0 = fp_sqr(p.x);       // X^2
    Fp t1 = fp_sqr(p.y);       // Y^2
    Fp t2 = fp_sqr(t1);        // Y^4

    Fp t3 = fp_add(p.x, t1);
    t3 = fp_sqr(t3);
    Fp t4 = fp_add(t0, t2);
    t3 = fp_sub(t3, t4);
    t3 = fp_double(t3);        // d = 2*((X+Y^2)^2 - X^2 - Y^4)

    t4 = fp_double(t0);
    t4 = fp_add(t4, t0);       // e = 3*X^2

    Fp t5 = fp_sqr(t4);        // f = e^2

    PointProjective r;
    r.x = fp_double(t3);
    r.x = fp_sub(t5, r.x);     // X3 = f - 2*d

    r.z = fp_mul(p.y, p.z);
    r.z = fp_double(r.z);      // Z3 = 2*Y*Z

    t5 = fp_sub(t3, r.x);
    t5 = fp_mul(t4, t5);
    t2 = fp_double(t2);
    t2 = fp_double(t2);
    t2 = fp_double(t2);        // 8*Y^4
    r.y = fp_sub(t5, t2);      // Y3 = e*(d-X3) - 8*Y^4

    return r;
}

// Mixed addition: projective + affine — noinline, flattened
__attribute__((noinline))
PointProjective point_add_mixed(PointProjective p, PointAffine q) {
    if (point_is_identity(p)) return point_from_affine(q);

    Fp t0 = fp_sqr(p.z);           // Z1^2
    Fp t1 = fp_mul(q.x, t0);       // U2 = X2*Z1^2
    Fp t2 = fp_sub(t1, p.x);       // H = U2 - X1

    if (fp_is_zero(t2)) {
        Fp t3 = fp_mul(p.z, t0);
        t3 = fp_mul(q.y, t3);
        t3 = fp_sub(t3, p.y);
        t3 = fp_double(t3);
        if (fp_is_zero(t3)) return point_double(p);
        return point_identity();
    }

    Fp t3 = fp_mul(p.z, t0);       // Z1^3
    t3 = fp_mul(q.y, t3);          // S2 = Y2*Z1^3

    PointProjective r;
    r.z = fp_mul(p.z, t2);
    r.z = fp_double(r.z);          // Z3 = 2*Z1*H

    t0 = fp_sqr(t2);               // H^2
    t1 = fp_double(t0);
    t1 = fp_double(t1);            // I = 4*H^2
    Fp t4 = fp_mul(p.x, t1);       // V = X1*I
    Fp t5 = fp_mul(t2, t1);        // J = H*I

    t3 = fp_sub(t3, p.y);
    t3 = fp_double(t3);            // rr = 2*(S2-Y1)

    r.x = fp_sqr(t3);
    r.x = fp_sub(r.x, t5);
    t0 = fp_double(t4);
    r.x = fp_sub(r.x, t0);        // X3 = rr^2 - J - 2*V

    t1 = fp_sub(t4, r.x);
    t1 = fp_mul(t3, t1);
    t2 = fp_mul(p.y, t5);
    t2 = fp_double(t2);
    r.y = fp_sub(t1, t2);          // Y3 = rr*(V-X3) - 2*Y1*J

    return r;
}

// Specialized mixed addition when Z=1 (first addition per bucket)
__attribute__((noinline))
PointProjective point_add_mixed_z1(PointAffine p, PointAffine q) {
    Fp t0 = fp_sub(q.x, p.x);      // H = X2 - X1

    if (fp_is_zero(t0)) {
        Fp t1 = fp_sub(q.y, p.y);
        t1 = fp_double(t1);
        if (fp_is_zero(t1)) {
            PointProjective r;
            r.x = p.x; r.y = p.y; r.z = fp_one();
            return point_double(r);
        }
        return point_identity();
    }

    PointProjective r;
    r.z = fp_double(t0);            // Z3 = 2*H

    Fp t1 = fp_sqr(t0);             // H^2
    Fp t2 = fp_double(t1);
    t2 = fp_double(t2);             // I = 4*H^2
    Fp t3 = fp_mul(p.x, t2);        // V = X1*I
    Fp t4 = fp_mul(t0, t2);         // J = H*I

    t1 = fp_sub(q.y, p.y);
    t1 = fp_double(t1);             // rr = 2*(Y2-Y1)

    r.x = fp_sqr(t1);
    r.x = fp_sub(r.x, t4);
    t0 = fp_double(t3);
    r.x = fp_sub(r.x, t0);          // X3 = rr^2 - J - 2*V

    t2 = fp_sub(t3, r.x);
    t2 = fp_mul(t1, t2);
    Fp t5 = fp_mul(p.y, t4);
    t5 = fp_double(t5);
    r.y = fp_sub(t2, t5);           // Y3 = rr*(V-X3) - 2*Y1*J

    return r;
}

// Full point addition — noinline, flattened
__attribute__((noinline))
PointProjective point_add(PointProjective p, PointProjective q) {
    if (point_is_identity(p)) return q;
    if (point_is_identity(q)) return p;

    Fp t0 = fp_sqr(p.z);           // Z1^2
    Fp t1 = fp_sqr(q.z);           // Z2^2
    Fp t2 = fp_mul(p.x, t1);       // U1
    Fp t3 = fp_mul(q.x, t0);       // U2
    Fp t4 = fp_mul(q.z, t1);       // Z2^3
    t4 = fp_mul(p.y, t4);          // S1
    Fp t5 = fp_mul(p.z, t0);       // Z1^3
    t5 = fp_mul(q.y, t5);          // S2

    t0 = fp_sub(t3, t2);           // H
    t1 = fp_sub(t5, t4);
    t1 = fp_double(t1);            // rr

    if (fp_is_zero(t0)) {
        if (fp_is_zero(t1)) return point_double(p);
        return point_identity();
    }

    PointProjective r;
    r.z = fp_mul(p.z, q.z);
    r.z = fp_mul(r.z, t0);
    r.z = fp_double(r.z);          // Z3 = 2*Z1*Z2*H

    t3 = fp_double(t0);
    t3 = fp_sqr(t3);               // I = (2H)^2
    t5 = fp_mul(t2, t3);           // V = U1*I
    Fp t6 = fp_mul(t0, t3);        // J = H*I

    r.x = fp_sqr(t1);
    r.x = fp_sub(r.x, t6);
    t0 = fp_double(t5);
    r.x = fp_sub(r.x, t0);        // X3

    t3 = fp_sub(t5, r.x);
    t3 = fp_mul(t1, t3);
    t2 = fp_mul(t4, t6);
    t2 = fp_double(t2);
    r.y = fp_sub(t3, t2);          // Y3

    return r;
}

// --- MSM Kernel: Pippenger's Bucket Method (CPU-sorted, lock-free) ---
//
// Host sorts points by bucket index (CPU counting sort — fast for small keys).
// GPU kernels operate on pre-sorted data without any atomic operations.
//
// Phase 1 (msm_reduce_sorted_buckets): Each thread handles one bucket.
//   Points for bucket i are contiguous at sorted_points[offset[i]..offset[i]+count[i]).
//   The thread sums them and writes the result.
// Phase 2 (msm_bucket_sum_parallel): Segmented running sum over weighted buckets.
// Host combines window results with Horner's method.

struct MsmParams {
    uint n_points;       // number of points
    uint window_bits;    // bits per window (e.g., 16)
    uint n_buckets;      // effective number of buckets per window (may be half for signed digits)
};

// Phase 1: Reduce pre-sorted points per bucket. No atomics needed.
// Batched across all windows: tid = window_idx * n_buckets + bucket_idx
// Uses sorted indices to read from original points array (saves sort memcpy).
// sorted_indices layout: [window0: n_points][window1: n_points]...
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
    uint nb = params.n_buckets;
    uint total = nb * n_windows;
    if (tid >= total) return;

    // Unpack: upper 16 bits = window, lower 16 bits = bucket index
    uint packed = count_sorted_map[tid];
    uint orig_bucket = packed & 0xFFFFu;
    uint orig_window = packed >> 16u;
    uint orig_pos = orig_window * nb + orig_bucket;

    if (orig_bucket == 0) {
        buckets[orig_pos] = point_identity();
        return;
    }

    uint count = bucket_counts[orig_pos];
    if (count == 0) {
        buckets[orig_pos] = point_identity();
        return;
    }
    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[orig_pos];

    // Read point with conditional Y negation (no zero check — Y is never zero for curve points)
    uint raw0 = sorted_indices[base + offset];
    PointAffine first = points[raw0 & 0x7FFFFFFFu];
    if (raw0 & 0x80000000u) { uint borrow; first.y = fp_sub_raw(fp_modulus(), first.y, borrow); }

    PointProjective acc;
    if (count == 1) {
        acc = point_from_affine(first);
    } else {
        uint raw1 = sorted_indices[base + offset + 1];
        PointAffine second = points[raw1 & 0x7FFFFFFFu];
        if (raw1 & 0x80000000u) { uint borrow; second.y = fp_sub_raw(fp_modulus(), second.y, borrow); }
        acc = point_add_mixed_z1(first, second);
        for (uint i = 2; i < count; i++) {
            uint rawI = sorted_indices[base + offset + i];
            PointAffine pt = points[rawI & 0x7FFFFFFFu];
            if (rawI & 0x80000000u) { uint borrow; pt.y = fp_sub_raw(fp_modulus(), pt.y, borrow); }
            acc = point_add_mixed(acc, pt);
        }
    }
    buckets[orig_pos] = acc;
}

// SIMD shuffle helper for PointProjective (24 uint lanes)
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
// Each thread accumulates its portion with stride 32, then SIMD tree reduction merges.
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
    uint nb = params.n_buckets;
    uint total = nb * n_windows;
    if (tgid >= total) return;

    // Same packed encoding as msm_reduce_sorted_buckets
    uint packed = count_sorted_map[tgid];
    uint orig_bucket = packed & 0xFFFFu;
    uint orig_window = packed >> 16u;
    uint orig_pos = orig_window * nb + orig_bucket;

    // Identity for bucket 0 or empty buckets
    if (orig_bucket == 0 || bucket_counts[orig_pos] == 0) {
        if (lid == 0) {
            buckets[orig_pos] = point_identity();
        }
        return;
    }

    uint count = bucket_counts[orig_pos];
    uint base = orig_window * params.n_points;
    uint offset = bucket_offsets[orig_pos];

    // Each thread accumulates points at stride 32
    PointProjective acc = point_identity();
    for (uint i = lid; i < count; i += 32) {
        uint rawI = sorted_indices[base + offset + i];
        PointAffine pt = points[rawI & 0x7FFFFFFFu];
        if (rawI & 0x80000000u) { uint borrow; pt.y = fp_sub_raw(fp_modulus(), pt.y, borrow); }

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

// Phase 2: Direct weighted bucket sum per segment.
// Each thread independently computes the exact weighted contribution of its segment:
//   segment_result = sum + (lo - 1) × running
// where sum = Σ (i - lo + 1) × bucket[i] via running-sum trick,
// and running = Σ bucket[i] over the segment.
// Host just adds all segment_results per window — no carry propagation needed.
//
// Batched: tid = window_idx * n_segments + segment_idx
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

    // Compute segment bounds (high to low bucket indices)
    int hi_s = int(n_buckets) - int(seg_idx * seg_size);
    int lo_raw_s = int((seg_idx + 1) * seg_size);
    int lo_s = (lo_raw_s >= int(n_buckets)) ? 1 : (int(n_buckets) - lo_raw_s);
    if (lo_s < 1) lo_s = 1;
    if (hi_s <= lo_s) {
        segment_results[tid] = point_identity();
        return;
    }

    PointProjective running;
    PointProjective sum;
    bool running_set = false;
    bool sum_set = false;

    uint hi = uint(hi_s);
    uint lo = uint(lo_s);
    for (uint i = hi - 1; i >= lo; i--) {
        PointProjective bucket = buckets[bucket_base + i];
        if (!point_is_identity(bucket)) {
            if (!running_set) {
                running = bucket;
                running_set = true;
            } else {
                running = point_add(running, bucket);
            }
        }
        if (running_set) {
            if (!sum_set) {
                sum = running;
                sum_set = true;
            } else {
                sum = point_add(sum, running);
            }
        }
        if (i == lo) break;
    }

    uint weight = lo - 1;
    if (weight > 0 && running_set) {
        bool weighted_set = false;
        PointProjective weighted;
        PointProjective base = running;
        uint k = weight;
        while (k > 0) {
            if (k & 1u) {
                if (!weighted_set) {
                    weighted = base;
                    weighted_set = true;
                } else {
                    weighted = point_add(weighted, base);
                }
            }
            base = point_double(base);
            k >>= 1;
        }
        if (weighted_set) {
            if (!sum_set) {
                sum = weighted;
                sum_set = true;
            } else {
                sum = point_add(sum, weighted);
            }
        }
    }

    if (!sum_set) {
        segment_results[tid] = point_identity();
    } else {
        segment_results[tid] = sum;
    }
}

// Phase 3: Parallel reduction of segment results per window.
// Each threadgroup handles one window, reducing n_segments results to a single sum.
// Uses threadgroup shared memory for tree reduction.
// Phase 3: Parallel reduction of segment results per window.
// Each threadgroup handles one window, reducing n_segments results to a single sum.
// Uses threadgroup shared memory for tree reduction.
kernel void msm_combine_segments(
    device const PointProjective* segment_results [[buffer(0)]],
    device PointProjective* window_results        [[buffer(1)]],
    constant uint& n_segments                     [[buffer(2)]],
    uint tgid                                     [[threadgroup_position_in_grid]],
    uint lid                                      [[thread_index_in_threadgroup]],
    uint tg_size                                  [[threads_per_threadgroup]]
) {
    // Each threadgroup = one window. Supports arbitrary n_segments.
    // Each thread pre-reduces a contiguous chunk, then tree reduce in shared memory.
    threadgroup PointProjective shared_buf[256]; // max tg_size entries

    uint base = tgid * n_segments;
    uint chunk = (n_segments + tg_size - 1) / tg_size;
    uint start = lid * chunk;
    uint end = min(start + chunk, n_segments);

    // Pre-reduce chunk
    PointProjective v = point_identity();
    for (uint i = start; i < end; i++) {
        PointProjective s = segment_results[base + i];
        if (!point_is_identity(s)) {
            if (point_is_identity(v)) { v = s; }
            else { v = point_add(v, s); }
        }
    }
    shared_buf[lid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
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

// Phase 4: Horner combination of window results.
// Computes: result = w[n-1] * 2^((n-1)*wb) + ... + w[1] * 2^wb + w[0]
// Using Horner's method: result = ((w[n-1] * 2^wb + w[n-2]) * 2^wb + ...) + w[0]
// Single thread, chained after combine_segments.
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
        // Multiply by 2^window_bits
        for (uint b = 0; b < window_bits; b++) {
            result = point_double(result);
        }
        // Add next window result
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

// --- GLV Scalar Decomposition Kernel ---
// BN254 scalar field order r
constant ulong FR_ORDER[4] = {
    0x43e1f593f0000001uL, 0x2833e84879b97091uL,
    0xb85045b68181585duL, 0x30644e72e131a029uL
};

// GLV lattice constants
constant ulong GLV_G1_0 = 0x7a7bd9d4391eb18duL;
constant ulong GLV_G1_1 = 0x4ccef014a773d2cfuL;
constant ulong GLV_G1_2 = 0x2uL;
constant ulong GLV_G2_0 = 0xd91d232ec7e0b3d7uL;
constant ulong GLV_G2_1 = 0x2uL;
constant ulong GLV_A1 = 0x89d3256894d213e3uL;
constant ulong GLV_MINUS_B1_0 = 0x8211bbeb7d4f1128uL;
constant ulong GLV_MINUS_B1_1 = 0x6f4d8248eeb859fcuL;
constant ulong GLV_A2_0 = 0x0be4e1541221250buL;
constant ulong GLV_A2_1 = 0x6f4d8248eeb859fduL;
constant ulong GLV_B2 = 0x89d3256894d213e3uL;

constant ulong HALF_R[4] = {
    (0x43e1f593f0000001uL >> 1) | (0x2833e84879b97091uL << 63),
    (0x2833e84879b97091uL >> 1) | (0xb85045b68181585duL << 63),
    (0xb85045b68181585duL >> 1) | (0x30644e72e131a029uL << 63),
    0x30644e72e131a029uL >> 1
};

// 256-bit add with carry (using ulong limbs)
void u256_add(thread ulong* r, thread const ulong* a, constant ulong* b, thread bool &carry) {
    ulong c = 0;
    for (int i = 0; i < 4; i++) {
        ulong s = a[i] + b[i];
        ulong t = s + c;
        r[i] = t;
        c = (s < a[i] || t < s) ? 1uL : 0uL;
    }
    carry = c != 0;
}

void u256_sub(thread ulong* r, thread const ulong* a, thread const ulong* b, thread bool &borrow) {
    ulong br = 0;
    for (int i = 0; i < 4; i++) {
        ulong diff = a[i] - b[i];
        ulong diff2 = diff - br;
        br = ((a[i] < b[i]) || (br && diff == 0)) ? 1uL : 0uL;
        r[i] = diff2;
    }
    borrow = br != 0;
}

void u256_sub_const(thread ulong* r, thread const ulong* a, constant ulong* b, thread bool &borrow) {
    ulong br = 0;
    for (int i = 0; i < 4; i++) {
        ulong diff = a[i] - b[i];
        ulong diff2 = diff - br;
        br = ((a[i] < b[i]) || (br && diff == 0)) ? 1uL : 0uL;
        r[i] = diff2;
    }
    borrow = br != 0;
}

void u256_sub_from_const(thread ulong* r, constant ulong* a, thread const ulong* b, thread bool &borrow) {
    ulong br = 0;
    for (int i = 0; i < 4; i++) {
        ulong diff = a[i] - b[i];
        ulong diff2 = diff - br;
        br = ((a[i] < b[i]) || (br && diff == 0)) ? 1uL : 0uL;
        r[i] = diff2;
    }
    borrow = br != 0;
}

bool u256_gte_const(thread const ulong* a, constant ulong* b) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

// 256×192 multiply, return bits [256..383] as (lo, hi)
void mul256x192(thread const ulong* k, ulong g0, ulong g1, ulong g2,
                thread ulong &out_lo, thread ulong &out_hi) {
    ulong prod[7] = {0,0,0,0,0,0,0};
    ulong gv[3] = {g0, g1, g2};
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 3; j++) {
            ulong hi = mulhi(k[i], gv[j]);
            ulong lo = k[i] * gv[j];
            ulong s1 = prod[i+j] + lo;
            ulong c1 = (s1 < prod[i+j]) ? 1uL : 0uL;
            ulong s2 = s1 + carry;
            ulong c2 = (s2 < s1) ? 1uL : 0uL;
            prod[i+j] = s2;
            carry = hi + c1 + c2;
        }
        prod[i+3] += carry;
    }
    out_lo = prod[4];
    out_hi = prod[5];
}

// 256×128 multiply, return bits [256..319]
ulong mul256x128(thread const ulong* k, ulong g0, ulong g1) {
    ulong prod[6] = {0,0,0,0,0,0};
    ulong gv[2] = {g0, g1};
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 2; j++) {
            ulong hi = mulhi(k[i], gv[j]);
            ulong lo = k[i] * gv[j];
            ulong s1 = prod[i+j] + lo;
            ulong c1 = (s1 < prod[i+j]) ? 1uL : 0uL;
            ulong s2 = s1 + carry;
            ulong c2 = (s2 < s1) ? 1uL : 0uL;
            prod[i+j] = s2;
            carry = hi + c1 + c2;
        }
        prod[i+2] += carry;
    }
    return prod[4];
}

// 128×128 multiply → 256-bit result
void mul128x128_gpu(ulong a0, ulong a1, ulong b0, ulong b1, thread ulong* r) {
    ulong h00 = mulhi(a0, b0), l00 = a0 * b0;
    ulong h01 = mulhi(a0, b1), l01 = a0 * b1;
    ulong h10 = mulhi(a1, b0), l10 = a1 * b0;
    ulong h11 = mulhi(a1, b1), l11 = a1 * b1;

    r[0] = l00;
    ulong s1 = l01 + h00;
    ulong c1a = (s1 < l01) ? 1uL : 0uL;
    ulong s1b = s1 + l10;
    ulong c1b = (s1b < s1) ? 1uL : 0uL;
    r[1] = s1b;
    ulong s2 = h01 + h10;
    ulong c2a = (s2 < h01) ? 1uL : 0uL;
    s2 += l11;
    ulong c2b = (s2 < l11) ? 1uL : 0uL;
    s2 += c1a + c1b;
    r[2] = s2;
    r[3] = h11 + c2a + c2b + ((s2 < c1a + c1b) ? 1uL : 0uL);
}

// 64×128 multiply → 192-bit
void mul64x128_gpu(ulong a, ulong b0, ulong b1, thread ulong &r0, thread ulong &r1, thread ulong &r2) {
    ulong h0 = mulhi(a, b0), l0 = a * b0;
    ulong h1 = mulhi(a, b1), l1 = a * b1;
    r0 = l0;
    ulong s1 = l1 + h0;
    r1 = s1;
    r2 = h1 + ((s1 < l1) ? 1uL : 0uL);
}

// 128×64 multiply → 192-bit
void mul128x64_gpu(ulong a0, ulong a1, ulong b, thread ulong &r0, thread ulong &r1, thread ulong &r2) {
    ulong h0 = mulhi(a0, b), l0 = a0 * b;
    ulong h1 = mulhi(a1, b), l1 = a1 * b;
    r0 = l0;
    ulong s1 = l1 + h0;
    r1 = s1;
    r2 = h1 + ((s1 < l1) ? 1uL : 0uL);
}

// GPU GLV scalar decomposition kernel
// Reads 256-bit scalars, writes 128-bit k1/k2 and neg flags
kernel void glv_decompose(
    const device uint* scalars_in [[buffer(0)]],    // n × 8 uint32 (256-bit scalars)
    device uint* k1_out [[buffer(1)]],              // n × 8 uint32 (output k1)
    device uint* k2_out [[buffer(2)]],              // n × 8 uint32 (output k2)
    device uchar* neg1_out [[buffer(3)]],
    device uchar* neg2_out [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    // Read scalar as 4×ulong
    const device uint* sp = scalars_in + gid * 8;
    ulong kr[4] = {
        ulong(sp[0]) | (ulong(sp[1]) << 32),
        ulong(sp[2]) | (ulong(sp[3]) << 32),
        ulong(sp[4]) | (ulong(sp[5]) << 32),
        ulong(sp[6]) | (ulong(sp[7]) << 32)
    };

    // Reduce mod r
    bool borrow;
    while (u256_gte_const(kr, FR_ORDER)) {
        ulong tmp[4];
        u256_sub_const(tmp, kr, FR_ORDER, borrow);
        for (int i = 0; i < 4; i++) kr[i] = tmp[i];
    }

    // c1 = (k * g1) >> 256
    ulong c1_lo, c1_hi;
    mul256x192(kr, GLV_G1_0, GLV_G1_1, GLV_G1_2, c1_lo, c1_hi);

    // c2 = (k * g2) >> 256
    ulong c2 = mul256x128(kr, GLV_G2_0, GLV_G2_1);

    // k1 = k - c2*a1 - c1*a2
    ulong c2a1_hi = mulhi(c2, GLV_A1);
    ulong c2a1_lo = c2 * GLV_A1;

    ulong c1a2[4];
    mul128x128_gpu(c1_lo, c1_hi, GLV_A2_0, GLV_A2_1, c1a2);

    ulong k1[4];
    ulong sub1[4] = {c2a1_lo, c2a1_hi, 0, 0};
    u256_sub(k1, kr, sub1, borrow);
    if (borrow) u256_add(k1, k1, FR_ORDER, borrow);
    ulong k1b[4];
    u256_sub(k1b, k1, c1a2, borrow);
    if (borrow) { ulong tmp[4]; u256_add(tmp, k1b, FR_ORDER, borrow); for (int i=0;i<4;i++) k1b[i]=tmp[i]; }
    for (int i = 0; i < 4; i++) k1[i] = k1b[i];

    // k2 = c2*|b1| - c1*b2
    ulong c2mb1_0, c2mb1_1, c2mb1_2;
    mul64x128_gpu(c2, GLV_MINUS_B1_0, GLV_MINUS_B1_1, c2mb1_0, c2mb1_1, c2mb1_2);

    ulong c1b2_0, c1b2_1, c1b2_2;
    mul128x64_gpu(c1_lo, c1_hi, GLV_B2, c1b2_0, c1b2_1, c1b2_2);

    ulong k2_a[4] = {c2mb1_0, c2mb1_1, c2mb1_2, 0};
    ulong k2_b[4] = {c1b2_0, c1b2_1, c1b2_2, 0};
    ulong k2[4];
    u256_sub(k2, k2_a, k2_b, borrow);
    bool k2_neg = false;
    if (borrow) {
        k2_neg = true;
        ulong n0 = ~k2[0] + 1;
        ulong c0 = (k2[0] == 0) ? 1uL : 0uL;
        ulong n1 = ~k2[1] + c0;
        ulong cc1 = (k2[1] == 0 && c0 == 1) ? 1uL : 0uL;
        ulong n2 = ~k2[2] + cc1;
        ulong cc2 = (k2[2] == 0 && cc1 == 1) ? 1uL : 0uL;
        ulong n3 = ~k2[3] + cc2;
        k2[0] = n0; k2[1] = n1; k2[2] = n2; k2[3] = n3;
    }

    // If k1 > r/2, negate
    bool neg1 = false;
    if (u256_gte_const(k1, HALF_R)) {
        u256_sub_from_const(k1, FR_ORDER, k1, borrow);
        neg1 = true;
    }

    // Write outputs
    device uint* k1p = k1_out + gid * 8;
    device uint* k2p = k2_out + gid * 8;
    for (int i = 0; i < 4; i++) {
        k1p[i*2] = uint(k1[i] & 0xFFFFFFFF);
        k1p[i*2+1] = uint(k1[i] >> 32);
        k2p[i*2] = uint(k2[i] & 0xFFFFFFFF);
        k2p[i*2+1] = uint(k2[i] >> 32);
    }
    neg1_out[gid] = neg1 ? 1 : 0;
    neg2_out[gid] = k2_neg ? 1 : 0;
}

// --- GLV Endomorphism Kernel ---
// Applies φ(P) = (β·x, y) and optional negation for GLV MSM.
// Reads original points[0..n-1], writes endomorphism points[n..2n-1]
// and optionally negates original points based on neg1 flags.
kernel void glv_endomorphism(
    device PointAffine* points [[buffer(0)]],
    const device uchar* neg1_flags [[buffer(1)]],
    const device uchar* neg2_flags [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    PointAffine p = points[gid];

    // Apply neg1 to original point
    if (neg1_flags[gid]) {
        // -P = (x, p - y)
        Fp py = fp_modulus();
        uint borrow;
        p.y = fp_sub_raw(py, p.y, borrow);
        points[gid] = p;
        // Read back the original p for endomorphism (before negation)
        p.y = fp_sub_raw(fp_modulus(), p.y, borrow); // undo for φ
    }

    // β in Montgomery form: cube root of unity in Fp
    Fp beta;
    beta.v[0] = 0xd782e155u; beta.v[1] = 0x71930c11u;
    beta.v[2] = 0xffbe3323u; beta.v[3] = 0xa6bb947cu;
    beta.v[4] = 0xd4741444u; beta.v[5] = 0xaa303344u;
    beta.v[6] = 0x26594943u; beta.v[7] = 0x2c3b3f0du;

    PointAffine endo;
    endo.x = fp_mul(beta, p.x);

    if (neg2_flags[gid]) {
        uint borrow;
        endo.y = fp_sub_raw(fp_modulus(), p.y, borrow);
    } else {
        endo.y = p.y;
    }

    points[n + gid] = endo;
}

// GPU signed-digit extraction: reads decomposed scalars, writes per-window bucket indices
// with sign bit encoding. Replaces CPU precomp phase.
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
