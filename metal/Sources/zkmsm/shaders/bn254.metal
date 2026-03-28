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

// Load modulus P into Fp
Fp fp_modulus() {
    Fp r;
    for (int i = 0; i < 8; i++) r.v[i] = P[i];
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

// Modular squaring (delegates to mul — separate function would increase register pressure)
Fp fp_sqr(Fp a) {
    return fp_mul(a, a);
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

// Point doubling in Jacobian coordinates
// Cost: 4M + 6S + 1*a + 7add (a=0 for BN254)
PointProjective point_double(PointProjective p) {
    if (point_is_identity(p)) return p;

    Fp a = fp_sqr(p.x);       // a = X^2
    Fp b = fp_sqr(p.y);       // b = Y^2
    Fp c = fp_sqr(b);         // c = Y^4

    Fp d = fp_sub(fp_sqr(fp_add(p.x, b)), fp_add(a, c));
    d = fp_double(d);         // d = 2*((X+Y^2)^2 - X^2 - Y^4)

    Fp e = fp_add(fp_double(a), a); // e = 3*X^2 (a_coeff=0 for BN254)

    Fp f = fp_sqr(e);         // f = (3*X^2)^2

    PointProjective r;
    r.x = fp_sub(f, fp_double(d));      // X3 = f - 2*d
    r.y = fp_sub(fp_mul(e, fp_sub(d, r.x)), fp_double(fp_double(fp_double(c)))); // Y3 = e*(d-X3) - 8*c
    r.z = fp_sub(fp_sqr(fp_add(p.y, p.z)), fp_add(b, fp_sqr(p.z))); // Z3 = (Y+Z)^2 - Y^2 - Z^2
    return r;
}

// Branchless mixed addition — no identity or same-x checks.
// Safe when caller guarantees p is non-identity and P ≠ ±Q (true for random points).
// Mixed addition: projective + affine, handles all edge cases (identity, P=Q, P=-Q)
// Defers s2/rr computation past h=0 check for better register pressure on common path.
PointProjective point_add_mixed(PointProjective p, PointAffine q) {
    if (point_is_identity(p)) return point_from_affine(q);

    Fp z1z1 = fp_sqr(p.z);
    Fp u2 = fp_mul(q.x, z1z1);
    Fp h = fp_sub(u2, p.x);

    if (fp_is_zero(h)) {
        // Rare: same x-coordinate, check if P=Q (double) or P=-Q (identity)
        Fp s2 = fp_mul(q.y, fp_mul(p.z, z1z1));
        Fp rr = fp_double(fp_sub(s2, p.y));
        if (fp_is_zero(rr)) return point_double(p);
        return point_identity();
    }

    // Common path: normal addition
    Fp s2 = fp_mul(q.y, fp_mul(p.z, z1z1));
    PointProjective result;
    result.z = fp_double(fp_mul(p.z, h));
    Fp hh = fp_sqr(h);
    Fp i = fp_double(fp_double(hh));
    Fp v = fp_mul(p.x, i);
    Fp j = fp_mul(h, i);
    Fp rr = fp_double(fp_sub(s2, p.y));
    result.x = fp_sub(fp_sub(fp_sqr(rr), j), fp_double(v));
    result.y = fp_sub(fp_mul(rr, fp_sub(v, result.x)),
                      fp_double(fp_mul(p.y, j)));
    return result;
}

// Full point addition: projective + projective
// Register-optimized: Z3 = 2*Z1*Z2*H frees z1z1, z2z2, p.z, q.z early.
PointProjective point_add(PointProjective p, PointProjective q) {
    if (point_is_identity(p)) return q;
    if (point_is_identity(q)) return p;

    Fp z1z1 = fp_sqr(p.z);
    Fp z2z2 = fp_sqr(q.z);
    Fp u1 = fp_mul(p.x, z2z2);
    Fp u2 = fp_mul(q.x, z1z1);
    Fp s1 = fp_mul(p.y, fp_mul(q.z, z2z2));      // z2z2 last use
    Fp s2 = fp_mul(q.y, fp_mul(p.z, z1z1));       // z1z1 last use

    Fp h = fp_sub(u2, u1);                         // u2 dead
    Fp rr = fp_double(fp_sub(s2, s1));              // s2 dead

    if (fp_is_zero(h)) {
        if (fp_is_zero(rr)) {
            return point_double(p);
        }
        return point_identity();
    }

    // Z3 = ((Z1+Z2)^2 - Z1^2 - Z2^2) * H = 2*Z1*Z2*H
    // Compute early to free p.z, q.z
    PointProjective result;
    result.z = fp_mul(fp_double(fp_mul(p.z, q.z)), h); // p.z, q.z dead

    Fp i = fp_sqr(fp_double(h));                    // I = (2H)^2
    Fp v = fp_mul(u1, i);                           // V = U1*I (u1 dead)
    Fp j = fp_mul(h, i);                            // J = H*I (h dead, i dead)

    result.x = fp_sub(fp_sub(fp_sqr(rr), j), fp_double(v));
    result.y = fp_sub(fp_mul(rr, fp_sub(v, result.x)),
                      fp_double(fp_mul(s1, j)));
    return result;
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
    uint tid                                   [[thread_position_in_grid]]
) {
    uint nb = 1u << params.window_bits;
    uint total = nb * n_windows;
    if (tid >= total) return;
    uint window_idx = tid >> params.window_bits;
    uint bucket_idx = tid & (nb - 1u);
    if (bucket_idx == 0) {
        buckets[tid] = point_identity();
        return;
    }

    uint count = bucket_counts[tid];
    if (count == 0) {
        buckets[tid] = point_identity();
        return;
    }

    uint base = window_idx * params.n_points;
    uint offset = bucket_offsets[tid];
    PointProjective acc = point_from_affine(points[sorted_indices[base + offset]]);
    for (uint i = 1; i < count; i++) {
        acc = point_add_mixed(acc, points[sorted_indices[base + offset + i]]);
    }
    buckets[tid] = acc;
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

    uint n_buckets = 1u << params.window_bits;
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

    // Adjust sum to get exact weighted contribution:
    // We computed sum = Σ (i - lo + 1) × bucket[i]
    // We want: Σ i × bucket[i] = sum + (lo - 1) × running
    uint weight = lo - 1;
    if (weight > 0 && !point_is_identity(running)) {
        // Scalar multiply: weight × running using double-and-add
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
    // Each threadgroup = one window. Supports up to 2*tg_size segments.
    // Each thread loads 2 segment results, adds them, then tree reduce on tg_size entries.
    threadgroup PointProjective shared_buf[256]; // max tg_size entries

    uint base = tgid * n_segments;
    uint idx0 = lid * 2;
    uint idx1 = lid * 2 + 1;

    // Load pair and pre-reduce
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
