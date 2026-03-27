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
    for (int i = 0; i < 10; i++) t[i] = 0;

    for (int i = 0; i < 8; i++) {
        // Step 1: t += a[i] * b
        ulong carry = 0;
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

// Modular squaring (uses mul for now, could be optimized)
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

// Mixed addition: projective + affine (saves multiplications when Z2=1)
// Cost: 7M + 4S + 9add
PointProjective point_add_mixed(PointProjective p, PointAffine q) {
    if (point_is_identity(p)) return point_from_affine(q);

    Fp z1z1 = fp_sqr(p.z);           // Z1^2
    Fp u2 = fp_mul(q.x, z1z1);       // U2 = X2*Z1^2
    Fp s2 = fp_mul(q.y, fp_mul(p.z, z1z1)); // S2 = Y2*Z1^3

    Fp h = fp_sub(u2, p.x);          // H = U2 - X1

    // Handle special cases: same x-coordinate
    if (fp_is_zero(h)) {
        Fp rr_check = fp_double(fp_sub(s2, p.y));
        if (fp_is_zero(rr_check)) {
            return point_double(p); // same point → double
        }
        return point_identity(); // inverse points → identity
    }

    Fp hh = fp_sqr(h);               // HH = H^2

    Fp i = fp_double(fp_double(hh));  // I = 4*H^2
    Fp j = fp_mul(h, i);             // J = H*I
    Fp rr = fp_double(fp_sub(s2, p.y)); // r = 2*(S2-Y1)

    Fp v = fp_mul(p.x, i);           // V = X1*I

    PointProjective result;
    result.x = fp_sub(fp_sub(fp_sqr(rr), j), fp_double(v)); // X3 = r^2 - J - 2*V
    result.y = fp_sub(fp_mul(rr, fp_sub(v, result.x)),
                      fp_double(fp_mul(p.y, j)));            // Y3 = r*(V-X3) - 2*Y1*J
    result.z = fp_sub(fp_sqr(fp_add(p.z, h)),
                      fp_add(z1z1, hh));                     // Z3 = (Z1+H)^2 - Z1^2 - HH
    return result;
}

// Full point addition: projective + projective
PointProjective point_add(PointProjective p, PointProjective q) {
    if (point_is_identity(p)) return q;
    if (point_is_identity(q)) return p;

    Fp z1z1 = fp_sqr(p.z);
    Fp z2z2 = fp_sqr(q.z);
    Fp u1 = fp_mul(p.x, z2z2);
    Fp u2 = fp_mul(q.x, z1z1);
    Fp s1 = fp_mul(p.y, fp_mul(q.z, z2z2));
    Fp s2 = fp_mul(q.y, fp_mul(p.z, z1z1));

    Fp h = fp_sub(u2, u1);
    Fp i = fp_sqr(fp_double(h));
    Fp j = fp_mul(h, i);
    Fp rr = fp_double(fp_sub(s2, s1));

    // Check if points are equal (h == 0 means same x-coordinate)
    if (fp_is_zero(h)) {
        if (fp_is_zero(rr)) {
            return point_double(p); // same point
        }
        return point_identity(); // inverse points
    }

    Fp v = fp_mul(u1, i);

    PointProjective result;
    result.x = fp_sub(fp_sub(fp_sqr(rr), j), fp_double(v));
    result.y = fp_sub(fp_mul(rr, fp_sub(v, result.x)),
                      fp_double(fp_mul(s1, j)));
    result.z = fp_mul(fp_sub(fp_sqr(fp_add(p.z, q.z)),
                             fp_add(z1z1, z2z2)), h);
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
    uint window_index;   // which window this dispatch handles
};

// Phase 1: Reduce pre-sorted points per bucket. No atomics needed.
// Supports batched execution across multiple windows:
//   tid = window_idx * n_buckets + bucket_idx
// sorted_points, bucket_offsets, bucket_counts are laid out per-window.
kernel void msm_reduce_sorted_buckets(
    device const PointAffine* sorted_points    [[buffer(0)]],
    device PointProjective* buckets            [[buffer(1)]],
    device const uint* bucket_offsets          [[buffer(2)]],
    device const uint* bucket_counts           [[buffer(3)]],
    constant MsmParams& params                 [[buffer(4)]],
    constant uint& n_windows                   [[buffer(5)]],
    uint tid                                   [[thread_position_in_grid]]
) {
    uint n_buckets = 1u << params.window_bits;
    uint total = n_buckets * n_windows;
    if (tid >= total) return;

    uint bucket_idx = tid % n_buckets;
    if (bucket_idx == 0) {
        buckets[tid] = point_identity();
        return;
    }

    uint count = bucket_counts[tid];
    if (count == 0) {
        buckets[tid] = point_identity();
        return;
    }

    uint offset = bucket_offsets[tid];
    PointProjective acc = point_from_affine(sorted_points[offset]);
    for (uint i = 1; i < count; i++) {
        acc = point_add_mixed(acc, sorted_points[offset + i]);
    }
    buckets[tid] = acc;
}

// Phase 2: Parallel segmented bucket sum.
// Splits the bucket range into segments, each processed by one thread.
// Each thread computes:
//   partial_sum = Σ running-sum over its segment (high to low)
//   carry = running total at the end of its segment (to pass to the next segment)
// Host combines segments: adjust each segment's sum by its carry weight.
//
// For n_segments threads processing buckets [seg_start..seg_end):
//   Thread k handles buckets [(k+1)*seg_size .. k*seg_size] (high to low within segment)
//
// Output: partial_sums[tid] and carries[tid]
// Batched bucket sum: tid = window_idx * n_segments + segment_idx
// Buckets for window w start at buckets[w * n_buckets].
// Outputs: partial_sums[tid], carries[tid] for each (window, segment).
kernel void msm_bucket_sum_parallel(
    device const PointProjective* buckets   [[buffer(0)]],
    device PointProjective* partial_sums    [[buffer(1)]],
    device PointProjective* carries         [[buffer(2)]],
    constant MsmParams& params              [[buffer(3)]],
    constant uint& n_segments               [[buffer(4)]],
    constant uint& n_windows                [[buffer(5)]],
    uint tid                                [[thread_position_in_grid]]
) {
    uint total = n_segments * n_windows;
    if (tid >= total) return;

    uint window_idx = tid / n_segments;
    uint seg_idx = tid % n_segments;

    uint n_buckets = 1u << params.window_bits;
    uint seg_size = (n_buckets + n_segments - 1) / n_segments;
    uint bucket_base = window_idx * n_buckets;

    uint hi = n_buckets - seg_idx * seg_size;
    uint lo_raw = (seg_idx + 1) * seg_size;
    uint lo = (lo_raw >= n_buckets) ? 1 : (n_buckets - lo_raw);
    if (lo < 1) lo = 1;
    if (hi <= lo) {
        partial_sums[tid] = point_identity();
        carries[tid] = point_identity();
        return;
    }

    PointProjective running = point_identity();
    PointProjective sum = point_identity();

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

    partial_sums[tid] = sum;
    carries[tid] = running;
}
