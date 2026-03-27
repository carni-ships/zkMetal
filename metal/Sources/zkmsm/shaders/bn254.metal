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
    0x0a6e141a, 0x47ab1eff, 0xcab8351f, 0x06d89f71
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
    // R mod p = 2^256 mod p
    Fp r;
    r.v[0] = 0xd35d438d; r.v[1] = 0x0a78eb28; r.v[2] = 0x7fd748d7;
    r.v[3] = 0xc8a21c71; r.v[4] = 0x8898fb76; r.v[5] = 0x0fc77c57;
    r.v[6] = 0x2e7e4076; r.v[7] = 0x12e2908d;
    return r;
}

// --- Montgomery Multiplication ---
// Computes (a * b * R^-1) mod p using CIOS method

Fp fp_mul(Fp a, Fp b) {
    // CIOS (Coarsely Integrated Operand Scanning) Montgomery multiplication
    // Working in 32-bit limbs for Metal GPU compatibility
    uint t[17]; // 9 limbs for product + overflow
    for (int i = 0; i < 17; i++) t[i] = 0;

    for (int i = 0; i < 8; i++) {
        // Multiply-accumulate: t += a[i] * b
        ulong carry = 0;
        for (int j = 0; j < 8; j++) {
            carry += ulong(t[j]) + ulong(a.v[i]) * ulong(b.v[j]);
            t[j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        t[8] += uint(carry);

        // Montgomery reduction
        uint m = t[0] * INV;
        carry = 0;
        for (int j = 0; j < 8; j++) {
            carry += ulong(t[j]) + ulong(m) * ulong(P[j]);
            t[j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        t[8] += uint(carry);

        // Shift right by 32 bits
        for (int j = 0; j < 8; j++) t[j] = t[j + 1];
        t[8] = 0;
    }

    Fp r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i];

    // Final reduction
    if (fp_gte(r, fp_modulus())) {
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

// --- MSM Kernel: Pippenger's Bucket Method ---
//
// Each threadgroup handles a window of scalar bits.
// Phase 1: Accumulate points into buckets based on scalar window value.
// Phase 2: Reduce buckets into a single result per window.
// Host combines window results with double-and-add.

// Shared buffer for bucket accumulation
struct MsmParams {
    uint n_points;       // number of points
    uint window_bits;    // bits per window (e.g., 16)
    uint window_index;   // which window this dispatch handles
};

kernel void msm_accumulate(
    device const PointAffine* points   [[buffer(0)]],
    device const uint* scalars         [[buffer(1)]],  // 8 x uint32 per scalar (256-bit LE)
    device PointProjective* buckets    [[buffer(2)]],   // (1 << window_bits) buckets per window
    constant MsmParams& params         [[buffer(3)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid >= params.n_points) return;

    uint window_bits = params.window_bits;
    uint window_index = params.window_index;

    // Extract window_bits bits starting at bit position (window_index * window_bits) from scalar
    uint bit_offset = window_index * window_bits;
    uint limb_idx = bit_offset / 32;
    uint bit_pos = bit_offset % 32;

    // Read the scalar limbs for this point
    uint scalar_offset = tid * 8;
    uint bucket_idx = 0;

    if (limb_idx < 8) {
        bucket_idx = (scalars[scalar_offset + limb_idx] >> bit_pos);
        if (bit_pos + window_bits > 32 && limb_idx + 1 < 8) {
            bucket_idx |= (scalars[scalar_offset + limb_idx + 1] << (32 - bit_pos));
        }
        bucket_idx &= ((1u << window_bits) - 1u);
    }

    if (bucket_idx == 0) return; // scalar window is zero, skip

    // Atomic-free accumulation: each thread writes to its own slot.
    // Host-side reduction combines overlapping bucket entries.
    // For a production implementation, use atomic bucket accumulation or
    // a scatter-gather approach. This simplified version uses per-thread output.
    PointProjective pt = point_from_affine(points[tid]);

    // Write: buckets[tid] = (bucket_idx, point)
    // Encode bucket_idx in the output for host-side reduction
    buckets[tid] = pt;
    // Store bucket index in a sideband (overload z.v[7] which is unused for identity)
    buckets[tid].z.v[7] = bucket_idx;
}

// Reduce phase: sum all points in the same bucket
kernel void msm_reduce_buckets(
    device const PointProjective* thread_buckets [[buffer(0)]],
    device PointProjective* reduced_buckets      [[buffer(1)]],
    constant MsmParams& params                   [[buffer(2)]],
    uint bucket_id                               [[thread_position_in_grid]]
) {
    uint n_buckets = 1u << params.window_bits;
    if (bucket_id >= n_buckets) return;

    PointProjective acc = point_identity();

    for (uint i = 0; i < params.n_points; i++) {
        PointProjective pt = thread_buckets[i];
        uint pt_bucket = pt.z.v[7];
        if (pt_bucket == bucket_id && !fp_is_zero(pt.z)) {
            // Restore z.v[7] before adding
            pt.z.v[7] = 0;
            if (point_is_identity(acc)) {
                acc = pt;
            } else {
                acc = point_add(acc, pt);
            }
        }
    }

    reduced_buckets[bucket_id] = acc;
}

// Final bucket reduction: compute window sum using running-sum technique
// result = sum(i * bucket[i]) for i in 1..n_buckets
// = bucket[n-1] + (bucket[n-1] + bucket[n-2]) + ...
kernel void msm_bucket_sum(
    device const PointProjective* reduced_buckets [[buffer(0)]],
    device PointProjective* window_result          [[buffer(1)]],
    constant MsmParams& params                     [[buffer(2)]],
    uint tid                                       [[thread_position_in_grid]]
) {
    if (tid != 0) return; // single-threaded reduction for correctness

    uint n_buckets = 1u << params.window_bits;
    PointProjective running = point_identity();
    PointProjective sum = point_identity();

    // Iterate from highest bucket to lowest
    for (uint i = n_buckets - 1; i >= 1; i--) {
        PointProjective bucket = reduced_buckets[i];
        if (!point_is_identity(bucket)) {
            running = point_add(running, bucket);
        }
        sum = point_add(sum, running);
    }

    window_result[0] = sum;
}
