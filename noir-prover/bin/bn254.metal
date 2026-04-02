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
    #pragma clang loop unroll(full)
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
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; i++) {
        c += long(a.v[i]) - long(b.v[i]);
        r.v[i] = uint(c & 0xFFFFFFFF);
        c >>= 32;
    }
    borrow = (c < 0) ? 1u : 0u;
    return r;
}

// Compare a >= b (branchless: a >= b iff a - b doesn't borrow)
bool fp_gte(Fp a, Fp b) {
    uint borrow;
    fp_sub_raw(a, b, borrow);
    return borrow == 0;
}

// Load modulus P into Fp
Fp fp_modulus() {
    Fp r;
    for (int i = 0; i < 8; i++) r.v[i] = P[i];
    return r;
}

// Modular addition: (a + b) mod p
// Requires inputs in [0, p). Output is in [0, p).
Fp fp_add(Fp a, Fp b) {
    uint carry;
    Fp r = fp_add_raw(a, b, carry);
    // a + b < 2p, so at most one subtraction of p needed
    uint borrow;
    Fp reduced = fp_sub_raw(r, fp_modulus(), borrow);
    return (carry || !borrow) ? reduced : r;
}

// Modular subtraction: (a - b) mod p
// Requires inputs in [0, p). Output is in [0, p).
Fp fp_sub(Fp a, Fp b) {
    uint borrow;
    Fp r = fp_sub_raw(a, b, borrow);
    if (borrow) {
        uint carry;
        r = fp_add_raw(r, fp_modulus(), carry);
    }
    return r;
}

// Field negation: -a mod p
// Handles input in [0, 2p) via fp_sub_raw + conditional reduction.
Fp fp_neg(Fp a) {
    // First reduce a to [0, p)
    uint borrow_r;
    Fp ra = fp_sub_raw(a, fp_modulus(), borrow_r);
    if (!borrow_r) a = ra; // a was >= p, use reduced
    // Now a ∈ [0, p). Compute p - a.
    uint borrow;
    Fp r = fp_sub_raw(fp_modulus(), a, borrow);
    // If a == 0, result is p → need to reduce to 0
    uint borrow2;
    Fp reduced = fp_sub_raw(r, fp_modulus(), borrow2);
    return borrow2 ? r : reduced;
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
// Computes (a * b * R^-1) mod p using CIOS method (32-bit limbs).
// 8×32-bit single-chain CIOS is faster than 4×64-bit on Apple GPU because
// the GPU is natively 32-bit — ulong operations use pairs of registers.
// Tested: 4×64 with mulhi is 2.3× slower on M3 Pro GPU.

Fp fp_mul(Fp a, Fp b) {
    uint t[10];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 10; i++) t[i] = 0;

    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; i++) {
        ulong carry = 0;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 8; j++) {
            carry += ulong(t[j]) + ulong(a.v[i]) * ulong(b.v[j]);
            t[j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        ulong ext = ulong(t[8]) + carry;
        t[8] = uint(ext & 0xFFFFFFFF);
        t[9] = uint(ext >> 32);

        uint m = t[0] * INV;
        carry = ulong(t[0]) + ulong(m) * ulong(P[0]);
        carry >>= 32;
        #pragma clang loop unroll(full)
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

// SOS squaring: 36 cross-term muls + 8 diagonal vs 64 for general mul.
// Uses uint t[17] (not ulong) to avoid excessive register pressure.
Fp fp_sqr(Fp a) {
    uint t[17];
    for (int i = 0; i < 17; i++) t[i] = 0;

    for (int i = 0; i < 7; i++) {
        ulong carry = 0;
        for (int j = i + 1; j < 8; j++) {
            carry += ulong(t[i + j]) + ulong(a.v[i]) * ulong(a.v[j]);
            t[i + j] = uint(carry & 0xFFFFFFFF);
            carry >>= 32;
        }
        t[i + 8] += uint(carry);
    }

    uint top_carry = 0;
    for (int i = 1; i < 16; i++) {
        ulong doubled = (ulong(t[i]) << 1) | ulong(top_carry);
        t[i] = uint(doubled & 0xFFFFFFFF);
        top_carry = uint(doubled >> 32);
    }

    ulong carry = 0;
    for (int i = 0; i < 8; i++) {
        carry += ulong(t[2*i]) + ulong(a.v[i]) * ulong(a.v[i]);
        t[2*i] = uint(carry & 0xFFFFFFFF);
        carry >>= 32;
        carry += ulong(t[2*i + 1]);
        t[2*i + 1] = uint(carry & 0xFFFFFFFF);
        carry >>= 32;
    }

    for (int i = 0; i < 8; i++) {
        uint m = t[i] * INV;
        ulong c = ulong(t[i]) + ulong(m) * ulong(P[0]);
        c >>= 32;
        for (int j = 1; j < 8; j++) {
            c += ulong(t[i + j]) + ulong(m) * ulong(P[j]);
            t[i + j] = uint(c & 0xFFFFFFFF);
            c >>= 32;
        }
        for (int j = i + 8; j < 16; j++) {
            c += ulong(t[j]);
            t[j] = uint(c & 0xFFFFFFFF);
            c >>= 32;
            if (c == 0) break;
        }
    }

    Fp r;
    for (int i = 0; i < 8; i++) r.v[i] = t[i + 8];
    if (fp_gte(r, fp_modulus())) {
        uint borrow;
        r = fp_sub_raw(r, fp_modulus(), borrow);
    }
    return r;
}

// Double in field: left-shift by 1, conditional subtract
Fp fp_double(Fp a) {
    Fp r;
    uint carry = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint doubled = (a.v[i] << 1) | carry;
        carry = a.v[i] >> 31;
        r.v[i] = doubled;
    }
    if (carry || fp_gte(r, fp_modulus())) {
        uint borrow;
        r = fp_sub_raw(r, fp_modulus(), borrow);
    }
    return r;
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

// Reduce a field element to [0, p) if it's in the weakly-reduced range [0, 2p).
// Barretenberg stores Montgomery form values that may not be fully reduced.
Fp fp_reduce(Fp a) {
    uint borrow;
    Fp reduced = fp_sub_raw(a, fp_modulus(), borrow);
    return borrow ? a : reduced;
}

bool point_is_identity(PointProjective p) {
    return fp_is_zero(p.z);
}

// Convert affine to projective, reducing coordinates to [0, p)
PointProjective point_from_affine(PointAffine a) {
    PointProjective p;
    p.x = fp_reduce(a.x);
    p.y = fp_reduce(a.y);
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
// Register-pressure-optimized: Z3 = 2*Z1*H frees p.z and z1z1 early.
PointProjective point_add_mixed_unsafe(PointProjective p, PointAffine q) {
    Fp z1z1 = fp_sqr(p.z);                      // Z1^2
    Fp u2 = fp_mul(q.x, z1z1);                  // U2 = X2*Z1^2
    Fp s2 = fp_mul(q.y, fp_mul(p.z, z1z1));     // S2 = Y2*Z1^3 (z1z1 last use)

    Fp h = fp_sub(u2, p.x);                     // H = U2 - X1 (u2 dead)

    // Compute Z3 early to free p.z
    PointProjective result;
    result.z = fp_double(fp_mul(p.z, h));        // Z3 = 2*Z1*H (p.z dead)

    Fp hh = fp_sqr(h);                           // H^2
    Fp i = fp_double(fp_double(hh));              // I = 4*H^2 (hh dead)
    Fp v = fp_mul(p.x, i);                       // V = X1*I (p.x dead)
    Fp j = fp_mul(h, i);                          // J = H*I (h dead, i dead)
    Fp rr = fp_double(fp_sub(s2, p.y));           // r = 2*(S2-Y1) (s2 dead)

    result.x = fp_sub(fp_sub(fp_sqr(rr), j), fp_double(v)); // X3 = r^2 - J - 2*V
    result.y = fp_sub(fp_mul(rr, fp_sub(v, result.x)),
                      fp_double(fp_mul(p.y, j)));             // Y3 = r*(V-X3) - 2*Y1*J
    return result;
}

// Mixed addition with full edge case handling (safe version)
// Handles: p==identity, p==q (doubling), p==-q (returns identity)
PointProjective point_add_mixed(PointProjective p, PointAffine q) {
    if (point_is_identity(p)) return point_from_affine(q);

    Fp z1z1 = fp_sqr(p.z);                      // Z1^2
    Fp u2 = fp_mul(q.x, z1z1);                  // U2 = X2*Z1^2
    Fp s2 = fp_mul(q.y, fp_mul(p.z, z1z1));     // S2 = Y2*Z1^3

    Fp h = fp_sub(u2, p.x);                     // H = U2 - X1

    // Edge case: same x-coordinate (P==Q or P==-Q)
    if (fp_is_zero(h)) {
        Fp rr = fp_double(fp_sub(s2, p.y));      // r = 2*(S2-Y1)
        if (fp_is_zero(rr)) {
            // P == Q: need doubling
            return point_double(p);
        }
        // P == -Q: result is identity
        return point_identity();
    }

    // Normal case: use fast path
    PointProjective result;
    result.z = fp_double(fp_mul(p.z, h));        // Z3 = 2*Z1*H

    Fp hh = fp_sqr(h);                           // H^2
    Fp i = fp_double(fp_double(hh));              // I = 4*H^2
    Fp v = fp_mul(p.x, i);                       // V = X1*I
    Fp j = fp_mul(h, i);                          // J = H*I
    Fp rr = fp_double(fp_sub(s2, p.y));           // r = 2*(S2-Y1)

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

// --- XYZZ Coordinates for Faster Mixed Addition ---
// XYZZ stores (X, Y, ZZ=Z^2, ZZZ=Z^3). Mixed addition costs 7M+2S = 9 muls
// instead of 8M+3S = 11 muls in Jacobian. Projective-projective remains 14 muls.

struct PointXYZZ {
    Fp x;
    Fp y;
    Fp zz;   // Z^2
    Fp zzz;  // Z^3
};

PointXYZZ xyzz_identity() {
    PointXYZZ p;
    p.x = fp_zero();
    p.y = fp_one();
    p.zz = fp_zero();
    p.zzz = fp_zero();
    return p;
}

bool xyzz_is_identity(PointXYZZ p) {
    return fp_is_zero(p.zz);
}

PointXYZZ xyzz_from_affine(PointAffine a) {
    PointXYZZ p;
    p.x = fp_reduce(a.x);
    p.y = fp_reduce(a.y);
    p.zz = fp_one();
    p.zzz = fp_one();
    return p;
}

// XYZZ mixed addition: (X1,Y1,ZZ1,ZZZ1) + (x2,y2)
// Cost: 7M + 2S = 9 multiplications (vs 11 for Jacobian)
// Uses fp_sqr for better ILP in the reduce kernel hot path.
PointXYZZ xyzz_add_mixed_unsafe(PointXYZZ p, PointAffine q) {
    Fp P = fp_sub(fp_mul(q.x, p.zz), p.x);       // P = x2*ZZ1 - X1  (1M)
    Fp R = fp_sub(fp_mul(q.y, p.zzz), p.y);       // R = y2*ZZZ1 - Y1 (1M)
    Fp PP = fp_sqr(P);                          // PP = P^2          (1S)
    Fp PPP = fp_mul(P, PP);                         // PPP = P*PP        (1M)
    Fp Q = fp_mul(p.x, PP);                         // Q = X1*PP         (1M)

    PointXYZZ result;
    result.x = fp_sub(fp_sub(fp_sqr(R), PPP), fp_double(Q)); // X3 = R^2 - PPP - 2Q (1S)
    result.y = fp_sub(fp_mul(R, fp_sub(Q, result.x)),
                      fp_mul(p.y, PPP));            // Y3 = R*(Q-X3) - Y1*PPP (2M)
    result.zz = fp_mul(p.zz, PP);                   // ZZ3 = ZZ1*PP     (1M)
    result.zzz = fp_mul(p.zzz, PPP);                // ZZZ3 = ZZZ1*PPP  (1M)
    return result;                                   // Total: 7M + 2S = 9
}

// XYZZ full addition: (X1,Y1,ZZ1,ZZZ1) + (X2,Y2,ZZ2,ZZZ2)
// Cost: 10M + 2S = 12 multiplications
PointXYZZ xyzz_add(PointXYZZ p, PointXYZZ q) {
    if (xyzz_is_identity(p)) return q;
    if (xyzz_is_identity(q)) return p;

    Fp U1 = fp_mul(p.x, q.zz);    // U1 = X1*ZZ2
    Fp U2 = fp_mul(q.x, p.zz);    // U2 = X2*ZZ1
    Fp S1 = fp_mul(p.y, q.zzz);   // S1 = Y1*ZZZ2
    Fp S2 = fp_mul(q.y, p.zzz);   // S2 = Y2*ZZZ1

    Fp H = fp_sub(U2, U1);
    Fp R = fp_sub(S2, S1);

    if (fp_is_zero(H)) {
        if (fp_is_zero(R)) {
            // P == Q: doubling
            // XYZZ doubling: 2M + 3S for a=0
            Fp U = fp_double(p.y);
            Fp V = fp_sqr(U);
            Fp W = fp_mul(U, V);
            Fp S = fp_double(fp_mul(p.x, V));
            Fp M = fp_add(fp_double(fp_sqr(p.x)), fp_sqr(p.x)); // 3*X^2
            PointXYZZ r;
            r.x = fp_sub(fp_sqr(M), fp_double(S));
            r.y = fp_sub(fp_mul(M, fp_sub(S, r.x)), fp_mul(W, p.y));
            r.zz = V;
            r.zzz = W;
            return r;
        }
        return xyzz_identity();
    }

    Fp HH = fp_sqr(H);                   // H^2
    Fp HHH = fp_mul(H, HH);                 // H^3
    Fp V = fp_mul(U1, HH);                  // V = U1*H^2

    PointXYZZ result;
    result.x = fp_sub(fp_sub(fp_sqr(R), HHH), fp_double(V));
    result.y = fp_sub(fp_mul(R, fp_sub(V, result.x)), fp_mul(S1, HHH));
    result.zz = fp_mul(fp_mul(p.zz, q.zz), HH);
    result.zzz = fp_mul(fp_mul(p.zzz, q.zzz), HHH);
    return result;
}

// Convert XYZZ to Jacobian projective (for final output)
// Z = ZZZ / ZZ. But division is expensive, so we use a trick:
// Output projective (X', Y', Z') where:
//   X' = X * ZZ, Y' = Y * ZZZ, Z' = ZZ * ZZZ (then X/Z^2 = X*ZZ/(ZZ*ZZZ)^2 works out)
// Actually, simpler: (X, Y, ZZ, ZZZ) maps to affine as x=X/ZZ, y=Y/ZZZ
// For Jacobian (X,Y,Z): x=X/Z^2, y=Y/Z^3
// Set Z = ZZZ/ZZ, then Z^2 = ZZZ^2/ZZ^2, Z^3 = ZZZ^3/ZZ^3
// That requires a division. Instead, output as:
//   X_j = X * ZZZ^2, Y_j = Y * ZZZ^3, Z_j = ZZ * ZZZ
// Then X_j/Z_j^2 = X*ZZZ^2/(ZZ*ZZZ)^2 = X*ZZZ^2/(ZZ^2*ZZZ^2) = X/ZZ^2 ... that's wrong.
// Let's just do the conversion properly:
//   Z = ZZZ * inv(ZZ) -- too expensive
// Better: keep XYZZ and convert to affine at the very end.
// For the Horner combine on CPU, we convert each window result XYZZ → Jacobian:
//   Z_j = ZZZ (arbitrary nonzero); X_j = X * ZZZ; Y_j = Y * ZZZ^2 / ZZ
// Actually this is getting complicated. Simplest correct conversion:
//   X_j = X, Y_j = Y, Z_j = 1 (WRONG — not affine)
// Let's use: affine x = X * inv(ZZ), y = Y * inv(ZZZ). Do batch inversion on CPU.
// For GPU kernel output: just output XYZZ and let CPU convert.
// But we need Jacobian for bucket_sum compatibility... let me just output XYZZ buckets
// and use xyzz_add in bucket_sum.

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

// Phase 0.5: Gather points into sorted order for sequential GPU access.
// sorted_indices[i] contains the original point index and a sign flag (bit 31).
// This kernel copies points into a contiguous sorted_points buffer, applying
// the sign negation. The reduce kernel then reads sequentially instead of randomly.
kernel void msm_gather_sorted_points(
    device const PointAffine* points           [[buffer(0)]],
    device PointAffine* sorted_points          [[buffer(1)]],
    device const uint* sorted_indices          [[buffer(2)]],
    constant uint& total_entries               [[buffer(3)]],
    uint tid                                   [[thread_position_in_grid]]
) {
    if (tid >= total_entries) return;
    uint raw = sorted_indices[tid];
    uint pidx = raw & 0x7FFFFFFFu;
    PointAffine pt = points[pidx];
    if (raw & 0x80000000u) pt.y = fp_neg(pt.y);
    sorted_points[tid] = pt;
}

// Phase 1: Reduce pre-sorted points per bucket. No atomics needed.
// Batched across all windows: tid = window_idx * n_buckets + bucket_idx
// sorted_indices layout: [window0: n_points][window1: n_points]...

// Threshold for parallel bucket reduction. Buckets with more points than this
// are handled by msm_reduce_large_bucket (256-thread parallel tree reduce).
// 256 is optimal: lower threshold (128) tested and was slower; higher (512) also slower.
constant uint LARGE_BUCKET_THRESHOLD = 256;

// Fused gather+reduce: reads directly from sorted_indices and original points.
// Reduce is compute-bound (~81ms for 1M points, of which ~58ms is field muls).
// Random access adds only ~2ms overhead vs sequential reads — GPU thread count (295K)
// provides sufficient latency hiding. Gathering points was tested and added net overhead.
kernel void msm_reduce_gathered(
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

    uint orig_pos = count_sorted_map[tid];
    uint orig_bucket = orig_pos % nb;

    if (orig_bucket == 0) {
        buckets[orig_pos] = point_identity();
        return;
    }

    uint count = bucket_counts[orig_pos];
    if (count == 0) {
        buckets[orig_pos] = point_identity();
        return;
    }

    // Skip large buckets — handled by reduce_large_bucket kernel (256-thread tree reduction).
    if (count > LARGE_BUCKET_THRESHOLD) {
        buckets[orig_pos] = point_identity();
        return;
    }

    uint window_idx = orig_pos / nb;
    uint base = window_idx * params.n_points;
    uint offset = bucket_offsets[orig_pos];

    uint b_off = base + offset;
    uint raw0 = sorted_indices[b_off];
    PointAffine pt0 = points[raw0 & 0x7FFFFFFFu];
    if (raw0 & 0x80000000u) pt0.y = fp_neg(pt0.y);
    PointProjective acc = point_from_affine(pt0);

    for (uint i = 1; i < count; i++) {
        uint raw = sorted_indices[b_off + i];
        PointAffine pt = points[raw & 0x7FFFFFFFu];
        if (raw & 0x80000000u) pt.y = fp_neg(pt.y);
        acc = point_add_mixed_unsafe(acc, pt);
    }
    buckets[orig_pos] = acc;
}

// Per-window reduce: processes one window at a time from a single-window sorted_points buffer.
// sorted_points[0..n_points] contains gathered points for the current window only.
// tid maps to a bucket within the current window (0..n_buckets-1).
kernel void msm_reduce_gathered_single_window(
    device const PointAffine* sorted_points    [[buffer(0)]],
    device PointProjective* buckets            [[buffer(1)]],
    device const uint* bucket_offsets          [[buffer(2)]],
    device const uint* bucket_counts           [[buffer(3)]],
    constant uint& window_offset               [[buffer(4)]],  // w * n_buckets
    constant uint& n_buckets                   [[buffer(5)]],
    device const uint* count_sorted_map        [[buffer(7)]],
    uint tid                                   [[thread_position_in_grid]]
) {
    if (tid >= n_buckets) return;

    // CSM maps tid to original bucket position within this window
    uint csm_idx = window_offset + tid;
    uint orig_pos = count_sorted_map[csm_idx];
    uint orig_bucket = orig_pos % n_buckets;

    if (orig_bucket == 0) {
        buckets[orig_pos] = point_identity();
        return;
    }

    uint count = bucket_counts[orig_pos];
    if (count == 0) {
        buckets[orig_pos] = point_identity();
        return;
    }

    if (count > LARGE_BUCKET_THRESHOLD) {
        return;  // Handled by parallel large-bucket kernel
    }

    uint offset = bucket_offsets[orig_pos];

    // Use XYZZ coordinates for faster mixed addition (9 muls vs 11 Jacobian)
    PointXYZZ acc = xyzz_from_affine(sorted_points[offset]);

    for (uint i = 1; i < count; i++) {
        acc = xyzz_add_mixed_unsafe(acc, sorted_points[offset + i]);
    }

    // Convert XYZZ → Jacobian: X_j = X*ZZ, Y_j = Y*ZZZ, Z_j = ZZ
    // Derivation: affine x = X/ZZ = (X*ZZ)/ZZ^2, y = Y/ZZZ = (Y*ZZZ)/ZZ^3
    PointProjective result;
    result.x = fp_mul(acc.x, acc.zz);
    result.y = fp_mul(acc.y, acc.zzz);
    result.z = acc.zz;
    buckets[orig_pos] = result;
}

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

    // Count-sorted mapping: adjacent threads process buckets with similar point counts.
    // This eliminates SIMD thread divergence — all threads in a warp finish together.
    uint orig_pos = count_sorted_map[tid];
    uint orig_bucket = orig_pos % nb;

    if (orig_bucket == 0) {
        buckets[orig_pos] = point_identity();
        return;
    }

    uint count = bucket_counts[orig_pos];
    if (count == 0) {
        buckets[orig_pos] = point_identity();
        return;
    }

    // Large buckets are handled by msm_reduce_large_bucket (256-thread parallel reduce).
    // Leave those as identity (blit-zeroed); that kernel overwrites with the correct result.
    if (count > LARGE_BUCKET_THRESHOLD) {
        return;
    }

    uint window_idx = orig_pos / nb;
    uint base = window_idx * params.n_points;
    uint offset = bucket_offsets[orig_pos];

    // Read first point and initialize accumulator.
    // Bit 31 of sorted_indices encodes the signed-digit sign flag.
    // When set, negate Y coordinate (p - y) before accumulating.
    uint b_off = base + offset;
    uint raw0 = sorted_indices[b_off];
    uint pidx0 = raw0 & 0x7FFFFFFFu;
    PointAffine pt0 = points[pidx0];
    if (raw0 & 0x80000000u) pt0.y = fp_neg(pt0.y);

    // Use XYZZ coordinates for faster mixed addition (9 muls vs 11 Jacobian)
    PointXYZZ acc = xyzz_from_affine(pt0);

    // Use unsafe mixed addition: skips identity/doubling checks.
    // Safe for MSM because bucket points are random — P==Q or P==-Q
    // has negligible probability O(1/p) ≈ 2^{-254}.
    for (uint i = 1; i < count; i++) {
        uint raw = sorted_indices[b_off + i];
        uint pidx = raw & 0x7FFFFFFFu;
        PointAffine pt = points[pidx];
        if (raw & 0x80000000u) pt.y = fp_neg(pt.y);
        acc = xyzz_add_mixed_unsafe(acc, pt);
    }

    // Convert XYZZ → Jacobian: X_j = X*ZZ, Y_j = Y*ZZZ, Z_j = ZZ
    PointProjective result;
    result.x = fp_mul(acc.x, acc.zz);
    result.y = fp_mul(acc.y, acc.zzz);
    result.z = acc.zz;
    buckets[orig_pos] = result;
}

// Parallel reduce: 1 threadgroup (32 threads) per bucket.
// Each thread accumulates its chunk of points via XYZZ mixed-add,
// then tree-reduce partial sums via projective addition in shared memory.
// This issues 32x more concurrent memory requests vs the sequential kernel,
// dramatically reducing memory-latency stalls for random-access point reads.
#define PARALLEL_REDUCE_TG_SIZE 32

kernel void msm_reduce_parallel(
    device const PointAffine* points           [[buffer(0)]],
    device PointProjective* buckets            [[buffer(1)]],
    device const uint* bucket_offsets          [[buffer(2)]],
    device const uint* bucket_counts           [[buffer(3)]],
    constant MsmParams& params                 [[buffer(4)]],
    constant uint& n_windows                   [[buffer(5)]],
    device const uint* sorted_indices          [[buffer(6)]],
    device const uint* count_sorted_map        [[buffer(7)]],
    uint group_id                              [[threadgroup_position_in_grid]],
    uint local_id                              [[thread_position_in_threadgroup]]
) {
    uint nb = params.n_buckets;
    uint total_buckets = nb * n_windows;
    if (group_id >= total_buckets) return;

    uint orig_pos = count_sorted_map[group_id];
    uint orig_bucket = orig_pos % nb;

    // Bucket 0 is identity (never has real points assigned)
    if (orig_bucket == 0) {
        if (local_id == 0) buckets[orig_pos] = point_identity();
        return;
    }

    uint count = bucket_counts[orig_pos];
    if (count == 0) {
        if (local_id == 0) buckets[orig_pos] = point_identity();
        return;
    }

    uint window_idx = orig_pos / nb;
    uint base = window_idx * params.n_points;
    uint offset = bucket_offsets[orig_pos];
    uint b_off = base + offset;

    // Each thread reduces its chunk via XYZZ mixed-add
    uint my_start = (count * local_id) / PARALLEL_REDUCE_TG_SIZE;
    uint my_end = (count * (local_id + 1)) / PARALLEL_REDUCE_TG_SIZE;

    PointProjective local_acc = point_identity();
    if (my_start < my_end) {
        uint raw0 = sorted_indices[b_off + my_start];
        uint pidx0 = raw0 & 0x7FFFFFFFu;
        PointAffine pt0 = points[pidx0];
        if (raw0 & 0x80000000u) pt0.y = fp_neg(pt0.y);

        PointXYZZ xyzz_acc = xyzz_from_affine(pt0);
        for (uint i = my_start + 1; i < my_end; i++) {
            uint raw = sorted_indices[b_off + i];
            uint pidx = raw & 0x7FFFFFFFu;
            PointAffine pt = points[pidx];
            if (raw & 0x80000000u) pt.y = fp_neg(pt.y);
            xyzz_acc = xyzz_add_mixed_unsafe(xyzz_acc, pt);
        }

        // Convert XYZZ -> projective for tree reduction
        local_acc.x = fp_mul(xyzz_acc.x, xyzz_acc.zz);
        local_acc.y = fp_mul(xyzz_acc.y, xyzz_acc.zzz);
        local_acc.z = xyzz_acc.zz;
    }

    // Tree reduction in shared memory
    threadgroup PointProjective shared_partials[PARALLEL_REDUCE_TG_SIZE];
    shared_partials[local_id] = local_acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = PARALLEL_REDUCE_TG_SIZE / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            PointProjective a = shared_partials[local_id];
            PointProjective b = shared_partials[local_id + stride];
            shared_partials[local_id] = point_add(a, b);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_id == 0) {
        buckets[orig_pos] = shared_partials[0];
    }
}

// Phase 1 half-reduce: 2 threads per bucket, each reduces half the points.
// Thread tid maps to bucket (tid/2), half (tid%2).
// Writes partial result to partial_results[2*bucket_pos + half].
// Buckets with count <= 1 are handled directly (no split needed).
kernel void msm_reduce_sorted_buckets_half(
    device const PointAffine* points           [[buffer(0)]],
    device PointProjective* partial_results    [[buffer(1)]],
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
    uint bucket_tid = tid / 2;
    uint half_idx = tid % 2;
    if (bucket_tid >= total) return;

    uint orig_pos = count_sorted_map[bucket_tid];
    uint orig_bucket = orig_pos % nb;

    if (orig_bucket == 0) {
        partial_results[2 * orig_pos + half_idx] = point_identity();
        return;
    }

    uint count = bucket_counts[orig_pos];
    if (count == 0) {
        partial_results[2 * orig_pos + half_idx] = point_identity();
        return;
    }

    // Split work: half_idx 0 gets [0, mid), half_idx 1 gets [mid, count)
    // Handles all bucket sizes including large (count > LARGE_BUCKET_THRESHOLD).
    uint mid = count / 2;
    uint my_start = (half_idx == 0) ? 0 : mid;
    uint my_end = (half_idx == 0) ? mid : count;

    if (my_start >= my_end) {
        partial_results[2 * orig_pos + half_idx] = point_identity();
        return;
    }

    uint window_idx = orig_pos / nb;
    uint base = window_idx * params.n_points;
    uint offset = bucket_offsets[orig_pos];
    uint b_off = base + offset;

    uint raw0 = sorted_indices[b_off + my_start];
    uint pidx0 = raw0 & 0x7FFFFFFFu;
    PointAffine pt0 = points[pidx0];
    if (raw0 & 0x80000000u) pt0.y = fp_neg(pt0.y);
    PointProjective acc = point_from_affine(pt0);

    for (uint i = my_start + 1; i < my_end; i++) {
        uint raw = sorted_indices[b_off + i];
        uint pidx = raw & 0x7FFFFFFFu;
        PointAffine pt = points[pidx];
        if (raw & 0x80000000u) pt.y = fp_neg(pt.y);
        acc = point_add_mixed_unsafe(acc, pt);
    }
    partial_results[2 * orig_pos + half_idx] = acc;
}

// Combine two partial results per bucket into final bucket value.
// Uses point_add (projective+projective) since both inputs are projective.
kernel void msm_combine_bucket_halves(
    device const PointProjective* partial_results  [[buffer(0)]],
    device PointProjective* buckets               [[buffer(1)]],
    device const uint* count_sorted_map           [[buffer(2)]],
    device const uint* bucket_counts              [[buffer(3)]],
    constant uint& n_buckets                      [[buffer(4)]],
    constant uint& n_windows                      [[buffer(5)]],
    uint tid                                      [[thread_position_in_grid]]
) {
    uint total = n_buckets * n_windows;
    if (tid >= total) return;

    uint orig_pos = count_sorted_map[tid];
    uint orig_bucket = orig_pos % n_buckets;

    if (orig_bucket == 0) {
        buckets[orig_pos] = point_identity();
        return;
    }

    uint count = bucket_counts[orig_pos];
    if (count == 0) {
        buckets[orig_pos] = point_identity();
        return;
    }

    // Combine both halves for all bucket sizes (including large buckets).
    // Half-reduce kernel now handles all sizes, so both halves always have valid results.
    PointProjective a = partial_results[2 * orig_pos];
    PointProjective b = partial_results[2 * orig_pos + 1];
    buckets[orig_pos] = point_add(a, b);
}

// Phase 1 quarter-reduce: 4 threads per bucket, each reduces 1/4 of the points.
// Thread tid maps to bucket (tid/4), quarter (tid%4).
// Writes partial result to partial_results[4*bucket_pos + quarter].
kernel void msm_reduce_sorted_buckets_quarter(
    device const PointAffine* points           [[buffer(0)]],
    device PointProjective* partial_results    [[buffer(1)]],
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
    uint bucket_tid = tid / 4;
    uint quarter_idx = tid % 4;
    if (bucket_tid >= total) return;

    uint orig_pos = count_sorted_map[bucket_tid];
    uint orig_bucket = orig_pos % nb;

    if (orig_bucket == 0) {
        partial_results[4 * orig_pos + quarter_idx] = point_identity();
        return;
    }

    uint count = bucket_counts[orig_pos];
    if (count == 0 || count > LARGE_BUCKET_THRESHOLD) {
        partial_results[4 * orig_pos + quarter_idx] = point_identity();
        return;
    }

    // Split work into 4 quarters
    uint q_start = (count * quarter_idx) / 4;
    uint q_end = (count * (quarter_idx + 1)) / 4;

    if (q_start >= q_end) {
        partial_results[4 * orig_pos + quarter_idx] = point_identity();
        return;
    }

    uint window_idx = orig_pos / nb;
    uint base = window_idx * params.n_points;
    uint offset = bucket_offsets[orig_pos];
    uint b_off = base + offset;

    uint raw0 = sorted_indices[b_off + q_start];
    uint pidx0 = raw0 & 0x7FFFFFFFu;
    PointAffine pt0 = points[pidx0];
    if (raw0 & 0x80000000u) pt0.y = fp_neg(pt0.y);
    PointProjective acc = point_from_affine(pt0);

    for (uint i = q_start + 1; i < q_end; i++) {
        uint raw = sorted_indices[b_off + i];
        uint pidx = raw & 0x7FFFFFFFu;
        PointAffine pt = points[pidx];
        if (raw & 0x80000000u) pt.y = fp_neg(pt.y);
        acc = point_add_mixed_unsafe(acc, pt);
    }
    partial_results[4 * orig_pos + quarter_idx] = acc;
}

// Combine four partial results per bucket into final bucket value.
// Tree combine: (0+1) + (2+3), then combine those two.
kernel void msm_combine_quarter_results(
    device const PointProjective* partial_results  [[buffer(0)]],
    device PointProjective* buckets               [[buffer(1)]],
    device const uint* count_sorted_map           [[buffer(2)]],
    device const uint* bucket_counts              [[buffer(3)]],
    constant uint& n_buckets                      [[buffer(4)]],
    constant uint& n_windows                      [[buffer(5)]],
    uint tid                                      [[thread_position_in_grid]]
) {
    uint total = n_buckets * n_windows;
    if (tid >= total) return;

    uint orig_pos = count_sorted_map[tid];
    uint orig_bucket = orig_pos % n_buckets;

    if (orig_bucket == 0) {
        buckets[orig_pos] = point_identity();
        return;
    }

    uint count = bucket_counts[orig_pos];
    if (count == 0) {
        buckets[orig_pos] = point_identity();
        return;
    }
    if (count > LARGE_BUCKET_THRESHOLD) {
        return; // handled by large bucket kernel
    }

    uint base = 4 * orig_pos;
    PointProjective a = point_add(partial_results[base], partial_results[base + 1]);
    PointProjective b = point_add(partial_results[base + 2], partial_results[base + 3]);
    buckets[orig_pos] = point_add(a, b);
}

// Parallel reduce for large buckets (count > LARGE_BUCKET_THRESHOLD).
// One threadgroup of 256 threads per bucket. Each thread reduces its portion,
// then thread 0 combines all partial results sequentially.
// This turns a 12ms single-thread bottleneck into ~30ms parallel work.
// Tested 64/32 thread variants but 256 achieves the best lb time on M3 Pro.
kernel void msm_reduce_large_bucket(
    device const PointAffine* points           [[buffer(0)]],
    device PointProjective* buckets            [[buffer(1)]],
    device const uint* bucket_offsets          [[buffer(2)]],
    device const uint* bucket_counts           [[buffer(3)]],
    device const uint* sorted_indices          [[buffer(4)]],
    device const uint* large_bucket_ids        [[buffer(5)]],
    constant uint& n_points_per_window         [[buffer(6)]],
    constant uint& n_buckets                   [[buffer(7)]],
    uint group_id                              [[threadgroup_position_in_grid]],
    uint local_id                              [[thread_position_in_threadgroup]],
    uint tg_size                               [[threads_per_threadgroup]]
) {
    uint bucket_pos = large_bucket_ids[group_id];
    uint count = bucket_counts[bucket_pos];
    uint window_idx = bucket_pos / n_buckets;
    uint base = window_idx * n_points_per_window;
    uint offset = bucket_offsets[bucket_pos];

    // Each thread reduces its chunk of the bucket
    uint my_start = (count * local_id) / tg_size;
    uint my_end = (count * (local_id + 1)) / tg_size;

    PointProjective local_acc = point_identity();
    if (my_start < my_end) {
        uint raw0 = sorted_indices[base + offset + my_start];
        uint pidx0 = raw0 & 0x7FFFFFFFu;
        PointAffine pt0 = points[pidx0];
        if (raw0 & 0x80000000u) pt0.y = fp_neg(pt0.y);
        local_acc = point_from_affine(pt0);
        for (uint i = my_start + 1; i < my_end; i++) {
            uint raw = sorted_indices[base + offset + i];
            uint pidx = raw & 0x7FFFFFFFu;
            PointAffine pt = points[pidx];
            if (raw & 0x80000000u) pt.y = fp_neg(pt.y);
            local_acc = point_add_mixed_unsafe(local_acc, pt);
        }
    }

    // Store partial result in threadgroup memory.
    // 256 threads × 96 bytes = 24KB per TG; tested 64/32 variants but 256 is fastest on M3 Pro.
    threadgroup PointProjective shared_partials[256];
    shared_partials[local_id] = local_acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction: combine partial results
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            PointProjective a = shared_partials[local_id];
            PointProjective b = shared_partials[local_id + stride];
            shared_partials[local_id] = point_add(a, b);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_id == 0) {
        buckets[bucket_pos] = shared_partials[0];
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

    PointProjective running = point_identity();
    PointProjective sum = point_identity();

    uint hi = uint(hi_s);
    uint lo = uint(lo_s);
    // Unconditional point_add: identity cases handled inside point_add
    // (returns q when p is identity, returns p when q is identity).
    // Removing manual identity checks eliminates 6 branch points per iteration,
    // reducing SIMD divergence in the GPU's SIMD groups.
    for (uint i = hi - 1; i >= lo; i--) {
        running = point_add(running, buckets[bucket_base + i]);
        sum = point_add(sum, running);
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
                weighted = point_add(weighted, base);
            }
            base = point_double(base);
            k >>= 1;
        }
        sum = point_add(sum, weighted);
    }

    segment_results[tid] = sum;
}

// ======================== GLV Endomorphism ========================
// Decomposes 256-bit scalars into 128-bit half-scalars using the BN254 GLV endomorphism.
// k·P = k1·P + k2·φ(P) where φ(P) = (β·x, y) and k1,k2 are ~128-bit.
// This halves the number of windows needed for MSM (8 instead of 16 at w=16).

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

// 256-bit unsigned integer helpers (ulong limbs for scalar arithmetic)
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

// GLV Endomorphism Kernel
// Applies φ(P) = (β·x, y) and optional negation for GLV MSM.
// Reads original points[0..n-1], writes endomorphism points[n..2n-1]
// and optionally negates original points based on neg1 flags.
kernel void glv_endomorphism(
    device const PointAffine* srs_points [[buffer(0)]],
    device PointAffine* points [[buffer(1)]],
    const device uchar* neg1_flags [[buffer(2)]],
    const device uchar* neg2_flags [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    PointAffine p = srs_points[gid];
    // Reduce coordinates from barretenberg's weakly-reduced [0, 2p) to [0, p)
    p.x = fp_reduce(p.x);
    p.y = fp_reduce(p.y);

    // Save original y for endomorphism before potential negation
    Fp orig_y = p.y;

    // Apply neg1 to original point
    if (neg1_flags[gid]) {
        p.y = fp_neg(p.y);
    }
    points[gid] = p;

    // β in Montgomery form: cube root of unity in Fp
    Fp beta;
    beta.v[0] = 0xd782e155u; beta.v[1] = 0x71930c11u;
    beta.v[2] = 0xffbe3323u; beta.v[3] = 0xa6bb947cu;
    beta.v[4] = 0xd4741444u; beta.v[5] = 0xaa303344u;
    beta.v[6] = 0x26594943u; beta.v[7] = 0x2c3b3f0du;

    PointAffine endo;
    endo.x = fp_mul(beta, p.x); // fp_mul always produces [0, p)

    if (neg2_flags[gid]) {
        endo.y = fp_neg(orig_y);
    } else {
        endo.y = orig_y;
    }

    points[n + gid] = endo;
}

// Precompute reduced + endomorphism points for SRS caching.
// Reads raw SRS points (potentially weakly-reduced [0,2p)),
// writes reduced original points to cache[0..n-1] and
// endomorphism points (beta*x, y) to cache[n..2n-1].
// This is run once per SRS and cached across MSMs.
kernel void glv_precompute_cache(
    device const PointAffine* srs_points [[buffer(0)]],
    device PointAffine* cache [[buffer(1)]],  // 2n entries
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    PointAffine p = srs_points[gid];
    p.x = fp_reduce(p.x);
    p.y = fp_reduce(p.y);
    cache[gid] = p;

    Fp beta;
    beta.v[0] = 0xd782e155u; beta.v[1] = 0x71930c11u;
    beta.v[2] = 0xffbe3323u; beta.v[3] = 0xa6bb947cu;
    beta.v[4] = 0xd4741444u; beta.v[5] = 0xaa303344u;
    beta.v[6] = 0x26594943u; beta.v[7] = 0x2c3b3f0du;

    PointAffine endo;
    endo.x = fp_mul(beta, p.x);
    endo.y = p.y;
    cache[n + gid] = endo;
}

// Fast endomorphism from cache: just apply neg flags, no fp_mul needed.
// Reads pre-reduced points from cache, writes to points buffer with negation.
kernel void glv_apply_neg_flags(
    device const PointAffine* cache [[buffer(0)]],  // 2n entries (original + endo)
    device PointAffine* points [[buffer(1)]],       // output: 2n entries
    const device uchar* neg1_flags [[buffer(2)]],
    const device uchar* neg2_flags [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    PointAffine p = cache[gid];
    if (neg1_flags[gid]) {
        p.y = fp_neg(p.y);
    }
    points[gid] = p;

    PointAffine endo = cache[n + gid];
    if (neg2_flags[gid]) {
        endo.y = fp_neg(endo.y);
    }
    points[n + gid] = endo;
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
    // Each threadgroup = one window. lid indexes segments within this window.
    threadgroup PointProjective shared_buf[256]; // max segments per threadgroup

    uint base = tgid * n_segments;

    // Load: each thread loads one segment result (or identity if out of range)
    if (lid < n_segments) {
        shared_buf[lid] = segment_results[base + lid];
    } else {
        shared_buf[lid] = point_identity();
    }
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

// Two-level combine: L1 pass reduces a 256-element chunk of a large segment array.
// Each threadgroup handles one chunk of one window.
// tgid = window_idx * n_chunks + chunk_idx
// Writes one result per (window, chunk) to chunk_results.
kernel void msm_combine_segments_l1(
    device const PointProjective* segment_results [[buffer(0)]],
    device PointProjective* chunk_results         [[buffer(1)]],
    constant uint& n_segments                     [[buffer(2)]],  // total segments per window
    constant uint& chunk_size                     [[buffer(3)]],  // segments per chunk (256)
    uint tgid                                     [[threadgroup_position_in_grid]],
    uint lid                                      [[thread_index_in_threadgroup]],
    uint tg_size                                  [[threads_per_threadgroup]]
) {
    threadgroup PointProjective shared_buf[256];

    // tgid = window_idx * n_chunks + chunk_idx
    uint n_chunks = (n_segments + chunk_size - 1) / chunk_size;
    uint window_idx = tgid / n_chunks;
    uint chunk_idx  = tgid % n_chunks;
    uint seg_base   = window_idx * n_segments + chunk_idx * chunk_size;

    // Load one segment per thread (or identity if out of range)
    uint seg_local = chunk_idx * chunk_size + lid;
    if (lid < chunk_size && seg_local < n_segments) {
        shared_buf[lid] = segment_results[seg_base + lid];
    } else {
        shared_buf[lid] = point_identity();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction within this chunk
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
        chunk_results[tgid] = shared_buf[0];
    }
}

// ---- GPU Counting Sort Kernels ----
// Replace CPU counting sort with GPU device-atomic histogram + scatter.
// Eliminates ~30ms CPU sort per MSM.

// Phase 2a: Build per-window histograms using device atomics.
// Each thread processes one point across all windows.
// Uses signed-digit decomposition: digits in [-half_nb, half_nb] with carry chain.
// This halves the number of buckets (32769 vs 65536) for faster bucket_sum.
kernel void msm_sort_histogram(
    device const uint* k1_data [[buffer(0)]],
    device const uint* k2_data [[buffer(1)]],
    device atomic_uint* all_counts [[buffer(2)]],  // n_windows * n_buckets
    constant uint& n [[buffer(3)]],
    constant uint& window_bits_val [[buffer(4)]],
    constant uint& n_buckets_val [[buffer(5)]],
    constant uint& n_windows_val [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint mask = (1u << window_bits_val) - 1;       // 0xFFFF for w=16
    uint full_nb = 1u << window_bits_val;            // 65536 for w=16
    uint half_nb = full_nb >> 1;                     // 32768 for w=16
    uint n_data_windows = n_windows_val - 1;         // 8 data windows + 1 carry overflow

    uint k1_base = gid * 8;
    uint k2_base = gid * 8;

    uint carry1 = 0, carry2 = 0;
    for (uint w = 0; w < n_data_windows; w++) {
        uint bit_offset = w * window_bits_val;
        uint limb_idx = bit_offset / 32;
        uint bit_pos = bit_offset % 32;

        uint d1 = k1_data[k1_base + limb_idx] >> bit_pos;
        uint d2 = k2_data[k2_base + limb_idx] >> bit_pos;
        if (bit_pos + window_bits_val > 32 && limb_idx + 1 < 8) {
            d1 |= k1_data[k1_base + limb_idx + 1] << (32 - bit_pos);
            d2 |= k2_data[k2_base + limb_idx + 1] << (32 - bit_pos);
        }
        d1 = (d1 & mask) + carry1;
        d2 = (d2 & mask) + carry2;
        carry1 = carry2 = 0;

        // Signed-digit conversion: if digit >= half_nb, negate and carry
        if (d1 >= half_nb) { d1 = full_nb - d1; carry1 = 1; }
        if (d2 >= half_nb) { d2 = full_nb - d2; carry2 = 1; }

        uint w_off = w * n_buckets_val;
        if (d1 != 0) atomic_fetch_add_explicit(&all_counts[w_off + d1], 1, memory_order_relaxed);
        if (d2 != 0) atomic_fetch_add_explicit(&all_counts[w_off + d2], 1, memory_order_relaxed);
    }
    // Carry overflow into extra window (digit is always 1, positive)
    uint carry_off = n_data_windows * n_buckets_val;
    if (carry1) atomic_fetch_add_explicit(&all_counts[carry_off + 1], 1, memory_order_relaxed);
    if (carry2) atomic_fetch_add_explicit(&all_counts[carry_off + 1], 1, memory_order_relaxed);
}

// Phase 2b: Compute prefix sums per window (each threadgroup = one window).
// Reads counts, writes offsets. Also copies counts to a separate buffer for later use.
kernel void msm_sort_prefix_sum(
    device const uint* counts_in [[buffer(0)]],   // n_windows * n_buckets (from histogram, read-only)
    device uint* offsets_out [[buffer(1)]],         // n_windows * n_buckets (prefix sums)
    device uint* scatter_pos [[buffer(2)]],         // n_windows * n_buckets (init to offsets for scatter)
    constant uint& n_buckets_val [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    // Each threadgroup handles one window. Thread 0 does sequential prefix sum.
    // counts_in is left unmodified (histogram already wrote correct values).
    if (lid != 0) return;

    uint base = tgid * n_buckets_val;
    uint running = 0;
    for (uint b = 0; b < n_buckets_val; b++) {
        uint c = counts_in[base + b];
        offsets_out[base + b] = running;
        scatter_pos[base + b] = running;
        running += c;
    }
}

// Phase 2c: Scatter point indices into sorted order using device atomics.
// Each thread processes one point across all windows.
// Uses signed-digit decomposition matching the histogram kernel.
// Sign bit is encoded in bit 31 of sorted_idx for reduce kernel.
kernel void msm_sort_scatter(
    device const uint* k1_data [[buffer(0)]],
    device const uint* k2_data [[buffer(1)]],
    device atomic_uint* scatter_pos [[buffer(2)]],  // per-bucket running position
    device uint* sorted_idx [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& window_bits_val [[buffer(5)]],
    constant uint& n_buckets_val [[buffer(6)]],
    constant uint& n_windows_val [[buffer(7)]],
    constant uint& n2 [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    uint mask = (1u << window_bits_val) - 1;
    uint full_nb = 1u << window_bits_val;
    uint half_nb = full_nb >> 1;
    uint n_data_windows = n_windows_val - 1;

    uint k1_base = gid * 8;
    uint k2_base = gid * 8;

    uint carry1 = 0, carry2 = 0;
    for (uint w = 0; w < n_data_windows; w++) {
        uint bit_offset = w * window_bits_val;
        uint limb_idx = bit_offset / 32;
        uint bit_pos = bit_offset % 32;

        uint d1 = k1_data[k1_base + limb_idx] >> bit_pos;
        uint d2 = k2_data[k2_base + limb_idx] >> bit_pos;
        if (bit_pos + window_bits_val > 32 && limb_idx + 1 < 8) {
            d1 |= k1_data[k1_base + limb_idx + 1] << (32 - bit_pos);
            d2 |= k2_data[k2_base + limb_idx + 1] << (32 - bit_pos);
        }
        d1 = (d1 & mask) + carry1;
        d2 = (d2 & mask) + carry2;
        carry1 = carry2 = 0;
        bool neg1 = false, neg2 = false;

        if (d1 >= half_nb) { d1 = full_nb - d1; carry1 = 1; neg1 = true; }
        if (d2 >= half_nb) { d2 = full_nb - d2; carry2 = 1; neg2 = true; }

        uint w_base = w * n_buckets_val;
        uint idx_base = w * n2;
        if (d1 != 0) {
            uint pos = atomic_fetch_add_explicit(&scatter_pos[w_base + d1], 1, memory_order_relaxed);
            sorted_idx[idx_base + pos] = gid | (neg1 ? 0x80000000u : 0u);
        }
        if (d2 != 0) {
            uint pos = atomic_fetch_add_explicit(&scatter_pos[w_base + d2], 1, memory_order_relaxed);
            sorted_idx[idx_base + pos] = (n + gid) | (neg2 ? 0x80000000u : 0u);
        }
    }
    // Carry overflow into extra window (always positive, no sign bit)
    uint carry_base = n_data_windows * n_buckets_val;
    uint carry_idx_base = n_data_windows * n2;
    if (carry1) {
        uint pos = atomic_fetch_add_explicit(&scatter_pos[carry_base + 1], 1, memory_order_relaxed);
        sorted_idx[carry_idx_base + pos] = gid;
    }
    if (carry2) {
        uint pos = atomic_fetch_add_explicit(&scatter_pos[carry_base + 1], 1, memory_order_relaxed);
        sorted_idx[carry_idx_base + pos] = n + gid;
    }
}

// GPU-side count-sorted mapping (CSM) kernel.
// Eliminates the CPU synchronization point between Phase 1+2 and Phase 3+4 by computing
// CSM on GPU. Also performs imbalance detection and large bucket identification.
//
// Each threadgroup handles one window. Uses counting sort on bucket counts to produce
// a mapping where adjacent threads in the reduce kernel have similar workload (SIMD efficiency).
//
// Shared memory layout:
//   hist[0..MAX_HIST-1]: histogram of bucket counts (hist[c] = # buckets with count c)
//   prefix[0..MAX_HIST-1]: prefix sum positions for scatter
//   max_count: max bucket count in this window
//
// The histogram size (MAX_HIST = 4096) limits the max bucket count. For well-behaved
// distributions this is always sufficient. Pathological cases are detected by the
// imbalance check and fall back to CPU.

kernel void msm_compute_csm(
    device const uint* all_counts [[buffer(0)]],
    device uint* count_sorted_map [[buffer(1)]],
    device uint* large_bucket_ids [[buffer(2)]],
    device atomic_uint* n_large_buckets_out [[buffer(3)]],
    device atomic_uint* imbalance_flag [[buffer(4)]],
    constant uint& n_buckets_val [[buffer(5)]],
    constant uint& n_windows_val [[buffer(6)]],
    constant uint& n2_val [[buffer(7)]],
    constant uint& full_data_windows_val [[buffer(8)]],
    constant uint& large_threshold_val [[buffer(9)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint window = tg_idx;
    if (window >= n_windows_val) return;

    uint w_off = window * n_buckets_val;

    // Shared memory for counting sort (in-place: hist is reused for prefix sums)
    constexpr uint MAX_HIST = 4096;
    threadgroup uint hist[MAX_HIST]; // reused as prefix array after prefix sum
    threadgroup atomic_uint tg_max_count;

    // Zero histogram
    for (uint i = tid; i < MAX_HIST; i += tg_size) {
        hist[i] = 0;
    }
    if (tid == 0) {
        atomic_store_explicit(&tg_max_count, 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Build histogram and find max count
    for (uint b = tid; b < n_buckets_val; b += tg_size) {
        uint c = all_counts[w_off + b];
        uint c_clamped = min(c, MAX_HIST - 1);
        atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[c_clamped], 1, memory_order_relaxed);
        atomic_fetch_max_explicit(&tg_max_count, c, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint mc = atomic_load_explicit(&tg_max_count, memory_order_relaxed);

    // Imbalance check: if max bucket count exceeds threshold, signal failure
    if (tid == 0) {
        uint n_check_windows = min(full_data_windows_val, n_windows_val > 1 ? n_windows_val - 1 : n_windows_val);
        if (window < n_check_windows && mc > n2_val / 10) {
            atomic_store_explicit(imbalance_flag, 1, memory_order_relaxed);
        }
        if (window < full_data_windows_val && mc > n2_val / 4) {
            atomic_store_explicit(imbalance_flag, 1, memory_order_relaxed);
        }
    }

    // Compute prefix sum in-place (descending: highest count first)
    // Overwrites hist[] with prefix sums — hist values are no longer needed after this.
    uint mc_clamped = min(mc, MAX_HIST - 1);
    if (tid == 0) {
        uint running = 0;
        for (uint c = mc_clamped + 1; c > 0; c--) {
            uint h = hist[c - 1];
            hist[c - 1] = running;
            running += h;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter into CSM and detect large buckets
    for (uint b = tid; b < n_buckets_val; b += tg_size) {
        uint c = all_counts[w_off + b];
        uint c_clamped = min(c, MAX_HIST - 1);
        uint pos = atomic_fetch_add_explicit((threadgroup atomic_uint*)&hist[c_clamped], 1, memory_order_relaxed);
        count_sorted_map[w_off + pos] = w_off + b;

        // Detect large buckets (skip bucket 0 which is always empty)
        if (b > 0 && c > large_threshold_val) {
            uint idx = atomic_fetch_add_explicit(n_large_buckets_out, 1, memory_order_relaxed);
            large_bucket_ids[idx] = w_off + b;
        }
    }
}


