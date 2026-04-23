// BLS12-381 GLV endomorphism kernels for MSM
// Scalar decomposition: k → (k1, k2) where k ≡ k1 + k2·λ (mod r)
// Endomorphism: φ(x,y) = (β·x, y)
//
// Lattice: v1 = (a1, 1), v2 = (-1, a1-1) where a1 = floor((r-1)/λ)
// Decomposition: c1 = floor(k·(a1-1)/r), k1 = k - c1·a1, k2 = c1

#include "../geometry/bls12381_curve.metal"

// Fr381 scalar field order r (4×64-bit LE)
constant ulong FR381_ORDER[4] = {
    0x0000000d8ed0000uL, 0x0000000d8db0000uL,
    0x00000000fc420euL, 0x73eda753299d7d48uL
};

// a1 = floor((r-1)/λ) (2×64-bit LE)
constant ulong GLV381_A1[2] = {
    0x21d42c0d00200001uL, 0x0000000000000008uL
};

// a1 - 1 (2×64-bit LE)
constant ulong GLV381_A1M1[2] = {
    0x21d42c0d00200000uL, 0x0000000000000008uL
};

// half_r = (r+1)/2 (4×64-bit LE)
constant ulong FR381_HALF[4] = {
    0x80000006e4768000uL, 0x00000006c6ed8000uL,
    0x000000007e210fuL, 0x39f6d3a99ffffe24uL
};

// --- 256-bit helpers ---

bool u256_gte_381(thread const ulong* a, constant ulong* b) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

void u256_sub_381(thread ulong* r, thread const ulong* a, thread const ulong* b, thread bool &borrow) {
    ulong br = 0;
    for (int i = 0; i < 4; i++) {
        ulong diff = a[i] - b[i];
        ulong diff2 = diff - br;
        br = ((a[i] < b[i]) || (br && diff == 0)) ? 1uL : 0uL;
        r[i] = diff2;
    }
    borrow = br != 0;
}

void u256_sub_const_381(thread ulong* r, thread const ulong* a, constant ulong* b, thread bool &borrow) {
    ulong br = 0;
    for (int i = 0; i < 4; i++) {
        ulong diff = a[i] - b[i];
        ulong diff2 = diff - br;
        br = ((a[i] < b[i]) || (br && diff == 0)) ? 1uL : 0uL;
        r[i] = diff2;
    }
    borrow = br != 0;
}

void u256_add_const_381(thread ulong* r, thread const ulong* a, constant ulong* b, thread bool &carry) {
    ulong c = 0;
    for (int i = 0; i < 4; i++) {
        ulong s = a[i] + b[i];
        ulong t = s + c;
        r[i] = t;
        c = (s < a[i] || t < s) ? 1uL : 0uL;
    }
    carry = c != 0;
}

void u256_sub_from_const_381(thread ulong* r, constant ulong* a, thread const ulong* b, thread bool &borrow) {
    ulong br = 0;
    for (int i = 0; i < 4; i++) {
        ulong diff = a[i] - b[i];
        ulong diff2 = diff - br;
        br = ((a[i] < b[i]) || (br && diff == 0)) ? 1uL : 0uL;
        r[i] = diff2;
    }
    borrow = br != 0;
}

// Multiply 256-bit k by 128-bit a, return high 128 bits of (k*a) / 2^256
void mul256x128_381(thread const ulong* k, constant ulong* a,
                    thread ulong &c1_lo, thread ulong &c1_hi) {
    ulong prod[6] = {0,0,0,0,0,0};
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 2; j++) {
            ulong hi = mulhi(k[i], a[j]);
            ulong lo = k[i] * a[j];
            ulong s1 = prod[i+j] + lo;
            ulong c1 = (s1 < prod[i+j]) ? 1uL : 0uL;
            ulong s2 = s1 + carry;
            ulong c2 = (s2 < s1) ? 1uL : 0uL;
            prod[i+j] = s2;
            carry = hi + c1 + c2;
        }
        prod[i+2] += carry;
    }
    // Approximate: c1 ≈ prod >> 253
    c1_lo = (prod[3] >> 59) | (prod[4] << 5);
    c1_hi = (prod[4] >> 59) | (prod[5] << 5);
}

// 128×128 multiply → 256-bit result
void mul128x128_381(ulong a0, ulong a1, constant ulong* b, thread ulong* r) {
    ulong h00 = mulhi(a0, b[0]), l00 = a0 * b[0];
    ulong h01 = mulhi(a0, b[1]), l01 = a0 * b[1];
    ulong h10 = mulhi(a1, b[0]), l10 = a1 * b[0];
    ulong h11 = mulhi(a1, b[1]), l11 = a1 * b[1];

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

// --- GLV Decomposition Kernel ---

kernel void glv381_decompose(
    const device uint* scalars_in [[buffer(0)]],
    device uint* k1_out [[buffer(1)]],
    device uint* k2_out [[buffer(2)]],
    device uchar* neg1_out [[buffer(3)]],
    device uchar* neg2_out [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    const device uint* sp = scalars_in + gid * 8;
    ulong kr[4] = {
        ulong(sp[0]) | (ulong(sp[1]) << 32),
        ulong(sp[2]) | (ulong(sp[3]) << 32),
        ulong(sp[4]) | (ulong(sp[5]) << 32),
        ulong(sp[6]) | (ulong(sp[7]) << 32)
    };

    // Reduce k mod r
    bool borrow;
    while (u256_gte_381(kr, FR381_ORDER)) {
        ulong tmp[4];
        u256_sub_const_381(tmp, kr, FR381_ORDER, borrow);
        for (int i = 0; i < 4; i++) kr[i] = tmp[i];
    }

    // c1 = approx floor(k * (a1-1) / r)
    ulong c1_lo, c1_hi;
    mul256x128_381(kr, GLV381_A1M1, c1_lo, c1_hi);

    // k1 = k - c1 * a1
    ulong c1a1[4];
    mul128x128_381(c1_lo, c1_hi, GLV381_A1, c1a1);

    ulong k1[4];
    u256_sub_381(k1, kr, c1a1, borrow);
    if (borrow) {
        // k - c1*a1 < 0: c1 was too big, adjust
        if (c1_lo == 0) c1_hi--;
        c1_lo--;
        bool carry;
        u256_add_const_381(k1, k1, FR381_ORDER, carry);
    }

    // Check if k1 >= r (c1 was too small)
    while (u256_gte_381(k1, FR381_ORDER)) {
        ulong tmp[4];
        u256_sub_const_381(tmp, k1, FR381_ORDER, borrow);
        for (int i = 0; i < 4; i++) k1[i] = tmp[i];
        c1_lo++; if (c1_lo == 0) c1_hi++;
    }

    // k2 = c1
    ulong k2[4] = {c1_lo, c1_hi, 0, 0};

    // Reduce to half-width: if k1 > half_r, negate
    bool neg1 = false;
    if (u256_gte_381(k1, FR381_HALF)) {
        u256_sub_from_const_381(k1, FR381_ORDER, k1, borrow);
        neg1 = true;
    }

    // k2 is always <= ~sqrt(r) ≈ a1, no need to check half_r
    bool neg2 = (c1_lo != 0 || c1_hi != 0);

    device uint* k1p = k1_out + gid * 8;
    device uint* k2p = k2_out + gid * 8;
    for (int i = 0; i < 4; i++) {
        k1p[i*2] = uint(k1[i] & 0xFFFFFFFF);
        k1p[i*2+1] = uint(k1[i] >> 32);
        k2p[i*2] = uint(k2[i] & 0xFFFFFFFF);
        k2p[i*2+1] = uint(k2[i] >> 32);
    }
    neg1_out[gid] = neg1 ? 1 : 0;
    neg2_out[gid] = neg2 ? 1 : 0;
}

// --- GLV Endomorphism Kernel ---
// Apply φ(P) = (β·x, y) and handle negation flags
constant Fp381 GLV381_BETA_MONT = {
    { 0xce84ea86, 0x5b623b5c, 0x5eead1ad, 0x8f13871d,
      0x83ae36ad, 0x7c2c93a8, 0x11e09e86, 0x4e9c404a,
      0x24e0d0b3, 0x5c671f3c, 0x2f5be8c0, 0x05c671f3 }
};

kernel void glv381_endomorphism(
    device Point381Affine* points [[buffer(0)]],
    const device uchar* neg1_flags [[buffer(1)]],
    const device uchar* neg2_flags [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    Point381Affine p = points[gid];

    // Apply neg1: negate P if needed
    if (neg1_flags[gid]) {
        p.y = fp381_neg(p.y);
    }

    // Compute endomorphism: φ(P) = (β·x, y)
    Point381Affine endo;
    endo.x = fp381_mul(GLV381_BETA_MONT, p.x);

    if (neg2_flags[gid]) {
        endo.y = fp381_neg(p.y);
    } else {
        endo.y = p.y;
    }

    points[n + gid] = endo;
}

// --- Combined Copy + Endomorphism Kernel ---
kernel void glv381_copy_and_endo(
    const device Point381Affine* src_points [[buffer(0)]],
    device Point381Affine* dst_points [[buffer(1)]],
    const device uchar* neg1_flags [[buffer(2)]],
    const device uchar* neg2_flags [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    Point381Affine p = src_points[gid];

    // Apply neg1: negate P if needed
    if (neg1_flags[gid]) {
        p.y = fp381_neg(p.y);
    }

    // Write original point (with neg1 applied) to first half
    dst_points[gid] = p;

    // Apply endomorphism: φ(P) = (β·x, y)
    Point381Affine endo;
    endo.x = fp381_mul(GLV381_BETA_MONT, p.x);

    if (neg2_flags[gid]) {
        endo.y = fp381_neg(p.y);
    } else {
        endo.y = p.y;
    }

    // Write endomorphized point to second half
    dst_points[n + gid] = endo;
}