// BLS12-377 GLV endomorphism kernels for MSM
// Scalar decomposition: k → (k1, k2) where k ≡ k1 + k2·λ (mod r)
// Endomorphism: φ(x,y) = (β·x, y)
//
// Lattice: v1 = (a1, 1), v2 = (-1, a1-1) where a1 = r - λ
// Decomposition: c1 = floor(k·(a1-1)/r), k1 = k - c1·a1, k2 = c1 (neg flag set)

#include "../fields/bls12377_fq.metal"

// Fr377 scalar field order r (4×64-bit LE)
constant ulong FR377_ORDER[4] = {
    0x0a11800000000001uL, 0x59aa76fed0000001uL,
    0x60b44d1e5c37b001uL, 0x12ab655e9a2ca556uL
};

// a1 = 91893752504881257701523279626832445441 (2×64-bit LE)
constant ulong GLV377_A1[2] = {
    0x0a11800000000001uL, 0x452217cc90000001uL
};

// a1 - 1 (2×64-bit LE)
constant ulong GLV377_A1M1[2] = {
    0x0a11800000000000uL, 0x452217cc90000001uL
};

// half_r = (r+1)/2 (4×64-bit LE)
constant ulong FR377_HALF[4] = {
    0x0508c00000000001uL, 0x2cd53b7f68000000uL,
    0xb05a268f2e1bd801uL, 0x0955b2af4d1652abuL
};

// --- 256-bit helpers ---

bool u256_gte_377(thread const ulong* a, constant ulong* b) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

void u256_sub_377(thread ulong* r, thread const ulong* a, thread const ulong* b, thread bool &borrow) {
    ulong br = 0;
    for (int i = 0; i < 4; i++) {
        ulong diff = a[i] - b[i];
        ulong diff2 = diff - br;
        br = ((a[i] < b[i]) || (br && diff == 0)) ? 1uL : 0uL;
        r[i] = diff2;
    }
    borrow = br != 0;
}

void u256_sub_const_377(thread ulong* r, thread const ulong* a, constant ulong* b, thread bool &borrow) {
    ulong br = 0;
    for (int i = 0; i < 4; i++) {
        ulong diff = a[i] - b[i];
        ulong diff2 = diff - br;
        br = ((a[i] < b[i]) || (br && diff == 0)) ? 1uL : 0uL;
        r[i] = diff2;
    }
    borrow = br != 0;
}

void u256_add_const_377(thread ulong* r, thread const ulong* a, constant ulong* b, thread bool &carry) {
    ulong c = 0;
    for (int i = 0; i < 4; i++) {
        ulong s = a[i] + b[i];
        ulong t = s + c;
        r[i] = t;
        c = (s < a[i] || t < s) ? 1uL : 0uL;
    }
    carry = c != 0;
}

void u256_sub_from_const_377(thread ulong* r, constant ulong* a, thread const ulong* b, thread bool &borrow) {
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
// This approximates floor(k*a / r) since r ≈ 2^253 ≈ 2^256/8
// We compute the full 384-bit product and extract [limbs 4,5] (the top 128 bits above 2^256)
void mul256x128_377(thread const ulong* k, constant ulong* a,
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
    // prod is 384 bits. The quotient c1 = prod / r ≈ prod >> 253
    // Since prod = k * (a1-1) and we want floor(prod / r):
    // Approximate: c1 ≈ prod[4..5] >> (253-256) = prod[4..5] << 3
    // But this is only approximate. For exact division, we need more work.
    //
    // Better approximation: since r = a1^2 - a1 + 1 and a1 ≈ 2^127,
    // r ≈ 2^254 - 2^127 + 1. The quotient prod/r ≈ prod >> 253 * (1 + 2^-127).
    //
    // For GPU speed, we use the Barrett-like approach:
    // c1 = (prod >> 251) (shift right by 251 bits)
    // Then adjust: if k - c1*a1 >= r, c1++; if k - c1*a1 < 0, c1--
    //
    // prod >> 251: shift right by 251 = 3*64 + 59, so take prod[3..5] >> 59
    c1_lo = (prod[3] >> 59) | (prod[4] << 5);
    c1_hi = (prod[4] >> 59) | (prod[5] << 5);
}

// 128×128 multiply → 256-bit result
void mul128x128_377(ulong a0, ulong a1, constant ulong* b, thread ulong* r) {
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

kernel void glv377_decompose(
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
    while (u256_gte_377(kr, FR377_ORDER)) {
        ulong tmp[4];
        u256_sub_const_377(tmp, kr, FR377_ORDER, borrow);
        for (int i = 0; i < 4; i++) kr[i] = tmp[i];
    }

    // c1 = approx floor(k * (a1-1) / r)
    ulong c1_lo, c1_hi;
    mul256x128_377(kr, GLV377_A1M1, c1_lo, c1_hi);

    // k1 = k - c1 * a1
    ulong c1a1[4];
    mul128x128_377(c1_lo, c1_hi, GLV377_A1, c1a1);

    ulong k1[4];
    u256_sub_377(k1, kr, c1a1, borrow);
    if (borrow) {
        // k - c1*a1 < 0: c1 was too big, adjust
        // c1--, k1 += a1
        if (c1_lo == 0) c1_hi--;
        c1_lo--;
        bool carry;
        u256_add_const_377(k1, k1, FR377_ORDER, carry);
    }

    // Check if k1 >= r (c1 was too small)
    while (u256_gte_377(k1, FR377_ORDER)) {
        ulong tmp[4];
        u256_sub_const_377(tmp, k1, FR377_ORDER, borrow);
        for (int i = 0; i < 4; i++) k1[i] = tmp[i];
        c1_lo++; if (c1_lo == 0) c1_hi++;
    }

    // k2 = c1 (stored as positive; neg2 flag indicates it's actually -c1)
    ulong k2[4] = {c1_lo, c1_hi, 0, 0};

    // Reduce to half-width: if k1 > half_r, negate
    bool neg1 = false;
    if (u256_gte_377(k1, FR377_HALF)) {
        u256_sub_from_const_377(k1, FR377_ORDER, k1, borrow);
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

kernel void glv377_endomorphism(
    device Point377Affine* points [[buffer(0)]],
    const device uchar* neg1_flags [[buffer(1)]],
    const device uchar* neg2_flags [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    Point377Affine p = points[gid];

    // Apply neg1: negate P if needed
    if (neg1_flags[gid]) {
        p.y = fq377_neg(p.y);
        points[gid] = p;
        // Reload for endomorphism computation
        p.y = fq377_neg(p.y);
    }

    // β in Montgomery form (cube root of unity in Fq377)
    // β = 258664426012969093929703085429980814127835149614277183275038967946009968870203535512256352201271898244626862047231
    // Stored as fq377_to_mont(β)
    Fq377 beta;
    beta.v[0]  = 0xffffff68; beta.v[1]  = 0x02cdffff;
    beta.v[2]  = 0x7fffffb1; beta.v[3]  = 0x51409f83;
    beta.v[4]  = 0x8a7d3ff2; beta.v[5]  = 0x9f7db3a9;
    beta.v[6]  = 0x6e7c6305; beta.v[7]  = 0x7b4e97b7;
    beta.v[8]  = 0x803c84e8; beta.v[9]  = 0x4cf495bf;
    beta.v[10] = 0xe2fdf49a; beta.v[11] = 0x008d6661;

    // That's actually R mod q (= 1 in Montgomery form). Need actual beta.
    // β in standard form: [0xffffffffffffffff, 0xd1e945779fffffff, 0x59064ee822fb5bff,
    //                      0xb8882a75cc9bc8e3, 0xbc8756ba8f8c524e, 0x01ae3a4617c510ea]
    // Convert to Montgomery: beta_mont = fq377_to_mont(beta_raw)
    // Must be precomputed. Placeholder - will be filled by CPU init.

    // Actually, compute β in Montgomery form from raw value
    Fq377 beta_raw;
    beta_raw.v[0]  = 0xffffffff; beta_raw.v[1]  = 0xffffffff;
    beta_raw.v[2]  = 0x9fffffff; beta_raw.v[3]  = 0xd1e94577;
    beta_raw.v[4]  = 0x22fb5bff; beta_raw.v[5]  = 0x59064ee8;
    beta_raw.v[6]  = 0xcc9bc8e3; beta_raw.v[7]  = 0xb8882a75;
    beta_raw.v[8]  = 0x8f8c524e; beta_raw.v[9]  = 0xbc8756ba;
    beta_raw.v[10] = 0x17c510ea; beta_raw.v[11] = 0x01ae3a46;
    Fq377 beta_mont = fq377_to_mont(beta_raw);

    Point377Affine endo;
    endo.x = fq377_mul(beta_mont, p.x);

    if (neg2_flags[gid]) {
        endo.y = fq377_neg(p.y);
    } else {
        endo.y = p.y;
    }

    points[n + gid] = endo;
}
