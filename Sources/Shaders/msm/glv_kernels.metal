// GLV endomorphism kernels for BN254 MSM
// Scalar decomposition: k -> (k1, k2) where k = k1 + k2*lambda (mod r)
// Endomorphism: phi(P) = (beta*x, y)

#include "../fields/bn254_fp.metal"

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

// --- 256-bit helpers (ulong limbs) ---

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

// --- Wide multiply helpers ---

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

void mul64x128_gpu(ulong a, ulong b0, ulong b1, thread ulong &r0, thread ulong &r1, thread ulong &r2) {
    ulong h0 = mulhi(a, b0), l0 = a * b0;
    ulong h1 = mulhi(a, b1), l1 = a * b1;
    r0 = l0;
    ulong s1 = l1 + h0;
    r1 = s1;
    r2 = h1 + ((s1 < l1) ? 1uL : 0uL);
}

void mul128x64_gpu(ulong a0, ulong a1, ulong b, thread ulong &r0, thread ulong &r1, thread ulong &r2) {
    ulong h0 = mulhi(a0, b), l0 = a0 * b;
    ulong h1 = mulhi(a1, b), l1 = a1 * b;
    r0 = l0;
    ulong s1 = l1 + h0;
    r1 = s1;
    r2 = h1 + ((s1 < l1) ? 1uL : 0uL);
}

// --- GLV Decomposition Kernel ---

kernel void glv_decompose(
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

    bool borrow;
    while (u256_gte_const(kr, FR_ORDER)) {
        ulong tmp[4];
        u256_sub_const(tmp, kr, FR_ORDER, borrow);
        for (int i = 0; i < 4; i++) kr[i] = tmp[i];
    }

    ulong c1_lo, c1_hi;
    mul256x192(kr, GLV_G1_0, GLV_G1_1, GLV_G1_2, c1_lo, c1_hi);
    ulong c2 = mul256x128(kr, GLV_G2_0, GLV_G2_1);

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

    bool neg1 = false;
    if (u256_gte_const(k1, HALF_R)) {
        u256_sub_from_const(k1, FR_ORDER, k1, borrow);
        neg1 = true;
    }

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

kernel void glv_endomorphism(
    device PointAffine* points [[buffer(0)]],
    const device uchar* neg1_flags [[buffer(1)]],
    const device uchar* neg2_flags [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    PointAffine p = points[gid];

    if (neg1_flags[gid]) {
        Fp py = fp_modulus();
        uint borrow;
        p.y = fp_sub_raw(py, p.y, borrow);
        points[gid] = p;
        p.y = fp_sub_raw(fp_modulus(), p.y, borrow);
    }

    // beta: cube root of unity in Fp (Montgomery form)
    Fp beta;
    beta.v[0] = 0xd782e155u; beta.v[1] = 0x71930c11u;
    beta.v[2] = 0xffbe3323u; beta.v[3] = 0xa6bb947cu;
    beta.v[4] = 0xd4741444u; beta.v[5] = 0xaa303344u;
    beta.v[6] = 0x26594943u; beta.v[7] = 0x2c3b3f0du;

    PointAffine endo;
    endo.x = fp_mul_karatsuba(beta, p.x);

    if (neg2_flags[gid]) {
        uint borrow;
        endo.y = fp_sub_raw(fp_modulus(), p.y, borrow);
    } else {
        endo.y = p.y;
    }

    points[n + gid] = endo;
}
