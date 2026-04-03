// secp256k1 GLV endomorphism kernels for MSM
// Scalar decomposition: k -> (k1, k2) where k = k1 + k2*lambda (mod n)
// Endomorphism: phi(P) = (beta*x, y) where beta^3 = 1 (mod p)

#include "../fields/secp256k1_fp.metal"

// secp256k1 scalar field order n
constant ulong SECP_N[4] = {
    0xBFD25E8CD0364141uL, 0xBAAEDCE6AF48A03BuL,
    0xFFFFFFFFFFFFFFFEuL, 0xFFFFFFFFFFFFFFFFuL
};

constant ulong SECP_HALF_N[4] = {
    (0xBFD25E8CD0364141uL >> 1) | (0xBAAEDCE6AF48A03BuL << 63),
    (0xBAAEDCE6AF48A03BuL >> 1) | (0xFFFFFFFFFFFFFFFEuL << 63),
    (0xFFFFFFFFFFFFFFFEuL >> 1) | (0xFFFFFFFFFFFFFFFFuL << 63),
    0xFFFFFFFFFFFFFFFFuL >> 1
};

// Babai rounding constants: g_i = floor(2^384 * basis_element / n)
constant ulong SECP_GLV_G1[4] = {
    0xe893209a45dbb031uL, 0x3daa8a1471e8ca7fuL,
    0xe86c90e49284eb15uL, 0x3086d221a7d46bcduL
};

constant ulong SECP_GLV_G2[4] = {
    0x1571b4ae8ac47f71uL, 0x221208ac9df506c6uL,
    0x6f547fa90abfe4c4uL, 0xe4437ed6010e8828uL
};

// Lattice basis vectors (short form, ~128 bits)
constant ulong SECP_GLV_A1[2] = {
    0xe86c90e49284eb15uL, 0x3086d221a7d46bcduL
};

constant ulong SECP_GLV_MINUS_B1[2] = {
    0x6f547fa90abfe4c3uL, 0xe4437ed6010e8828uL
};

// a2 is 129 bits: 0x114ca50f7a8e2f3f657c1108d9d44cfd8
constant ulong SECP_GLV_A2[3] = {
    0x57c1108d9d44cfd8uL, 0x14ca50f7a8e2f3f6uL, 0x1uL
};

// b2 = a1
constant ulong SECP_GLV_B2[2] = {
    0xe86c90e49284eb15uL, 0x3086d221a7d46bcduL
};

// --- 256-bit helpers ---

void secp_u256_add(thread ulong* r, thread const ulong* a, constant ulong* b, thread bool &carry) {
    ulong c = 0;
    for (int i = 0; i < 4; i++) {
        ulong s = a[i] + b[i];
        ulong t = s + c;
        r[i] = t;
        c = (s < a[i] || t < s) ? 1uL : 0uL;
    }
    carry = c != 0;
}

void secp_u256_sub(thread ulong* r, thread const ulong* a, thread const ulong* b, thread bool &borrow) {
    ulong br = 0;
    for (int i = 0; i < 4; i++) {
        ulong diff = a[i] - b[i];
        ulong diff2 = diff - br;
        br = ((a[i] < b[i]) || (br && diff == 0)) ? 1uL : 0uL;
        r[i] = diff2;
    }
    borrow = br != 0;
}

void secp_u256_sub_const(thread ulong* r, thread const ulong* a, constant ulong* b, thread bool &borrow) {
    ulong br = 0;
    for (int i = 0; i < 4; i++) {
        ulong diff = a[i] - b[i];
        ulong diff2 = diff - br;
        br = ((a[i] < b[i]) || (br && diff == 0)) ? 1uL : 0uL;
        r[i] = diff2;
    }
    borrow = br != 0;
}

void secp_u256_sub_from_const(thread ulong* r, constant ulong* a, thread const ulong* b, thread bool &borrow) {
    ulong br = 0;
    for (int i = 0; i < 4; i++) {
        ulong diff = a[i] - b[i];
        ulong diff2 = diff - br;
        br = ((a[i] < b[i]) || (br && diff == 0)) ? 1uL : 0uL;
        r[i] = diff2;
    }
    borrow = br != 0;
}

bool secp_u256_gte_const(thread const ulong* a, constant ulong* b) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

// --- Wide multiply: 256 x 256 -> top 128 bits (limbs [6,7] of 512-bit product) ---
// Returns c = floor(k * g / 2^384)

void secp_mul256x256_top128(thread const ulong* k, constant ulong* g,
                             thread ulong &out_lo, thread ulong &out_hi) {
    // Full 4x4 schoolbook multiply, but we only need limbs [6] and [7].
    // We still must compute lower limbs for carries.
    ulong prod[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
            ulong hi = mulhi(k[i], g[j]);
            ulong lo = k[i] * g[j];
            ulong s1 = prod[i+j] + lo;
            ulong c1 = (s1 < prod[i+j]) ? 1uL : 0uL;
            ulong s2 = s1 + carry;
            ulong c2 = (s2 < s1) ? 1uL : 0uL;
            prod[i+j] = s2;
            carry = hi + c1 + c2;
        }
        if (i + 4 < 8) prod[i+4] += carry;
    }
    out_lo = prod[6];
    out_hi = prod[7];
}

// 128 x 128 -> 256-bit result
void secp_mul128x128(ulong a0, ulong a1, constant ulong* b, thread ulong* r) {
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

// 128 x 192 (a2 is 129 bits but stored as 3 limbs) -> 320-bit result (5 limbs)
void secp_mul128x192(ulong c0, ulong c1, constant ulong* a2, thread ulong* r) {
    // c (2 limbs) x a2 (3 limbs) -> 5 limbs
    for (int i = 0; i < 5; i++) r[i] = 0;
    ulong cv[2] = {c0, c1};
    for (int i = 0; i < 2; i++) {
        ulong carry = 0;
        for (int j = 0; j < 3; j++) {
            ulong hi = mulhi(cv[i], a2[j]);
            ulong lo = cv[i] * a2[j];
            ulong s1 = r[i+j] + lo;
            ulong cc1 = (s1 < r[i+j]) ? 1uL : 0uL;
            ulong s2 = s1 + carry;
            ulong cc2 = (s2 < s1) ? 1uL : 0uL;
            r[i+j] = s2;
            carry = hi + cc1 + cc2;
        }
        r[i+3] += carry;
    }
}

// --- GLV Decomposition Kernel ---

kernel void secp_glv_decompose(
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

    // Reduce mod n
    bool borrow;
    while (secp_u256_gte_const(kr, SECP_N)) {
        ulong tmp[4];
        secp_u256_sub_const(tmp, kr, SECP_N, borrow);
        for (int i = 0; i < 4; i++) kr[i] = tmp[i];
    }

    // c1 = floor(k * g1 / 2^384), c2 = floor(k * g2 / 2^384)
    ulong c1_lo, c1_hi, c2_lo, c2_hi;
    secp_mul256x256_top128(kr, SECP_GLV_G1, c1_lo, c1_hi);
    secp_mul256x256_top128(kr, SECP_GLV_G2, c2_lo, c2_hi);

    // k1 = k - c1*a1 - c2*a2 (mod n)
    // c2*a2 can be up to 257 bits, so we compute c1*a1 + c2*a2 first,
    // then subtract from k, handling carries properly.
    ulong c1a1[4];
    secp_mul128x128(c1_lo, c1_hi, SECP_GLV_A1, c1a1);

    ulong c2a2[5];
    secp_mul128x192(c2_lo, c2_hi, SECP_GLV_A2, c2a2);

    // sum = c1*a1 + c2*a2 (up to 258 bits, but close to k by design)
    ulong sum[5] = {0, 0, 0, 0, 0};
    ulong carry_val = 0;
    for (int i = 0; i < 4; i++) {
        ulong s1 = c1a1[i] + c2a2[i];
        ulong c1 = (s1 < c1a1[i]) ? 1uL : 0uL;
        ulong s2 = s1 + carry_val;
        ulong c2 = (s2 < s1) ? 1uL : 0uL;
        sum[i] = s2;
        carry_val = c1 + c2;
    }
    sum[4] = c2a2[4] + carry_val;

    // k1 = k - sum (mod n). k is 256 bits, sum is ~256 bits.
    // If sum > k, result wraps and we add n.
    ulong k1[4];
    ulong br = 0;
    for (int i = 0; i < 4; i++) {
        ulong diff = kr[i] - sum[i];
        ulong diff2 = diff - br;
        br = ((kr[i] < sum[i]) || (br && diff == 0)) ? 1uL : 0uL;
        k1[i] = diff2;
    }
    // Account for sum[4] overflow and any remaining borrow
    if (br || sum[4]) {
        // We need to add n back (possibly multiple times)
        // Since sum ≈ k, the difference is at most ~128 bits + a few n's
        bool carry_unused;
        secp_u256_add(k1, k1, SECP_N, carry_unused);
        // In rare cases with sum[4] > 0, may need another addition
        if (sum[4] > 1) {
            secp_u256_add(k1, k1, SECP_N, carry_unused);
        }
    }

    // k2 = c1*minus_b1 - c2*b2
    ulong c1mb1[4];
    secp_mul128x128(c1_lo, c1_hi, SECP_GLV_MINUS_B1, c1mb1);

    ulong c2b2[4];
    secp_mul128x128(c2_lo, c2_hi, SECP_GLV_B2, c2b2);

    ulong k2[4];
    secp_u256_sub(k2, c1mb1, c2b2, borrow);
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

    // Centering: if k1 > n/2, negate
    bool neg1 = false;
    if (secp_u256_gte_const(k1, SECP_HALF_N)) {
        secp_u256_sub_from_const(k1, SECP_N, k1, borrow);
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
// For each point i in [0, n):
//   points[i]: apply neg1 (negate y if needed)
//   points[n+i] = (beta*x, ±y): endomorphism point with neg2

kernel void secp_glv_endomorphism(
    device SecpPointAffine* points [[buffer(0)]],
    const device uchar* neg1_flags [[buffer(1)]],
    const device uchar* neg2_flags [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    SecpPointAffine p = points[gid];

    // Apply neg1 to original point
    if (neg1_flags[gid]) {
        SecpFp py;
        for (int i = 0; i < 8; i++) py.v[i] = SECP_P[i];
        uint borrow;
        p.y = secp_sub_raw(py, p.y, borrow);
        points[gid] = p;
        // Restore p.y for endomorphism computation below
        SecpFp py2;
        for (int i = 0; i < 8; i++) py2.v[i] = SECP_P[i];
        p.y = secp_sub_raw(py2, p.y, borrow);
    }

    // beta: cube root of unity in Fp (Montgomery form)
    SecpFp beta;
    beta.v[0] = 0x8e81894eu; beta.v[1] = 0x58a4361cu;
    beta.v[2] = 0x1c4b80afu; beta.v[3] = 0x03fde163u;
    beta.v[4] = 0xd02e3905u; beta.v[5] = 0xf8e98978u;
    beta.v[6] = 0xbcbb3d53u; beta.v[7] = 0x7a4a36aeu;

    SecpPointAffine endo;
    endo.x = secp_mul(beta, p.x);

    if (neg2_flags[gid]) {
        uint borrow;
        SecpFp py;
        for (int i = 0; i < 8; i++) py.v[i] = SECP_P[i];
        endo.y = secp_sub_raw(py, p.y, borrow);
    } else {
        endo.y = p.y;
    }

    points[n + gid] = endo;
}
