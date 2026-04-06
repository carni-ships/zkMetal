// XHash12 GPU kernel for Goldilocks field
// 12-element state, rate=8, capacity=4, S-box x^7
// Permutation: (FB)(E)(FB)(E)(FB)(E)(M) — 7 logical rounds
// Each thread computes one independent XHash12 permutation.

#include "../fields/goldilocks.metal"

#define XHASH_STATE 12
#define XHASH_RATE 8

// MDS matrix first row (12x12 circulant)
constant ulong MDS_ROW[12] = {7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8};

// S-box: x -> x^7
Gl xh_sbox(Gl x) {
    Gl x2 = gl_mul(x, x);
    Gl x3 = gl_mul(x2, x);
    Gl x6 = gl_mul(x3, x3);
    return gl_mul(x6, x);
}

// Inverse S-box: x -> x^(1/7)
// Exponent: 10540996611094048183 = 0x9249249249249247
// Uses addition chain from Miden's production code.
Gl xh_inv_sbox(Gl x) {
    Gl t1 = gl_mul(x, x);           // x^2
    Gl t2 = gl_mul(t1, t1);         // x^4

    // t3 = x^(100100) via 3 squarings of t2 then mul t2
    Gl t3 = t2;
    t3 = gl_mul(t3, t3); t3 = gl_mul(t3, t3); t3 = gl_mul(t3, t3);
    t3 = gl_mul(t3, t2);

    // t4 = t3^(2^6) * t3
    Gl t4 = t3;
    for (int i = 0; i < 6; i++) t4 = gl_mul(t4, t4);
    t4 = gl_mul(t4, t3);

    // t5 = t4^(2^12) * t4
    Gl t5 = t4;
    for (int i = 0; i < 12; i++) t5 = gl_mul(t5, t5);
    t5 = gl_mul(t5, t4);

    // t6 = t5^(2^6) * t3
    Gl t6 = t5;
    for (int i = 0; i < 6; i++) t6 = gl_mul(t6, t6);
    t6 = gl_mul(t6, t3);

    // t7 = t6^(2^31) * t6
    Gl t7 = t6;
    for (int i = 0; i < 31; i++) t7 = gl_mul(t7, t7);
    t7 = gl_mul(t7, t6);

    // Final: (t7^2 * t6)^2^2 * (t1 * t2 * x)
    Gl a = gl_mul(t7, t7);
    a = gl_mul(a, t6);
    a = gl_mul(a, a);
    a = gl_mul(a, a);
    Gl b = gl_mul(gl_mul(t1, t2), x);
    return gl_mul(a, b);
}

// MDS matrix multiply (12x12 circulant)
void xh_mds(thread Gl *s) {
    Gl result[XHASH_STATE];
    for (int i = 0; i < XHASH_STATE; i++) {
        Gl acc = gl_zero();
        for (int j = 0; j < XHASH_STATE; j++) {
            int idx = (j + XHASH_STATE - i) % XHASH_STATE;
            acc = gl_add(acc, gl_mul(s[j], Gl{MDS_ROW[idx]}));
        }
        result[i] = acc;
    }
    for (int i = 0; i < XHASH_STATE; i++) s[i] = result[i];
}

// Add round constants
void xh_add_constants(thread Gl *s, constant ulong *c) {
    for (int i = 0; i < XHASH_STATE; i++) {
        s[i] = gl_add(s[i], Gl{c[i]});
    }
}

// Cubic extension field Fp3 = Fp[x]/(x^3 - x - 1)
// Multiply: a * b mod (x^3 - x - 1)
struct Fp3 { Gl a, b, c; };

Fp3 fp3_mul(Fp3 x, Fp3 y) {
    Gl a0b0 = gl_mul(x.a, y.a);
    Gl a1b1 = gl_mul(x.b, y.b);
    Gl a2b2 = gl_mul(x.c, y.c);

    Gl s01a = gl_add(x.a, x.b), s01b = gl_add(y.a, y.b);
    Gl p01 = gl_mul(s01a, s01b);
    Gl s02a = gl_add(x.a, x.c), s02b = gl_add(y.a, y.c);
    Gl p02 = gl_mul(s02a, s02b);
    Gl s12a = gl_add(x.b, x.c), s12b = gl_add(y.b, y.c);
    Gl p12 = gl_mul(s12a, s12b);

    Gl diff = gl_sub(a0b0, a1b1);

    Gl c0 = gl_sub(gl_add(p12, diff), a2b2);
    Gl c1 = gl_sub(gl_sub(gl_add(p01, p12), gl_add(a1b1, a1b1)), a0b0);
    Gl c2 = gl_sub(p02, diff);

    return Fp3{c0, c1, c2};
}

Fp3 fp3_sqr(Fp3 x) {
    Gl a2sq = gl_mul(x.c, x.c);
    Gl a1a2 = gl_mul(x.b, x.c);
    Gl d_a1a2 = gl_add(a1a2, a1a2);

    Gl c0 = gl_add(gl_mul(x.a, x.a), d_a1a2);
    Gl a0a1 = gl_mul(x.a, x.b);
    Gl c1 = gl_add(gl_add(a0a1, a1a2), gl_add(gl_add(a0a1, a1a2), a2sq));
    Gl a0a2 = gl_mul(x.a, x.c);
    Gl c2 = gl_add(gl_add(a0a2, a0a2), gl_add(gl_mul(x.b, x.b), a2sq));

    return Fp3{c0, c1, c2};
}

// x^7 in Fp3: x -> x^2 -> x^3 -> x^6 -> x^7
Fp3 fp3_pow7(Fp3 x) {
    Fp3 x2 = fp3_sqr(x);
    Fp3 x3 = fp3_mul(x2, x);
    Fp3 x6 = fp3_sqr(x3);
    return fp3_mul(x6, x);
}

// FB round: MDS -> ARK1 -> x^7 -> MDS -> ARK2 -> x^(1/7)
void xh_fb_round(thread Gl *s, constant ulong *ark1, constant ulong *ark2) {
    xh_mds(s);
    xh_add_constants(s, ark1);
    for (int i = 0; i < XHASH_STATE; i++) s[i] = xh_sbox(s[i]);

    xh_mds(s);
    xh_add_constants(s, ark2);
    for (int i = 0; i < XHASH_STATE; i++) s[i] = xh_inv_sbox(s[i]);
}

// E round: ARK1 -> Fp3 x^7 on 4 triplets
void xh_ext_round(thread Gl *s, constant ulong *ark1) {
    xh_add_constants(s, ark1);

    for (int t = 0; t < 4; t++) {
        int base = t * 3;
        Fp3 elem = {s[base], s[base+1], s[base+2]};
        Fp3 res = fp3_pow7(elem);
        s[base] = res.a;
        s[base+1] = res.b;
        s[base+2] = res.c;
    }
}

// Full XHash12 permutation
void xh12_permute(thread Gl *s, constant ulong *ark1, constant ulong *ark2) {
    // (FB)(E)(FB)(E)(FB)(E)(M)
    xh_fb_round(s, ark1 + 0*12, ark2 + 0*12);
    xh_ext_round(s, ark1 + 1*12);
    xh_fb_round(s, ark1 + 2*12, ark2 + 2*12);
    xh_ext_round(s, ark1 + 3*12);
    xh_fb_round(s, ark1 + 4*12, ark2 + 4*12);
    xh_ext_round(s, ark1 + 5*12);
    // M round
    xh_mds(s);
    xh_add_constants(s, ark1 + 6*12);
}

// --- GPU Kernels ---

// Batch permutation: each thread processes one 12-element state
kernel void xhash12_batch_permute(
    device ulong *states [[buffer(0)]],
    constant ulong *ark1 [[buffer(1)]],
    constant ulong *ark2 [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint offset = gid * XHASH_STATE;
    Gl s[XHASH_STATE];
    for (int i = 0; i < XHASH_STATE; i++) {
        s[i] = Gl{states[offset + i]};
    }

    xh12_permute(s, ark1, ark2);

    for (int i = 0; i < XHASH_STATE; i++) {
        states[offset + i] = s[i].v;
    }
}

// Batch 2-to-1 hash: each thread hashes (left[i], right[i]) -> digest[i]
// Input: pairs buffer with 2*4 = 8 Goldilocks elements per pair
// Output: digest buffer with 4 elements per hash
kernel void xhash12_batch_hash(
    device const ulong *pairs [[buffer(0)]],
    device ulong *digests [[buffer(1)]],
    constant ulong *ark1 [[buffer(2)]],
    constant ulong *ark2 [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint in_offset = gid * 8;
    Gl s[XHASH_STATE];

    // Load left (4 elements) into rate[0..3]
    for (int i = 0; i < 4; i++) s[i] = Gl{pairs[in_offset + i]};
    // Load right (4 elements) into rate[4..7]
    for (int i = 0; i < 4; i++) s[i+4] = Gl{pairs[in_offset + 4 + i]};
    // Capacity = 0
    for (int i = 8; i < XHASH_STATE; i++) s[i] = gl_zero();

    xh12_permute(s, ark1, ark2);

    uint out_offset = gid * 4;
    for (int i = 0; i < 4; i++) {
        digests[out_offset + i] = s[i].v;
    }
}
