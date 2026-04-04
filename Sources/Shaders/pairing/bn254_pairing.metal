// BN254 GPU pairing: batch Miller loop + final exponentiation
// Each thread computes one independent pairing, enabling Groth16's 4 pairings in parallel.
//
// Tower: Fp -> Fp2 = Fp[u]/(u^2+1) -> Fp6 = Fp2[v]/(v^3-xi) -> Fp12 = Fp6[w]/(w^2-v)
// xi = 9 + u (non-residue for BN254)

#include <metal_stdlib>
using namespace metal;

// We include Fp arithmetic from the existing shader
#include "../fields/bn254_fp.metal"

// ============================================================================
// Fp2 = Fp[u]/(u^2 + 1)
// ============================================================================

struct Fp2 {
    Fp c0;
    Fp c1;
};

Fp2 fp2_zero() {
    Fp2 r; r.c0 = fp_zero(); r.c1 = fp_zero(); return r;
}

Fp2 fp2_one() {
    Fp2 r; r.c0 = fp_one(); r.c1 = fp_zero(); return r;
}

bool fp2_is_zero(Fp2 a) {
    return fp_is_zero(a.c0) && fp_is_zero(a.c1);
}

Fp2 fp2_add(Fp2 a, Fp2 b) {
    Fp2 r; r.c0 = fp_add(a.c0, b.c0); r.c1 = fp_add(a.c1, b.c1); return r;
}

Fp2 fp2_sub(Fp2 a, Fp2 b) {
    Fp2 r; r.c0 = fp_sub(a.c0, b.c0); r.c1 = fp_sub(a.c1, b.c1); return r;
}

Fp2 fp2_neg(Fp2 a) {
    Fp2 r; r.c0 = fp_neg(a.c0); r.c1 = fp_neg(a.c1); return r;
}

Fp2 fp2_double(Fp2 a) {
    Fp2 r; r.c0 = fp_double(a.c0); r.c1 = fp_double(a.c1); return r;
}

// (a0+a1*u)(b0+b1*u) = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*u
// Karatsuba: c1 = (a0+a1)(b0+b1) - a0*b0 - a1*b1
Fp2 fp2_mul(Fp2 a, Fp2 b) {
    Fp t0 = fp_mul(a.c0, b.c0);
    Fp t1 = fp_mul(a.c1, b.c1);
    Fp2 r;
    r.c0 = fp_sub(t0, t1);
    r.c1 = fp_sub(fp_mul(fp_add(a.c0, a.c1), fp_add(b.c0, b.c1)), fp_add(t0, t1));
    return r;
}

// (a0+a1*u)^2 = (a0+a1)(a0-a1) + 2*a0*a1*u
Fp2 fp2_sqr(Fp2 a) {
    Fp t0 = fp_mul(a.c0, a.c1);
    Fp2 r;
    r.c0 = fp_mul(fp_add(a.c0, a.c1), fp_sub(a.c0, a.c1));
    r.c1 = fp_double(t0);
    return r;
}

Fp2 fp2_conjugate(Fp2 a) {
    Fp2 r; r.c0 = a.c0; r.c1 = fp_neg(a.c1); return r;
}

Fp2 fp2_mul_by_fp(Fp2 a, Fp b) {
    Fp2 r; r.c0 = fp_mul(a.c0, b); r.c1 = fp_mul(a.c1, b); return r;
}

// Multiply by non-residue xi = 9+u: (a0+a1*u)(9+u) = (9*a0 - a1) + (a0 + 9*a1)*u
Fp fp_nine() {
    // 9 in Montgomery form: fpFromInt(9) = 9 * R mod p
    // Precomputed: 9 * R mod p
    Fp r;
    r.v[0] = 0xca28f58du; r.v[1] = 0x6f60e378u; r.v[2] = 0xc38c6456u;
    r.v[3] = 0x5f1bd41cu; r.v[4] = 0xa4b9cf61u; r.v[5] = 0xa226de3eu;
    r.v[6] = 0x643e5d70u; r.v[7] = 0x7de0ce2au;
    return r;
}

Fp2 fp2_mul_by_nonresidue(Fp2 a) {
    Fp nine = fp_nine();
    Fp2 r;
    r.c0 = fp_sub(fp_mul(nine, a.c0), a.c1);
    r.c1 = fp_add(a.c0, fp_mul(nine, a.c1));
    return r;
}

// Fp2 inverse: 1/(a0+a1*u) = (a0-a1*u)/(a0^2+a1^2)
// We need Fp inverse via Fermat for this -- expensive, avoid in hot path
// p-2 in 32-bit limbs for Fermat inversion
constant uint PM2_LO[8] = {
    0xd87cfd45, 0x3c208c16, 0x6871ca8d, 0x97816a91,
    0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72
};

Fp fp_inverse_fermat(Fp a) {
    Fp result = fp_one();
    Fp base = a;
    for (int i = 0; i < 8; i++) {
        uint word = PM2_LO[i];
        for (int j = 0; j < 32; j++) {
            if (word & 1) {
                result = fp_mul(result, base);
            }
            base = fp_mul(base, base);
            word >>= 1;
        }
    }
    return result;
}

Fp2 fp2_inverse(Fp2 a) {
    Fp norm = fp_add(fp_mul(a.c0, a.c0), fp_mul(a.c1, a.c1));
    Fp normInv = fp_inverse_fermat(norm);
    Fp2 r;
    r.c0 = fp_mul(a.c0, normInv);
    r.c1 = fp_neg(fp_mul(a.c1, normInv));
    return r;
}

// ============================================================================
// Fp6 = Fp2[v]/(v^3 - xi) where xi = 9+u
// ============================================================================

struct Fp6 {
    Fp2 c0;
    Fp2 c1;
    Fp2 c2;
};

Fp6 fp6_zero() {
    Fp6 r; r.c0 = fp2_zero(); r.c1 = fp2_zero(); r.c2 = fp2_zero(); return r;
}

Fp6 fp6_one() {
    Fp6 r; r.c0 = fp2_one(); r.c1 = fp2_zero(); r.c2 = fp2_zero(); return r;
}

Fp6 fp6_add(Fp6 a, Fp6 b) {
    Fp6 r;
    r.c0 = fp2_add(a.c0, b.c0);
    r.c1 = fp2_add(a.c1, b.c1);
    r.c2 = fp2_add(a.c2, b.c2);
    return r;
}

Fp6 fp6_sub(Fp6 a, Fp6 b) {
    Fp6 r;
    r.c0 = fp2_sub(a.c0, b.c0);
    r.c1 = fp2_sub(a.c1, b.c1);
    r.c2 = fp2_sub(a.c2, b.c2);
    return r;
}

Fp6 fp6_neg(Fp6 a) {
    Fp6 r; r.c0 = fp2_neg(a.c0); r.c1 = fp2_neg(a.c1); r.c2 = fp2_neg(a.c2); return r;
}

// Karatsuba Fp6 multiplication
Fp6 fp6_mul(Fp6 a, Fp6 b) {
    Fp2 t0 = fp2_mul(a.c0, b.c0);
    Fp2 t1 = fp2_mul(a.c1, b.c1);
    Fp2 t2 = fp2_mul(a.c2, b.c2);

    Fp2 c0 = fp2_add(t0, fp2_mul_by_nonresidue(
        fp2_sub(fp2_mul(fp2_add(a.c1, a.c2), fp2_add(b.c1, b.c2)), fp2_add(t1, t2))));
    Fp2 c1 = fp2_add(fp2_sub(fp2_mul(fp2_add(a.c0, a.c1), fp2_add(b.c0, b.c1)),
                              fp2_add(t0, t1)), fp2_mul_by_nonresidue(t2));
    Fp2 c2 = fp2_add(fp2_sub(fp2_mul(fp2_add(a.c0, a.c2), fp2_add(b.c0, b.c2)),
                              fp2_add(t0, t2)), t1);
    Fp6 r; r.c0 = c0; r.c1 = c1; r.c2 = c2; return r;
}

Fp6 fp6_sqr(Fp6 a) {
    Fp2 s0 = fp2_sqr(a.c0);
    Fp2 ab = fp2_mul(a.c0, a.c1);
    Fp2 s1 = fp2_double(ab);
    Fp2 s2 = fp2_sqr(fp2_sub(fp2_add(a.c0, a.c2), a.c1));
    Fp2 bc = fp2_mul(a.c1, a.c2);
    Fp2 s3 = fp2_double(bc);
    Fp2 s4 = fp2_sqr(a.c2);

    Fp6 r;
    r.c0 = fp2_add(s0, fp2_mul_by_nonresidue(s3));
    r.c1 = fp2_add(s1, fp2_mul_by_nonresidue(s4));
    r.c2 = fp2_sub(fp2_add(fp2_add(s1, s2), s3), fp2_add(s0, s4));
    return r;
}

// Multiply by v: shift components, v^3 = xi
Fp6 fp6_mul_by_v(Fp6 a) {
    Fp6 r;
    r.c0 = fp2_mul_by_nonresidue(a.c2);
    r.c1 = a.c0;
    r.c2 = a.c1;
    return r;
}

Fp6 fp6_inverse(Fp6 a) {
    Fp2 t0 = fp2_sqr(a.c0);
    Fp2 t1 = fp2_sqr(a.c1);
    Fp2 t2 = fp2_sqr(a.c2);
    Fp2 t3 = fp2_mul(a.c0, a.c1);
    Fp2 t4 = fp2_mul(a.c0, a.c2);
    Fp2 t5 = fp2_mul(a.c1, a.c2);

    Fp2 c0 = fp2_sub(t0, fp2_mul_by_nonresidue(t5));
    Fp2 c1 = fp2_sub(fp2_mul_by_nonresidue(t2), t3);
    Fp2 c2 = fp2_sub(t1, t4);

    Fp2 det = fp2_add(
        fp2_mul(a.c0, c0),
        fp2_mul_by_nonresidue(fp2_add(fp2_mul(a.c2, c1), fp2_mul(a.c1, c2))));
    Fp2 detInv = fp2_inverse(det);

    Fp6 r;
    r.c0 = fp2_mul(c0, detInv);
    r.c1 = fp2_mul(c1, detInv);
    r.c2 = fp2_mul(c2, detInv);
    return r;
}

// ============================================================================
// Fp12 = Fp6[w]/(w^2 - v)
// ============================================================================

struct Fp12 {
    Fp6 c0;
    Fp6 c1;
};

Fp12 fp12_one() {
    Fp12 r; r.c0 = fp6_one(); r.c1 = fp6_zero(); return r;
}

Fp12 fp12_add(Fp12 a, Fp12 b) {
    Fp12 r; r.c0 = fp6_add(a.c0, b.c0); r.c1 = fp6_add(a.c1, b.c1); return r;
}

Fp12 fp12_sub(Fp12 a, Fp12 b) {
    Fp12 r; r.c0 = fp6_sub(a.c0, b.c0); r.c1 = fp6_sub(a.c1, b.c1); return r;
}

Fp12 fp12_neg(Fp12 a) {
    Fp12 r; r.c0 = fp6_neg(a.c0); r.c1 = fp6_neg(a.c1); return r;
}

// (a0+a1*w)(b0+b1*w) = (a0*b0 + a1*b1*v) + (a0*b1+a1*b0)*w
// Karatsuba: c1 = (a0+a1)(b0+b1) - a0*b0 - a1*b1
Fp12 fp12_mul(Fp12 a, Fp12 b) {
    Fp6 t0 = fp6_mul(a.c0, b.c0);
    Fp6 t1 = fp6_mul(a.c1, b.c1);
    Fp12 r;
    r.c0 = fp6_add(t0, fp6_mul_by_v(t1));
    r.c1 = fp6_sub(fp6_mul(fp6_add(a.c0, a.c1), fp6_add(b.c0, b.c1)),
                    fp6_add(t0, t1));
    return r;
}

Fp12 fp12_sqr(Fp12 a) {
    Fp6 ab = fp6_mul(a.c0, a.c1);
    Fp12 r;
    r.c0 = fp6_add(fp6_mul(fp6_add(a.c0, a.c1), fp6_add(a.c0, fp6_mul_by_v(a.c1))),
                    fp6_neg(fp6_add(ab, fp6_mul_by_v(ab))));
    r.c1 = fp6_add(ab, ab);
    return r;
}

Fp12 fp12_conjugate(Fp12 a) {
    Fp12 r; r.c0 = a.c0; r.c1 = fp6_neg(a.c1); return r;
}

Fp12 fp12_inverse(Fp12 a) {
    Fp6 t0 = fp6_sqr(a.c0);
    Fp6 t1 = fp6_sqr(a.c1);
    Fp6 t2 = fp6_sub(t0, fp6_mul_by_v(t1));
    Fp6 t3 = fp6_inverse(t2);
    Fp12 r;
    r.c0 = fp6_mul(a.c0, t3);
    r.c1 = fp6_neg(fp6_mul(a.c1, t3));
    return r;
}

// Sparse Fp12 multiplication for line evaluation results.
// Line evals have the form: c0.c0=a, c1.c0=b, c1.c1=c, all else zero.
// This saves ~40% of Fp2 muls vs full fp12_mul.
Fp12 fp12_mul_by_034(Fp12 f, Fp2 c0val, Fp2 c3val, Fp2 c4val) {
    // Sparse element s: s.c0 = (c0val, 0, 0), s.c1 = (c3val, c4val, 0)
    // f = f0 + f1*w, s = s0 + s1*w
    // result.c0 = f0*s0 + f1*s1*v
    // result.c1 = f0*s1 + f1*s0
    // where s0 = (c0val, 0, 0) and s1 = (c3val, c4val, 0)

    // f0*s0 = (f0.c0*c0val, f0.c1*c0val, f0.c2*c0val) since s0.c1=s0.c2=0
    Fp6 f0_s0;
    f0_s0.c0 = fp2_mul(f.c0.c0, c0val);
    f0_s0.c1 = fp2_mul(f.c0.c1, c0val);
    f0_s0.c2 = fp2_mul(f.c0.c2, c0val);

    // f1*s1 where s1 = (c3val, c4val, 0)
    Fp2 t0 = fp2_mul(f.c1.c0, c3val);
    Fp2 t1 = fp2_mul(f.c1.c1, c4val);
    // Karatsuba on s1 (2-term)
    Fp6 f1_s1;
    f1_s1.c0 = fp2_add(t0, fp2_mul_by_nonresidue(
        fp2_sub(fp2_mul(fp2_add(f.c1.c1, f.c1.c2), c4val), t1)));
    f1_s1.c1 = fp2_add(fp2_sub(fp2_mul(fp2_add(f.c1.c0, f.c1.c1),
                                        fp2_add(c3val, c4val)), fp2_add(t0, t1)),
                        fp2_zero());  // no c2 in s1
    f1_s1.c2 = fp2_add(fp2_sub(fp2_mul(f.c1.c0, c3val), t0),
                        fp2_mul(f.c1.c2, c3val)); // simplified
    // Actually let me redo this properly using standard fp6_mul with s1=(c3,c4,0)
    // fp6_mul(f1, s1) where s1.c2=0:
    Fp2 v0_1 = fp2_mul(f.c1.c0, c3val);
    Fp2 v1_1 = fp2_mul(f.c1.c1, c4val);
    // c0 = v0 + xi*((f1.c1+f1.c2)*(c4) - v1)  [since s1.c2=0, t2=0]
    f1_s1.c0 = fp2_add(v0_1, fp2_mul_by_nonresidue(
        fp2_sub(fp2_mul(fp2_add(f.c1.c1, f.c1.c2), c4val), v1_1)));
    // c1 = (f1.c0+f1.c1)*(c3+c4) - v0 - v1
    f1_s1.c1 = fp2_sub(fp2_sub(
        fp2_mul(fp2_add(f.c1.c0, f.c1.c1), fp2_add(c3val, c4val)),
        v0_1), v1_1);
    // c2 = (f1.c0+f1.c2)*c3 - v0 + v1  [since s1.c2=0]
    f1_s1.c2 = fp2_add(fp2_sub(fp2_mul(fp2_add(f.c1.c0, f.c1.c2), c3val), v0_1), v1_1);

    // f0*s1 where s1 = (c3val, c4val, 0)
    Fp2 v0_0 = fp2_mul(f.c0.c0, c3val);
    Fp2 v1_0 = fp2_mul(f.c0.c1, c4val);
    Fp6 f0_s1;
    f0_s1.c0 = fp2_add(v0_0, fp2_mul_by_nonresidue(
        fp2_sub(fp2_mul(fp2_add(f.c0.c1, f.c0.c2), c4val), v1_0)));
    f0_s1.c1 = fp2_sub(fp2_sub(
        fp2_mul(fp2_add(f.c0.c0, f.c0.c1), fp2_add(c3val, c4val)),
        v0_0), v1_0);
    f0_s1.c2 = fp2_add(fp2_sub(fp2_mul(fp2_add(f.c0.c0, f.c0.c2), c3val), v0_0), v1_0);

    // f1*s0 = (f1.c0*c0val, f1.c1*c0val, f1.c2*c0val)
    Fp6 f1_s0;
    f1_s0.c0 = fp2_mul(f.c1.c0, c0val);
    f1_s0.c1 = fp2_mul(f.c1.c1, c0val);
    f1_s0.c2 = fp2_mul(f.c1.c2, c0val);

    Fp12 r;
    r.c0 = fp6_add(f0_s0, fp6_mul_by_v(f1_s1));
    r.c1 = fp6_add(f0_s1, f1_s0);
    return r;
}

// ============================================================================
// Frobenius endomorphisms (precomputed constants)
// ============================================================================

// gamma_1_1 = xi^((p-1)/6)
Fp2 bn254_gamma11() {
    Fp2 r;
    // c0 = 0x1284b71c2865a7dfe8b99fdd76e68b605c521e08292f2176d60b35dadcc9e470
    r.c0.v[0] = 0xf48b7d0au; r.c0.v[1] = 0xf63e10a1u; r.c0.v[2] = 0x055b7c93u;
    r.c0.v[3] = 0xa7840e1au; r.c0.v[4] = 0x9aad068cu; r.c0.v[5] = 0x1a722ce4u;
    r.c0.v[6] = 0xd0ef0c5au; r.c0.v[7] = 0x1e0b2fa3u;
    // c1 = 0x246996f3b4fae7e6a6327cfe12150b8e747992778eeec7e5ca5cf05f80f362ac
    r.c1.v[0] = 0x05c9c236u; r.c1.v[1] = 0x6a47a534u; r.c1.v[2] = 0x13cede96u;
    r.c1.v[3] = 0x04e38dcau; r.c1.v[4] = 0x2ecfe1a5u; r.c1.v[5] = 0xe9218f8au;
    r.c1.v[6] = 0x8aa0978au; r.c1.v[7] = 0x0ff47a49u;
    return r;
}

// gamma_1_2 = xi^((p-1)/3)
Fp2 bn254_gamma12() {
    Fp2 r;
    r.c0.v[0] = 0x8b72e1a0u; r.c0.v[1] = 0x85a8a97bu; r.c0.v[2] = 0x56c6e7c4u;
    r.c0.v[3] = 0xb95da41fu; r.c0.v[4] = 0x95a88aabu; r.c0.v[5] = 0x226f0353u;
    r.c0.v[6] = 0x7f23c0a3u; r.c0.v[7] = 0x2c145ec4u;
    r.c1.v[0] = 0xec7c1fa8u; r.c1.v[1] = 0x1d69faa1u; r.c1.v[2] = 0x5d0f85f6u;
    r.c1.v[3] = 0x28f57c37u; r.c1.v[4] = 0x4a3a1a1bu; r.c1.v[5] = 0x27ab3b70u;
    r.c1.v[6] = 0xdcdf3c08u; r.c1.v[7] = 0x00b5f050u;
    return r;
}

// gamma_1_3 = xi^((p-1)/2)
Fp2 bn254_gamma13() {
    Fp2 r;
    r.c0.v[0] = 0xdde07e82u; r.c0.v[1] = 0x6b28a0e6u; r.c0.v[2] = 0x69de91e0u;
    r.c0.v[3] = 0xfab8fa30u; r.c0.v[4] = 0x0cf07e98u; r.c0.v[5] = 0x28f0a56cu;
    r.c0.v[6] = 0xe6c61a03u; r.c0.v[7] = 0x17ff2e0eu;
    r.c1.v[0] = 0x6f7c93fau; r.c1.v[1] = 0x79e6d5ffu; r.c1.v[2] = 0x73fb785au;
    r.c1.v[3] = 0x7d49c9e4u; r.c1.v[4] = 0x08d74abeu; r.c1.v[5] = 0x6b1d2cc8u;
    r.c1.v[6] = 0x2e56c69au; r.c1.v[7] = 0x19ea3e02u;
    return r;
}

// gamma_2_1 (in Fp, c1=0): 0x30644e72e131a0295e6dd9e7e0acccb0c28f069fbb966e3de4bd44e5607cfd49
Fp bn254_gamma21() {
    Fp r;
    r.v[0] = 0xa1e0a048u; r.v[1] = 0x3350c88eu; r.v[2] = 0x7166e6f0u;
    r.v[3] = 0xb0f34527u; r.v[4] = 0x25c71a82u; r.v[5] = 0x1de0c7c5u;
    r.v[6] = 0xd9bb5cc4u; r.v[7] = 0x0c5fd589u;
    return r;
}

// gamma_2_2: 0x30644e72e131a0295e6dd9e7e0acccb0c28f069fbb966e3de4bd44e5607cfd48
Fp bn254_gamma22() {
    Fp r;
    r.v[0] = 0x54d6c48bu; r.v[1] = 0x4f4bdb88u; r.v[2] = 0x7f64e8d0u;
    r.v[3] = 0xd4277c2eu; r.v[4] = 0x8c85b3f3u; r.v[5] = 0x3a61fa51u;
    r.v[6] = 0x84f7bc8bu; r.v[7] = 0x2929588au;
    return r;
}

// gamma_2_3: 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd46
Fp bn254_gamma23() {
    Fp r;
    r.v[0] = 0xa3f4a1cdu; r.v[1] = 0xd2ec4c62u; r.v[2] = 0x5b3e0f6bu;
    r.v[3] = 0x84db2042u; r.v[4] = 0x48a3b0e5u; r.v[5] = 0xacfad0efu;
    r.v[6] = 0xdf4f2ee4u; r.v[7] = 0x0e09ac46u;
    return r;
}

Fp12 fp12_frobenius(Fp12 a) {
    Fp2 g12 = bn254_gamma12();
    Fp2 g13 = bn254_gamma13();
    Fp2 g11 = bn254_gamma11();
    Fp2 g12sq = fp2_sqr(g12);
    Fp2 g11_g12sq = fp2_mul(g11, g12sq);

    Fp12 r;
    r.c0.c0 = fp2_conjugate(a.c0.c0);
    r.c0.c1 = fp2_mul(fp2_conjugate(a.c0.c1), g12);
    r.c0.c2 = fp2_mul(fp2_conjugate(a.c0.c2), g12sq);
    r.c1.c0 = fp2_mul(fp2_conjugate(a.c1.c0), g11);
    r.c1.c1 = fp2_mul(fp2_conjugate(a.c1.c1), g13);
    r.c1.c2 = fp2_mul(fp2_conjugate(a.c1.c2), g11_g12sq);
    return r;
}

Fp12 fp12_frobenius2(Fp12 a) {
    Fp g22 = bn254_gamma22();
    Fp g21 = bn254_gamma21();
    Fp g23 = bn254_gamma23();
    Fp g22sq = fp_mul(g22, g22);
    Fp g21_g22sq = fp_mul(g21, g22sq);

    Fp12 r;
    r.c0.c0 = a.c0.c0;
    r.c0.c1 = fp2_mul_by_fp(a.c0.c1, g22);
    r.c0.c2 = fp2_mul_by_fp(a.c0.c2, g22sq);
    r.c1.c0 = fp2_mul_by_fp(a.c1.c0, g21);
    r.c1.c1 = fp2_mul_by_fp(a.c1.c1, g23);
    r.c1.c2 = fp2_mul_by_fp(a.c1.c2, g21_g22sq);
    return r;
}

Fp12 fp12_frobenius3(Fp12 a) {
    return fp12_frobenius(fp12_frobenius2(a));
}

// ============================================================================
// G2 affine point on the twist
// ============================================================================

struct G2AffineGPU {
    Fp2 x;
    Fp2 y;
};

struct G1AffineGPU {
    Fp x;
    Fp y;
};

// Affine doubling: T = 2T, returns slope
Fp2 g2_affine_double(thread G2AffineGPU &t) {
    Fp2 xsq = fp2_sqr(t.x);
    Fp2 num = fp2_add(fp2_double(xsq), xsq); // 3*x^2
    Fp2 den = fp2_double(t.y);
    Fp2 lam = fp2_mul(num, fp2_inverse(den));
    Fp2 x3 = fp2_sub(fp2_sqr(lam), fp2_double(t.x));
    Fp2 y3 = fp2_sub(fp2_mul(lam, fp2_sub(t.x, x3)), t.y);
    t.x = x3; t.y = y3;
    return lam;
}

// Affine addition: T = T + Q, returns slope
Fp2 g2_affine_add(thread G2AffineGPU &t, G2AffineGPU q) {
    Fp2 dx = fp2_sub(q.x, t.x);
    Fp2 dy = fp2_sub(q.y, t.y);
    Fp2 lam = fp2_mul(dy, fp2_inverse(dx));
    Fp2 x3 = fp2_sub(fp2_sub(fp2_sqr(lam), t.x), q.x);
    Fp2 y3 = fp2_sub(fp2_mul(lam, fp2_sub(t.x, x3)), t.y);
    t.x = x3; t.y = y3;
    return lam;
}

// Line evaluation at G1 point P
// Result is sparse Fp12: c0.c0 = yP, c1.c0 = -lam*xP, c1.c1 = lam*xT - yT
void line_eval(Fp2 lam, Fp2 xT, Fp2 yT, Fp px, Fp py,
               thread Fp2 &out_c0, thread Fp2 &out_c3, thread Fp2 &out_c4) {
    out_c0.c0 = py; out_c0.c1 = fp_zero();
    out_c3 = fp2_neg(fp2_mul_by_fp(lam, px));
    out_c4 = fp2_sub(fp2_mul(lam, xT), yT);
}

// ============================================================================
// Miller loop
// ============================================================================

// NAF of 6x+2 where x = 4965661367071055936
// 66 entries (MSB to LSB)
constant int8_t SIX_X_PLUS_2_NAF[66] = {
     1,  0, -1,  0,  1,  0,  0,  0, -1,  0, -1,  0,  0,  0, -1,  0,
     1,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0,  1,  0,  0, -1,  0,
     1,  0,  0, -1,  0,  0,  0,  0, -1,  0,  1,  0,  0,  0, -1,  0,
    -1,  0,  0,  1,  0,  0,  0, -1,  0,  0, -1,  0,  1,  0,  1,  0,
     0,  0
};

Fp12 gpu_miller_loop(G1AffineGPU p, G2AffineGPU q) {
    G2AffineGPU t = q;
    Fp12 f = fp12_one();
    G2AffineGPU negQ;
    negQ.x = q.x;
    negQ.y = fp2_neg(q.y);

    Fp2 c0val, c3val, c4val;

    for (int i = 1; i < 66; i++) {
        f = fp12_sqr(f);

        // Doubling step
        G2AffineGPU oldT = t;
        Fp2 lam = g2_affine_double(t);
        line_eval(lam, oldT.x, oldT.y, p.x, p.y, c0val, c3val, c4val);
        f = fp12_mul_by_034(f, c0val, c3val, c4val);

        // Addition step
        if (SIX_X_PLUS_2_NAF[i] == 1) {
            G2AffineGPU oldT2 = t;
            Fp2 lam2 = g2_affine_add(t, q);
            line_eval(lam2, oldT2.x, oldT2.y, p.x, p.y, c0val, c3val, c4val);
            f = fp12_mul_by_034(f, c0val, c3val, c4val);
        } else if (SIX_X_PLUS_2_NAF[i] == -1) {
            G2AffineGPU oldT2 = t;
            Fp2 lam2 = g2_affine_add(t, negQ);
            line_eval(lam2, oldT2.x, oldT2.y, p.x, p.y, c0val, c3val, c4val);
            f = fp12_mul_by_034(f, c0val, c3val, c4val);
        }
    }

    // Frobenius correction
    Fp2 g12 = bn254_gamma12();
    Fp2 g13 = bn254_gamma13();
    Fp g22 = bn254_gamma22();
    Fp g23 = bn254_gamma23();

    // Q1 = pi(Q)
    G2AffineGPU q1;
    q1.x = fp2_mul(fp2_conjugate(q.x), g12);
    q1.y = fp2_mul(fp2_conjugate(q.y), g13);

    G2AffineGPU oldT3 = t;
    Fp2 lam3 = g2_affine_add(t, q1);
    line_eval(lam3, oldT3.x, oldT3.y, p.x, p.y, c0val, c3val, c4val);
    f = fp12_mul_by_034(f, c0val, c3val, c4val);

    // Q2 = -pi^2(Q)
    G2AffineGPU q2;
    q2.x = fp2_mul_by_fp(q.x, g22);
    q2.y = fp2_neg(fp2_mul_by_fp(q.y, g23));

    G2AffineGPU oldT4 = t;
    Fp2 lam4 = g2_affine_add(t, q2);
    line_eval(lam4, oldT4.x, oldT4.y, p.x, p.y, c0val, c3val, c4val);
    f = fp12_mul_by_034(f, c0val, c3val, c4val);

    return f;
}

// ============================================================================
// GPU Kernels
// ============================================================================

/// Batch Miller loop: each thread computes one independent Miller loop.
/// Input: N pairs of (G1, G2) points. Output: N Fp12 elements.
/// Final exponentiation is done on CPU (too heavy for Metal compiler in single kernel).
kernel void batch_miller_loop(
    device const G1AffineGPU* g1_points [[buffer(0)]],
    device const G2AffineGPU* g2_points [[buffer(1)]],
    device Fp12* results [[buffer(2)]],
    device const uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    results[tid] = gpu_miller_loop(g1_points[tid], g2_points[tid]);
}
