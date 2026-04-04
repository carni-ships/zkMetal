// BabyJubjub twisted Edwards curve point operations for Metal GPU
// Curve: a*x^2 + y^2 = 1 + d*x^2*y^2
// a = 168700, d = 168696
// Base field: BN254 Fr
// Extended coordinates: (X, Y, Z, T) where x = X/Z, y = Y/Z, T = XY/Z

#ifndef BABYJUBJUB_CURVE_METAL
#define BABYJUBJUB_CURVE_METAL

#include "../fields/bn254_fr.metal"

// Point types
struct BJJPointAffine {
    Fr x;
    Fr y;
};

struct BJJPointExtended {
    Fr x;
    Fr y;
    Fr z;
    Fr t;
};

// Curve constants a=168700, d=168696 in Montgomery form
// Precomputed: frFromInt(168700) and frFromInt(168696)
// These are passed as parameters or computed inline.

// Convert raw integer to Montgomery form
Fr bjj_from_int(uint val) {
    Fr raw = fr_zero();
    raw.v[0] = val;
    // Multiply by R^2 mod r to get Montgomery form
    Fr r2;
    r2.v[0] = 0xae216da7u; r2.v[1] = 0x1bb8e645u;
    r2.v[2] = 0xe35c59e3u; r2.v[3] = 0x53fe3ab1u;
    r2.v[4] = 0x53bb8085u; r2.v[5] = 0x8c49833du;
    r2.v[6] = 0x7f4e44a5u; r2.v[7] = 0x0216d0b1u;
    return fr_mul(raw, r2);
}

Fr bjj_a() {
    return bjj_from_int(168700u);
}

Fr bjj_d() {
    return bjj_from_int(168696u);
}

BJJPointExtended bjj_identity() {
    BJJPointExtended p;
    p.x = fr_zero();
    p.y = fr_one();
    p.z = fr_one();
    p.t = fr_zero();
    return p;
}

bool bjj_is_identity(BJJPointExtended p) {
    if (!fr_is_zero(p.x)) return false;
    uint borrow;
    Fr diff = fr_sub_raw(p.y, p.z, borrow);
    return fr_is_zero(diff) && borrow == 0;
}

BJJPointExtended bjj_from_affine(BJJPointAffine a) {
    BJJPointExtended p;
    p.x = a.x;
    p.y = a.y;
    p.z = fr_one();
    p.t = fr_mul(a.x, a.y);
    return p;
}

// Point addition using extended coordinates
// For ax^2 + y^2 = 1 + dx^2y^2:
// A=X1*X2, B=Y1*Y2, C=d*T1*T2, D=Z1*Z2
// E=(X1+Y1)*(X2+Y2)-A-B, F=D-C, G=D+C, H=B-a*A
// X3=E*F, Y3=G*H, T3=E*H, Z3=F*G
BJJPointExtended bjj_add(BJJPointExtended p, BJJPointExtended q, Fr a_const, Fr d_const) {
    Fr aa = fr_mul(p.x, q.x);
    Fr bb = fr_mul(p.y, q.y);
    Fr cc = fr_mul(fr_mul(p.t, q.t), d_const);
    Fr dd = fr_mul(p.z, q.z);
    Fr e = fr_sub(fr_mul(fr_add(p.x, p.y), fr_add(q.x, q.y)), fr_add(aa, bb));
    Fr f = fr_sub(dd, cc);
    Fr g = fr_add(dd, cc);
    Fr h = fr_sub(bb, fr_mul(a_const, aa));

    BJJPointExtended r;
    r.x = fr_mul(e, f);
    r.y = fr_mul(g, h);
    r.z = fr_mul(f, g);
    r.t = fr_mul(e, h);
    return r;
}

// Point doubling
// A=X^2, B=Y^2, C=2*Z^2, D=a*A
// E=(X+Y)^2-A-B, G=D+B, F=G-C, H=D-B
BJJPointExtended bjj_double(BJJPointExtended p, Fr a_const) {
    Fr aa = fr_sqr(p.x);
    Fr bb = fr_sqr(p.y);
    Fr cc = fr_double(fr_sqr(p.z));
    Fr dd = fr_mul(a_const, aa);
    Fr e = fr_sub(fr_sqr(fr_add(p.x, p.y)), fr_add(aa, bb));
    Fr g = fr_add(dd, bb);
    Fr f = fr_sub(g, cc);
    Fr h = fr_sub(dd, bb);

    BJJPointExtended r;
    r.x = fr_mul(e, f);
    r.y = fr_mul(g, h);
    r.z = fr_mul(f, g);
    r.t = fr_mul(e, h);
    return r;
}

// Point negation
BJJPointExtended bjj_neg(BJJPointExtended p) {
    BJJPointExtended r;
    r.x = fr_sub(fr_modulus(), p.x);  // This is wrong for Montgomery; use fr_sub(zero, x)
    r.x = fr_sub(fr_zero(), p.x);
    r.y = p.y;
    r.z = p.z;
    r.t = fr_sub(fr_zero(), p.t);
    return r;
}

// Scalar multiplication kernel: each thread computes scalar * point
kernel void bjj_scalar_mul(
    device const BJJPointAffine* points [[buffer(0)]],
    device const uint* scalars [[buffer(1)]],  // 8 uint32 per scalar (256 bits)
    device BJJPointAffine* results [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    Fr a_const = bjj_a();
    Fr d_const = bjj_d();

    BJJPointExtended base = bjj_from_affine(points[gid]);
    BJJPointExtended acc = bjj_identity();

    uint offset = gid * 8;
    for (int limb = 0; limb < 8; limb++) {
        uint word = scalars[offset + limb];
        for (int bit = 0; bit < 32; bit++) {
            if (word & 1u) {
                acc = bjj_add(acc, base, a_const, d_const);
            }
            base = bjj_double(base, a_const);
            word >>= 1;
        }
    }

    // Convert back to affine (requires field inversion — expensive on GPU)
    // For batch processing, output extended and convert on CPU
    // For now, store extended X,Y (caller handles Z normalization)
    // Simple approach: just store the extended point as-is
    // Actually, we output affine by computing inverse
    // GPU inversion via Fermat: a^(p-2) — very expensive, ~256 fr_sqr + ~128 fr_mul
    // Better: output extended and batch-invert on CPU
    results[gid].x = acc.x;  // Actually X (not x = X/Z)
    results[gid].y = acc.y;  // Actually Y (not y = Y/Z)
    // Caller must handle Z coordinate for proper affine conversion
}

// Batch scalar multiplication storing extended results
// Output is 4 Fr values per point (X, Y, Z, T)
kernel void bjj_batch_scalar_mul(
    device const BJJPointAffine* points [[buffer(0)]],
    device const uint* scalars [[buffer(1)]],
    device Fr* results [[buffer(2)]],  // 4 Fr per output point
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    Fr a_const = bjj_a();
    Fr d_const = bjj_d();

    BJJPointExtended base = bjj_from_affine(points[gid]);
    BJJPointExtended acc = bjj_identity();

    uint offset = gid * 8;
    for (int limb = 0; limb < 8; limb++) {
        uint word = scalars[offset + limb];
        for (int bit = 0; bit < 32; bit++) {
            if (word & 1u) {
                acc = bjj_add(acc, base, a_const, d_const);
            }
            base = bjj_double(base, a_const);
            word >>= 1;
        }
    }

    uint out = gid * 4;
    results[out + 0] = acc.x;
    results[out + 1] = acc.y;
    results[out + 2] = acc.z;
    results[out + 3] = acc.t;
}

#endif // BABYJUBJUB_CURVE_METAL
