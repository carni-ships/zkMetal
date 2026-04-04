// BN254 elliptic curve point operations for Metal GPU
// y^2 = x^3 + 3, Jacobian projective coordinates

#ifndef BN254_CURVE_METAL
#define BN254_CURVE_METAL

#include "../fields/bn254_fp.metal"

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

PointProjective point_from_affine(PointAffine a) {
    PointProjective p;
    p.x = a.x;
    p.y = a.y;
    p.z = fp_one();
    return p;
}

// Point doubling: 4M + 6S + 7add (a=0 for BN254)
// Uses fp_mul(a,a) instead of fp_sqr to reduce register pressure (CIOS 10 temps vs SOS 17)
PointProjective point_double(PointProjective p) {
    if (point_is_identity(p)) return p;

    Fp a = fp_mul(p.x, p.x);
    Fp b = fp_mul(p.y, p.y);
    Fp c = fp_mul(b, b);

    Fp d = fp_sub(fp_mul(fp_add(p.x, b), fp_add(p.x, b)), fp_add(a, c));
    d = fp_double(d);

    Fp e = fp_add(fp_double(a), a); // 3*X^2
    Fp f = fp_mul(e, e);

    PointProjective r;
    r.x = fp_sub(f, fp_double(d));
    r.y = fp_sub(fp_mul(e, fp_sub(d, r.x)), fp_double(fp_double(fp_double(c))));
    Fp yz = fp_add(p.y, p.z);
    r.z = fp_sub(fp_mul(yz, yz), fp_add(b, fp_mul(p.z, p.z)));
    return r;
}

// Mixed addition: projective + affine
// Uses fp_mul(a,a) instead of fp_sqr to reduce register pressure
PointProjective point_add_mixed(PointProjective p, PointAffine q) {
    if (point_is_identity(p)) return point_from_affine(q);

    Fp z1z1 = fp_mul(p.z, p.z);
    Fp u2 = fp_mul(q.x, z1z1);
    Fp h = fp_sub(u2, p.x);

    if (fp_is_zero(h)) {
        Fp s2 = fp_mul(q.y, fp_mul(p.z, z1z1));
        Fp rr = fp_double(fp_sub(s2, p.y));
        if (fp_is_zero(rr)) return point_double(p);
        return point_identity();
    }

    Fp s2 = fp_mul(q.y, fp_mul(p.z, z1z1));
    PointProjective result;
    result.z = fp_double(fp_mul(p.z, h));
    Fp hh = fp_mul(h, h);
    Fp i = fp_double(fp_double(hh));
    Fp v = fp_mul(p.x, i);
    Fp j = fp_mul(h, i);
    Fp rr = fp_double(fp_sub(s2, p.y));
    result.x = fp_sub(fp_sub(fp_mul(rr, rr), j), fp_double(v));
    result.y = fp_sub(fp_mul(rr, fp_sub(v, result.x)),
                      fp_double(fp_mul(p.y, j)));
    return result;
}

// Fast mixed addition: no identity/doubling checks.
// Use only when p is guaranteed non-identity and p != ±q.
// Saves branch overhead in tight MSM bucket reduction loops.
PointProjective point_add_mixed_unsafe(PointProjective p, PointAffine q) {
    Fp z1z1 = fp_mul(p.z, p.z);
    Fp u2 = fp_mul(q.x, z1z1);
    Fp s2 = fp_mul(q.y, fp_mul(p.z, z1z1));
    Fp h = fp_sub(u2, p.x);
    PointProjective result;
    result.z = fp_double(fp_mul(p.z, h));
    Fp hh = fp_mul(h, h);
    Fp i = fp_double(fp_double(hh));
    Fp v = fp_mul(p.x, i);
    Fp j = fp_mul(h, i);
    Fp rr = fp_double(fp_sub(s2, p.y));
    result.x = fp_sub(fp_sub(fp_mul(rr, rr), j), fp_double(v));
    result.y = fp_sub(fp_mul(rr, fp_sub(v, result.x)),
                      fp_double(fp_mul(p.y, j)));
    return result;
}

// Full addition: projective + projective
// Uses fp_mul(a,a) instead of fp_sqr to reduce register pressure
PointProjective point_add(PointProjective p, PointProjective q) {
    if (point_is_identity(p)) return q;
    if (point_is_identity(q)) return p;

    Fp z1z1 = fp_mul(p.z, p.z);
    Fp z2z2 = fp_mul(q.z, q.z);
    Fp u1 = fp_mul(p.x, z2z2);
    Fp u2 = fp_mul(q.x, z1z1);
    Fp s1 = fp_mul(p.y, fp_mul(q.z, z2z2));
    Fp s2 = fp_mul(q.y, fp_mul(p.z, z1z1));

    Fp h = fp_sub(u2, u1);
    Fp rr = fp_double(fp_sub(s2, s1));

    if (fp_is_zero(h)) {
        if (fp_is_zero(rr)) return point_double(p);
        return point_identity();
    }

    PointProjective result;
    result.z = fp_mul(fp_double(fp_mul(p.z, q.z)), h);

    Fp dh = fp_double(h);
    Fp i = fp_mul(dh, dh);
    Fp v = fp_mul(u1, i);
    Fp j = fp_mul(h, i);

    result.x = fp_sub(fp_sub(fp_mul(rr, rr), j), fp_double(v));
    result.y = fp_sub(fp_mul(rr, fp_sub(v, result.x)),
                      fp_double(fp_mul(s1, j)));
    return result;
}

#endif // BN254_CURVE_METAL
