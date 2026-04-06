// BLS12-381 elliptic curve G1 point operations for Metal GPU
// y^2 = x^3 + 4, Jacobian projective coordinates

#ifndef BLS12381_CURVE_METAL
#define BLS12381_CURVE_METAL

#include "../fields/bls12381_fq.metal"

Point381Projective point381_identity() {
    Point381Projective p;
    p.x = fp381_one();
    p.y = fp381_one();
    p.z = fp381_zero();
    return p;
}

bool point381_is_identity(Point381Projective p) {
    return fp381_is_zero(p.z);
}

Point381Projective point381_from_affine(Point381Affine a) {
    Point381Projective p;
    p.x = a.x;
    p.y = a.y;
    p.z = fp381_one();
    return p;
}

// Point doubling: 4M + 6S + 7add (a=0 for BLS12-381: y^2 = x^3 + 4)
// Uses fp381_mul(a,a) instead of fp381_sqr to reduce register pressure
Point381Projective point381_double(Point381Projective p) {
    if (point381_is_identity(p)) return p;

    Fp381 a = fp381_mul(p.x, p.x);
    Fp381 b = fp381_mul(p.y, p.y);
    Fp381 c = fp381_mul(b, b);

    Fp381 d = fp381_sub(fp381_mul(fp381_add(p.x, b), fp381_add(p.x, b)), fp381_add(a, c));
    d = fp381_double(d);

    Fp381 e = fp381_add(fp381_double(a), a); // 3*X^2
    Fp381 f = fp381_mul(e, e);

    Point381Projective r;
    r.x = fp381_sub(f, fp381_double(d));
    r.y = fp381_sub(fp381_mul(e, fp381_sub(d, r.x)), fp381_double(fp381_double(fp381_double(c))));
    Fp381 yz = fp381_add(p.y, p.z);
    r.z = fp381_sub(fp381_mul(yz, yz), fp381_add(b, fp381_mul(p.z, p.z)));
    return r;
}

// Mixed addition: projective + affine
Point381Projective point381_add_mixed(Point381Projective p, Point381Affine q) {
    if (point381_is_identity(p)) return point381_from_affine(q);

    Fp381 z1z1 = fp381_mul(p.z, p.z);
    Fp381 u2 = fp381_mul(q.x, z1z1);
    Fp381 h = fp381_sub(u2, p.x);

    if (fp381_is_zero(h)) {
        Fp381 s2 = fp381_mul(q.y, fp381_mul(p.z, z1z1));
        Fp381 rr = fp381_double(fp381_sub(s2, p.y));
        if (fp381_is_zero(rr)) return point381_double(p);
        return point381_identity();
    }

    Fp381 s2 = fp381_mul(q.y, fp381_mul(p.z, z1z1));
    Point381Projective result;
    result.z = fp381_double(fp381_mul(p.z, h));
    Fp381 hh = fp381_mul(h, h);
    Fp381 i = fp381_double(fp381_double(hh));
    Fp381 v = fp381_mul(p.x, i);
    Fp381 j = fp381_mul(h, i);
    Fp381 rr = fp381_double(fp381_sub(s2, p.y));
    result.x = fp381_sub(fp381_sub(fp381_mul(rr, rr), j), fp381_double(v));
    result.y = fp381_sub(fp381_mul(rr, fp381_sub(v, result.x)),
                      fp381_double(fp381_mul(p.y, j)));
    return result;
}

// Full addition: projective + projective
Point381Projective point381_add(Point381Projective p, Point381Projective q) {
    if (point381_is_identity(p)) return q;
    if (point381_is_identity(q)) return p;

    Fp381 z1z1 = fp381_mul(p.z, p.z);
    Fp381 z2z2 = fp381_mul(q.z, q.z);
    Fp381 u1 = fp381_mul(p.x, z2z2);
    Fp381 u2 = fp381_mul(q.x, z1z1);
    Fp381 s1 = fp381_mul(p.y, fp381_mul(q.z, z2z2));
    Fp381 s2 = fp381_mul(q.y, fp381_mul(p.z, z1z1));

    Fp381 h = fp381_sub(u2, u1);
    Fp381 rr = fp381_double(fp381_sub(s2, s1));

    if (fp381_is_zero(h)) {
        if (fp381_is_zero(rr)) return point381_double(p);
        return point381_identity();
    }

    Point381Projective result;
    result.z = fp381_mul(fp381_double(fp381_mul(p.z, q.z)), h);

    Fp381 dh = fp381_double(h);
    Fp381 ii = fp381_mul(dh, dh);
    Fp381 v = fp381_mul(u1, ii);
    Fp381 j = fp381_mul(h, ii);

    result.x = fp381_sub(fp381_sub(fp381_mul(rr, rr), j), fp381_double(v));
    result.y = fp381_sub(fp381_mul(rr, fp381_sub(v, result.x)),
                      fp381_double(fp381_mul(s1, j)));
    return result;
}

#endif // BLS12381_CURVE_METAL
