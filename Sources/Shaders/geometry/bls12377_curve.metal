// BLS12-377 elliptic curve G1 point operations for Metal GPU
// y^2 = x^3 + 1, Jacobian projective coordinates

#ifndef BLS12377_CURVE_METAL
#define BLS12377_CURVE_METAL

#include "../fields/bls12377_fq.metal"

Point377Projective point377_identity() {
    Point377Projective p;
    p.x = fq377_one();
    p.y = fq377_one();
    p.z = fq377_zero();
    return p;
}

bool point377_is_identity(Point377Projective p) {
    return fq377_is_zero(p.z);
}

Point377Projective point377_from_affine(Point377Affine a) {
    Point377Projective p;
    p.x = a.x;
    p.y = a.y;
    p.z = fq377_one();
    return p;
}

// Point doubling: 4M + 6S + 7add (a=0 for BLS12-377)
// Uses fq377_mul(a,a) instead of fq377_sqr to reduce register pressure
Point377Projective point377_double(Point377Projective p) {
    if (point377_is_identity(p)) return p;

    Fq377 a = fq377_mul(p.x, p.x);
    Fq377 b = fq377_mul(p.y, p.y);
    Fq377 c = fq377_mul(b, b);

    Fq377 d = fq377_sub(fq377_mul(fq377_add(p.x, b), fq377_add(p.x, b)), fq377_add(a, c));
    d = fq377_double(d);

    Fq377 e = fq377_add(fq377_double(a), a); // 3*X^2
    Fq377 f = fq377_mul(e, e);

    Point377Projective r;
    r.x = fq377_sub(f, fq377_double(d));
    r.y = fq377_sub(fq377_mul(e, fq377_sub(d, r.x)), fq377_double(fq377_double(fq377_double(c))));
    Fq377 yz = fq377_add(p.y, p.z);
    r.z = fq377_sub(fq377_mul(yz, yz), fq377_add(b, fq377_mul(p.z, p.z)));
    return r;
}

// Mixed addition: projective + affine
Point377Projective point377_add_mixed(Point377Projective p, Point377Affine q) {
    if (point377_is_identity(p)) return point377_from_affine(q);

    Fq377 z1z1 = fq377_mul(p.z, p.z);
    Fq377 u2 = fq377_mul(q.x, z1z1);
    Fq377 h = fq377_sub(u2, p.x);

    if (fq377_is_zero(h)) {
        Fq377 s2 = fq377_mul(q.y, fq377_mul(p.z, z1z1));
        Fq377 rr = fq377_double(fq377_sub(s2, p.y));
        if (fq377_is_zero(rr)) return point377_double(p);
        return point377_identity();
    }

    Fq377 s2 = fq377_mul(q.y, fq377_mul(p.z, z1z1));
    Point377Projective result;
    result.z = fq377_double(fq377_mul(p.z, h));
    Fq377 hh = fq377_mul(h, h);
    Fq377 i = fq377_double(fq377_double(hh));
    Fq377 v = fq377_mul(p.x, i);
    Fq377 j = fq377_mul(h, i);
    Fq377 rr = fq377_double(fq377_sub(s2, p.y));
    result.x = fq377_sub(fq377_sub(fq377_mul(rr, rr), j), fq377_double(v));
    result.y = fq377_sub(fq377_mul(rr, fq377_sub(v, result.x)),
                      fq377_double(fq377_mul(p.y, j)));
    return result;
}

// Full addition: projective + projective
Point377Projective point377_add(Point377Projective p, Point377Projective q) {
    if (point377_is_identity(p)) return q;
    if (point377_is_identity(q)) return p;

    Fq377 z1z1 = fq377_mul(p.z, p.z);
    Fq377 z2z2 = fq377_mul(q.z, q.z);
    Fq377 u1 = fq377_mul(p.x, z2z2);
    Fq377 u2 = fq377_mul(q.x, z1z1);
    Fq377 s1 = fq377_mul(p.y, fq377_mul(q.z, z2z2));
    Fq377 s2 = fq377_mul(q.y, fq377_mul(p.z, z1z1));

    Fq377 h = fq377_sub(u2, u1);
    Fq377 rr = fq377_double(fq377_sub(s2, s1));

    if (fq377_is_zero(h)) {
        if (fq377_is_zero(rr)) return point377_double(p);
        return point377_identity();
    }

    Point377Projective result;
    result.z = fq377_mul(fq377_double(fq377_mul(p.z, q.z)), h);

    Fq377 dh = fq377_double(h);
    Fq377 ii = fq377_mul(dh, dh);
    Fq377 v = fq377_mul(u1, ii);
    Fq377 j = fq377_mul(h, ii);

    result.x = fq377_sub(fq377_sub(fq377_mul(rr, rr), j), fq377_double(v));
    result.y = fq377_sub(fq377_mul(rr, fq377_sub(v, result.x)),
                      fq377_double(fq377_mul(s1, j)));
    return result;
}

#endif // BLS12377_CURVE_METAL
