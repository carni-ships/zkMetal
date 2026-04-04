// Grumpkin elliptic curve point operations for Metal GPU
// y^2 = x^3 - 17 (a=0, b=-17), Jacobian projective coordinates
// Base field = BN254 Fr (scalar field of BN254)

#ifndef GRUMPKIN_CURVE_METAL
#define GRUMPKIN_CURVE_METAL

#include "../fields/bn254_fr.metal"

struct GrumpkinPointAffine {
    Fr x;
    Fr y;
};

struct GrumpkinPointProjective {
    Fr x;
    Fr y;
    Fr z;
};

GrumpkinPointProjective grumpkin_point_identity() {
    GrumpkinPointProjective p;
    p.x = fr_one();
    p.y = fr_one();
    p.z = fr_zero();
    return p;
}

bool grumpkin_point_is_identity(GrumpkinPointProjective p) {
    return fr_is_zero(p.z);
}

GrumpkinPointProjective grumpkin_point_from_affine(GrumpkinPointAffine a) {
    GrumpkinPointProjective p;
    p.x = a.x;
    p.y = a.y;
    p.z = fr_one();
    return p;
}

Fr fr_neg(Fr a) {
    if (fr_is_zero(a)) return a;
    uint borrow;
    return fr_sub_raw(fr_modulus(), a, borrow);
}

// Point doubling: 4M + 6S + 7add (a=0 for Grumpkin: y^2 = x^3 - 17)
GrumpkinPointProjective grumpkin_point_double(GrumpkinPointProjective p) {
    if (grumpkin_point_is_identity(p)) return p;

    Fr a = fr_mul(p.x, p.x);
    Fr b = fr_mul(p.y, p.y);
    Fr c = fr_mul(b, b);

    Fr d = fr_sub(fr_mul(fr_add(p.x, b), fr_add(p.x, b)), fr_add(a, c));
    d = fr_double(d);

    Fr e = fr_add(fr_double(a), a); // 3*X^2
    Fr f = fr_mul(e, e);

    GrumpkinPointProjective r;
    r.x = fr_sub(f, fr_double(d));
    r.y = fr_sub(fr_mul(e, fr_sub(d, r.x)), fr_double(fr_double(fr_double(c))));
    Fr yz = fr_add(p.y, p.z);
    r.z = fr_sub(fr_mul(yz, yz), fr_add(b, fr_mul(p.z, p.z)));
    return r;
}

// Mixed addition: projective + affine
GrumpkinPointProjective grumpkin_point_add_mixed(GrumpkinPointProjective p, GrumpkinPointAffine q) {
    if (grumpkin_point_is_identity(p)) return grumpkin_point_from_affine(q);

    Fr z1z1 = fr_mul(p.z, p.z);
    Fr u2 = fr_mul(q.x, z1z1);
    Fr h = fr_sub(u2, p.x);

    if (fr_is_zero(h)) {
        Fr s2 = fr_mul(q.y, fr_mul(p.z, z1z1));
        Fr rr = fr_double(fr_sub(s2, p.y));
        if (fr_is_zero(rr)) return grumpkin_point_double(p);
        return grumpkin_point_identity();
    }

    Fr s2 = fr_mul(q.y, fr_mul(p.z, z1z1));
    GrumpkinPointProjective result;
    result.z = fr_double(fr_mul(p.z, h));
    Fr hh = fr_mul(h, h);
    Fr i = fr_double(fr_double(hh));
    Fr v = fr_mul(p.x, i);
    Fr j = fr_mul(h, i);
    Fr rr = fr_double(fr_sub(s2, p.y));
    result.x = fr_sub(fr_sub(fr_mul(rr, rr), j), fr_double(v));
    result.y = fr_sub(fr_mul(rr, fr_sub(v, result.x)),
                      fr_double(fr_mul(p.y, j)));
    return result;
}

// Fast mixed addition: no identity/doubling checks
GrumpkinPointProjective grumpkin_point_add_mixed_unsafe(GrumpkinPointProjective p, GrumpkinPointAffine q) {
    Fr z1z1 = fr_mul(p.z, p.z);
    Fr u2 = fr_mul(q.x, z1z1);
    Fr s2 = fr_mul(q.y, fr_mul(p.z, z1z1));
    Fr h = fr_sub(u2, p.x);
    GrumpkinPointProjective result;
    result.z = fr_double(fr_mul(p.z, h));
    Fr hh = fr_mul(h, h);
    Fr i = fr_double(fr_double(hh));
    Fr v = fr_mul(p.x, i);
    Fr j = fr_mul(h, i);
    Fr rr = fr_double(fr_sub(s2, p.y));
    result.x = fr_sub(fr_sub(fr_mul(rr, rr), j), fr_double(v));
    result.y = fr_sub(fr_mul(rr, fr_sub(v, result.x)),
                      fr_double(fr_mul(p.y, j)));
    return result;
}

// Full addition: projective + projective
GrumpkinPointProjective grumpkin_point_add(GrumpkinPointProjective p, GrumpkinPointProjective q) {
    if (grumpkin_point_is_identity(p)) return q;
    if (grumpkin_point_is_identity(q)) return p;

    Fr z1z1 = fr_mul(p.z, p.z);
    Fr z2z2 = fr_mul(q.z, q.z);
    Fr u1 = fr_mul(p.x, z2z2);
    Fr u2 = fr_mul(q.x, z1z1);
    Fr s1 = fr_mul(p.y, fr_mul(q.z, z2z2));
    Fr s2 = fr_mul(q.y, fr_mul(p.z, z1z1));

    Fr h = fr_sub(u2, u1);
    Fr rr = fr_double(fr_sub(s2, s1));

    if (fr_is_zero(h)) {
        if (fr_is_zero(rr)) return grumpkin_point_double(p);
        return grumpkin_point_identity();
    }

    GrumpkinPointProjective result;
    result.z = fr_mul(fr_double(fr_mul(p.z, q.z)), h);

    Fr dh = fr_double(h);
    Fr ii = fr_mul(dh, dh);
    Fr v = fr_mul(u1, ii);
    Fr j = fr_mul(h, ii);

    result.x = fr_sub(fr_sub(fr_mul(rr, rr), j), fr_double(v));
    result.y = fr_sub(fr_mul(rr, fr_sub(v, result.x)),
                      fr_double(fr_mul(s1, j)));
    return result;
}

#endif // GRUMPKIN_CURVE_METAL
