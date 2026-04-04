// secp256k1 elliptic curve point operations for Metal GPU
// y^2 = x^3 + 7, Jacobian projective coordinates

#ifndef SECP256K1_CURVE_METAL
#define SECP256K1_CURVE_METAL

#include "../fields/secp256k1_fp.metal"

SecpPointProjective secp_point_identity() {
    SecpPointProjective p;
    p.x = secp_one();
    p.y = secp_one();
    p.z = secp_zero();
    return p;
}

bool secp_point_is_identity(SecpPointProjective p) {
    return secp_is_zero(p.z);
}

SecpPointProjective secp_point_from_affine(SecpPointAffine a) {
    SecpPointProjective p;
    p.x = a.x;
    p.y = a.y;
    p.z = secp_one();
    return p;
}

// Point doubling: 4M + 6S + 7add (a=0 for secp256k1)
// Uses secp_mul(a,a) instead of secp_sqr to reduce register pressure
SecpPointProjective secp_point_double(SecpPointProjective p) {
    if (secp_point_is_identity(p)) return p;

    SecpFp a = secp_mul(p.x, p.x);
    SecpFp b = secp_mul(p.y, p.y);
    SecpFp c = secp_mul(b, b);

    SecpFp d = secp_sub(secp_mul(secp_add(p.x, b), secp_add(p.x, b)), secp_add(a, c));
    d = secp_double(d);

    SecpFp e = secp_add(secp_double(a), a); // 3*X^2 (a=0 for secp256k1)
    SecpFp f = secp_mul(e, e);

    SecpPointProjective r;
    r.x = secp_sub(f, secp_double(d));
    r.y = secp_sub(secp_mul(e, secp_sub(d, r.x)), secp_double(secp_double(secp_double(c))));
    SecpFp yz = secp_add(p.y, p.z);
    r.z = secp_sub(secp_mul(yz, yz), secp_add(b, secp_mul(p.z, p.z)));
    return r;
}

// Mixed addition: projective + affine
SecpPointProjective secp_point_add_mixed(SecpPointProjective p, SecpPointAffine q) {
    if (secp_point_is_identity(p)) return secp_point_from_affine(q);

    SecpFp z1z1 = secp_mul(p.z, p.z);
    SecpFp u2 = secp_mul(q.x, z1z1);
    SecpFp h = secp_sub(u2, p.x);

    if (secp_is_zero(h)) {
        SecpFp s2 = secp_mul(q.y, secp_mul(p.z, z1z1));
        SecpFp rr = secp_double(secp_sub(s2, p.y));
        if (secp_is_zero(rr)) return secp_point_double(p);
        return secp_point_identity();
    }

    SecpFp s2 = secp_mul(q.y, secp_mul(p.z, z1z1));
    SecpPointProjective result;
    result.z = secp_double(secp_mul(p.z, h));
    SecpFp hh = secp_mul(h, h);
    SecpFp i = secp_double(secp_double(hh));
    SecpFp v = secp_mul(p.x, i);
    SecpFp j = secp_mul(h, i);
    SecpFp rr = secp_double(secp_sub(s2, p.y));
    result.x = secp_sub(secp_sub(secp_mul(rr, rr), j), secp_double(v));
    result.y = secp_sub(secp_mul(rr, secp_sub(v, result.x)),
                        secp_double(secp_mul(p.y, j)));
    return result;
}

// Fast mixed addition: no identity/doubling checks.
// Use only when p is guaranteed non-identity and p != ±q.
// Saves branch overhead in tight MSM bucket reduction loops.
SecpPointProjective secp_point_add_mixed_unsafe(SecpPointProjective p, SecpPointAffine q) {
    SecpFp z1z1 = secp_mul(p.z, p.z);
    SecpFp u2 = secp_mul(q.x, z1z1);
    SecpFp s2 = secp_mul(q.y, secp_mul(p.z, z1z1));
    SecpFp h = secp_sub(u2, p.x);
    SecpPointProjective result;
    result.z = secp_double(secp_mul(p.z, h));
    SecpFp hh = secp_mul(h, h);
    SecpFp i = secp_double(secp_double(hh));
    SecpFp v = secp_mul(p.x, i);
    SecpFp j = secp_mul(h, i);
    SecpFp rr = secp_double(secp_sub(s2, p.y));
    result.x = secp_sub(secp_sub(secp_mul(rr, rr), j), secp_double(v));
    result.y = secp_sub(secp_mul(rr, secp_sub(v, result.x)),
                        secp_double(secp_mul(p.y, j)));
    return result;
}

// Full addition: projective + projective
SecpPointProjective secp_point_add(SecpPointProjective p, SecpPointProjective q) {
    if (secp_point_is_identity(p)) return q;
    if (secp_point_is_identity(q)) return p;

    SecpFp z1z1 = secp_mul(p.z, p.z);
    SecpFp z2z2 = secp_mul(q.z, q.z);
    SecpFp u1 = secp_mul(p.x, z2z2);
    SecpFp u2 = secp_mul(q.x, z1z1);
    SecpFp s1 = secp_mul(p.y, secp_mul(q.z, z2z2));
    SecpFp s2 = secp_mul(q.y, secp_mul(p.z, z1z1));

    SecpFp h = secp_sub(u2, u1);
    SecpFp rr = secp_double(secp_sub(s2, s1));

    if (secp_is_zero(h)) {
        if (secp_is_zero(rr)) return secp_point_double(p);
        return secp_point_identity();
    }

    SecpPointProjective result;
    result.z = secp_mul(secp_double(secp_mul(p.z, q.z)), h);

    SecpFp dh = secp_double(h);
    SecpFp i = secp_mul(dh, dh);
    SecpFp v = secp_mul(u1, i);
    SecpFp j = secp_mul(h, i);

    result.x = secp_sub(secp_sub(secp_mul(rr, rr), j), secp_double(v));
    result.y = secp_sub(secp_mul(rr, secp_sub(v, result.x)),
                        secp_double(secp_mul(s1, j)));
    return result;
}

// Fast projective addition: no identity checks.
// Use only when both p and q are guaranteed non-identity.
SecpPointProjective secp_point_add_unsafe(SecpPointProjective p, SecpPointProjective q) {
    SecpFp z1z1 = secp_mul(p.z, p.z);
    SecpFp z2z2 = secp_mul(q.z, q.z);
    SecpFp u1 = secp_mul(p.x, z2z2);
    SecpFp u2 = secp_mul(q.x, z1z1);
    SecpFp s1 = secp_mul(p.y, secp_mul(q.z, z2z2));
    SecpFp s2 = secp_mul(q.y, secp_mul(p.z, z1z1));

    SecpFp h = secp_sub(u2, u1);
    SecpFp rr = secp_double(secp_sub(s2, s1));

    if (secp_is_zero(h)) {
        if (secp_is_zero(rr)) return secp_point_double(p);
        return secp_point_identity();
    }

    SecpPointProjective result;
    result.z = secp_mul(secp_double(secp_mul(p.z, q.z)), h);

    SecpFp dh = secp_double(h);
    SecpFp i = secp_mul(dh, dh);
    SecpFp v = secp_mul(u1, i);
    SecpFp j = secp_mul(h, i);

    result.x = secp_sub(secp_sub(secp_mul(rr, rr), j), secp_double(v));
    result.y = secp_sub(secp_mul(rr, secp_sub(v, result.x)),
                        secp_double(secp_mul(s1, j)));
    return result;
}

#endif // SECP256K1_CURVE_METAL
