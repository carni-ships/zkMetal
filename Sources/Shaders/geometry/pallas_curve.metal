// Pallas elliptic curve point operations for Metal GPU
// y^2 = x^3 + 5, Jacobian projective coordinates

#ifndef PALLAS_CURVE_METAL
#define PALLAS_CURVE_METAL

#include "../fields/pallas_fp.metal"

PallasPointProjective pallas_point_identity() {
    PallasPointProjective p;
    p.x = pallas_one();
    p.y = pallas_one();
    p.z = pallas_zero();
    return p;
}

bool pallas_point_is_identity(PallasPointProjective p) {
    return pallas_is_zero(p.z);
}

PallasPointProjective pallas_point_from_affine(PallasPointAffine a) {
    PallasPointProjective p;
    p.x = a.x;
    p.y = a.y;
    p.z = pallas_one();
    return p;
}

// Point doubling: 4M + 6S + 7add (a=0 for Pallas: y^2 = x^3 + 5)
PallasPointProjective pallas_point_double(PallasPointProjective p) {
    if (pallas_point_is_identity(p)) return p;

    PallasFp a = pallas_mul(p.x, p.x);
    PallasFp b = pallas_mul(p.y, p.y);
    PallasFp c = pallas_mul(b, b);

    PallasFp d = pallas_sub(pallas_mul(pallas_add(p.x, b), pallas_add(p.x, b)), pallas_add(a, c));
    d = pallas_double(d);

    PallasFp e = pallas_add(pallas_double(a), a); // 3*X^2
    PallasFp f = pallas_mul(e, e);

    PallasPointProjective r;
    r.x = pallas_sub(f, pallas_double(d));
    r.y = pallas_sub(pallas_mul(e, pallas_sub(d, r.x)), pallas_double(pallas_double(pallas_double(c))));
    PallasFp yz = pallas_add(p.y, p.z);
    r.z = pallas_sub(pallas_mul(yz, yz), pallas_add(b, pallas_mul(p.z, p.z)));
    return r;
}

// Mixed addition: projective + affine
PallasPointProjective pallas_point_add_mixed(PallasPointProjective p, PallasPointAffine q) {
    if (pallas_point_is_identity(p)) return pallas_point_from_affine(q);

    PallasFp z1z1 = pallas_mul(p.z, p.z);
    PallasFp u2 = pallas_mul(q.x, z1z1);
    PallasFp h = pallas_sub(u2, p.x);

    if (pallas_is_zero(h)) {
        PallasFp s2 = pallas_mul(q.y, pallas_mul(p.z, z1z1));
        PallasFp rr = pallas_double(pallas_sub(s2, p.y));
        if (pallas_is_zero(rr)) return pallas_point_double(p);
        return pallas_point_identity();
    }

    PallasFp s2 = pallas_mul(q.y, pallas_mul(p.z, z1z1));
    PallasPointProjective result;
    result.z = pallas_double(pallas_mul(p.z, h));
    PallasFp hh = pallas_mul(h, h);
    PallasFp i = pallas_double(pallas_double(hh));
    PallasFp v = pallas_mul(p.x, i);
    PallasFp j = pallas_mul(h, i);
    PallasFp rr = pallas_double(pallas_sub(s2, p.y));
    result.x = pallas_sub(pallas_sub(pallas_mul(rr, rr), j), pallas_double(v));
    result.y = pallas_sub(pallas_mul(rr, pallas_sub(v, result.x)),
                      pallas_double(pallas_mul(p.y, j)));
    return result;
}

// Fast mixed addition: no identity/doubling checks
PallasPointProjective pallas_point_add_mixed_unsafe(PallasPointProjective p, PallasPointAffine q) {
    PallasFp z1z1 = pallas_mul(p.z, p.z);
    PallasFp u2 = pallas_mul(q.x, z1z1);
    PallasFp s2 = pallas_mul(q.y, pallas_mul(p.z, z1z1));
    PallasFp h = pallas_sub(u2, p.x);
    PallasPointProjective result;
    result.z = pallas_double(pallas_mul(p.z, h));
    PallasFp hh = pallas_mul(h, h);
    PallasFp i = pallas_double(pallas_double(hh));
    PallasFp v = pallas_mul(p.x, i);
    PallasFp j = pallas_mul(h, i);
    PallasFp rr = pallas_double(pallas_sub(s2, p.y));
    result.x = pallas_sub(pallas_sub(pallas_mul(rr, rr), j), pallas_double(v));
    result.y = pallas_sub(pallas_mul(rr, pallas_sub(v, result.x)),
                      pallas_double(pallas_mul(p.y, j)));
    return result;
}

// Full addition: projective + projective
PallasPointProjective pallas_point_add(PallasPointProjective p, PallasPointProjective q) {
    if (pallas_point_is_identity(p)) return q;
    if (pallas_point_is_identity(q)) return p;

    PallasFp z1z1 = pallas_mul(p.z, p.z);
    PallasFp z2z2 = pallas_mul(q.z, q.z);
    PallasFp u1 = pallas_mul(p.x, z2z2);
    PallasFp u2 = pallas_mul(q.x, z1z1);
    PallasFp s1 = pallas_mul(p.y, pallas_mul(q.z, z2z2));
    PallasFp s2 = pallas_mul(q.y, pallas_mul(p.z, z1z1));

    PallasFp h = pallas_sub(u2, u1);
    PallasFp rr = pallas_double(pallas_sub(s2, s1));

    if (pallas_is_zero(h)) {
        if (pallas_is_zero(rr)) return pallas_point_double(p);
        return pallas_point_identity();
    }

    PallasPointProjective result;
    result.z = pallas_mul(pallas_double(pallas_mul(p.z, q.z)), h);

    PallasFp dh = pallas_double(h);
    PallasFp ii = pallas_mul(dh, dh);
    PallasFp v = pallas_mul(u1, ii);
    PallasFp j = pallas_mul(h, ii);

    result.x = pallas_sub(pallas_sub(pallas_mul(rr, rr), j), pallas_double(v));
    result.y = pallas_sub(pallas_mul(rr, pallas_sub(v, result.x)),
                      pallas_double(pallas_mul(s1, j)));
    return result;
}

#endif // PALLAS_CURVE_METAL
