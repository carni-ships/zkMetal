// Vesta elliptic curve point operations for Metal GPU
// y^2 = x^3 + 5, Jacobian projective coordinates

#ifndef VESTA_CURVE_METAL
#define VESTA_CURVE_METAL

#include "../fields/vesta_fp.metal"

VestaPointProjective vesta_point_identity() {
    VestaPointProjective p;
    p.x = vesta_one();
    p.y = vesta_one();
    p.z = vesta_zero();
    return p;
}

bool vesta_point_is_identity(VestaPointProjective p) {
    return vesta_is_zero(p.z);
}

VestaPointProjective vesta_point_from_affine(VestaPointAffine a) {
    VestaPointProjective p;
    p.x = a.x;
    p.y = a.y;
    p.z = vesta_one();
    return p;
}

// Point doubling: 4M + 6S + 7add (a=0 for Vesta: y^2 = x^3 + 5)
VestaPointProjective vesta_point_double(VestaPointProjective p) {
    if (vesta_point_is_identity(p)) return p;

    VestaFp a = vesta_mul(p.x, p.x);
    VestaFp b = vesta_mul(p.y, p.y);
    VestaFp c = vesta_mul(b, b);

    VestaFp d = vesta_sub(vesta_mul(vesta_add(p.x, b), vesta_add(p.x, b)), vesta_add(a, c));
    d = vesta_double(d);

    VestaFp e = vesta_add(vesta_double(a), a); // 3*X^2
    VestaFp f = vesta_mul(e, e);

    VestaPointProjective r;
    r.x = vesta_sub(f, vesta_double(d));
    r.y = vesta_sub(vesta_mul(e, vesta_sub(d, r.x)), vesta_double(vesta_double(vesta_double(c))));
    VestaFp yz = vesta_add(p.y, p.z);
    r.z = vesta_sub(vesta_mul(yz, yz), vesta_add(b, vesta_mul(p.z, p.z)));
    return r;
}

// Mixed addition: projective + affine
VestaPointProjective vesta_point_add_mixed(VestaPointProjective p, VestaPointAffine q) {
    if (vesta_point_is_identity(p)) return vesta_point_from_affine(q);

    VestaFp z1z1 = vesta_mul(p.z, p.z);
    VestaFp u2 = vesta_mul(q.x, z1z1);
    VestaFp h = vesta_sub(u2, p.x);

    if (vesta_is_zero(h)) {
        VestaFp s2 = vesta_mul(q.y, vesta_mul(p.z, z1z1));
        VestaFp rr = vesta_double(vesta_sub(s2, p.y));
        if (vesta_is_zero(rr)) return vesta_point_double(p);
        return vesta_point_identity();
    }

    VestaFp s2 = vesta_mul(q.y, vesta_mul(p.z, z1z1));
    VestaPointProjective result;
    result.z = vesta_double(vesta_mul(p.z, h));
    VestaFp hh = vesta_mul(h, h);
    VestaFp i = vesta_double(vesta_double(hh));
    VestaFp v = vesta_mul(p.x, i);
    VestaFp j = vesta_mul(h, i);
    VestaFp rr = vesta_double(vesta_sub(s2, p.y));
    result.x = vesta_sub(vesta_sub(vesta_mul(rr, rr), j), vesta_double(v));
    result.y = vesta_sub(vesta_mul(rr, vesta_sub(v, result.x)),
                      vesta_double(vesta_mul(p.y, j)));
    return result;
}

// Fast mixed addition: no identity/doubling checks
VestaPointProjective vesta_point_add_mixed_unsafe(VestaPointProjective p, VestaPointAffine q) {
    VestaFp z1z1 = vesta_mul(p.z, p.z);
    VestaFp u2 = vesta_mul(q.x, z1z1);
    VestaFp s2 = vesta_mul(q.y, vesta_mul(p.z, z1z1));
    VestaFp h = vesta_sub(u2, p.x);
    VestaPointProjective result;
    result.z = vesta_double(vesta_mul(p.z, h));
    VestaFp hh = vesta_mul(h, h);
    VestaFp i = vesta_double(vesta_double(hh));
    VestaFp v = vesta_mul(p.x, i);
    VestaFp j = vesta_mul(h, i);
    VestaFp rr = vesta_double(vesta_sub(s2, p.y));
    result.x = vesta_sub(vesta_sub(vesta_mul(rr, rr), j), vesta_double(v));
    result.y = vesta_sub(vesta_mul(rr, vesta_sub(v, result.x)),
                      vesta_double(vesta_mul(p.y, j)));
    return result;
}

// Full addition: projective + projective
VestaPointProjective vesta_point_add(VestaPointProjective p, VestaPointProjective q) {
    if (vesta_point_is_identity(p)) return q;
    if (vesta_point_is_identity(q)) return p;

    VestaFp z1z1 = vesta_mul(p.z, p.z);
    VestaFp z2z2 = vesta_mul(q.z, q.z);
    VestaFp u1 = vesta_mul(p.x, z2z2);
    VestaFp u2 = vesta_mul(q.x, z1z1);
    VestaFp s1 = vesta_mul(p.y, vesta_mul(q.z, z2z2));
    VestaFp s2 = vesta_mul(q.y, vesta_mul(p.z, z1z1));

    VestaFp h = vesta_sub(u2, u1);
    VestaFp rr = vesta_double(vesta_sub(s2, s1));

    if (vesta_is_zero(h)) {
        if (vesta_is_zero(rr)) return vesta_point_double(p);
        return vesta_point_identity();
    }

    VestaPointProjective result;
    result.z = vesta_mul(vesta_double(vesta_mul(p.z, q.z)), h);

    VestaFp dh = vesta_double(h);
    VestaFp ii = vesta_mul(dh, dh);
    VestaFp v = vesta_mul(u1, ii);
    VestaFp j = vesta_mul(h, ii);

    result.x = vesta_sub(vesta_sub(vesta_mul(rr, rr), j), vesta_double(v));
    result.y = vesta_sub(vesta_mul(rr, vesta_sub(v, result.x)),
                      vesta_double(vesta_mul(s1, j)));
    return result;
}

#endif // VESTA_CURVE_METAL
