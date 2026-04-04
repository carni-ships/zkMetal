// Ed25519 elliptic curve point operations for Metal GPU
// Twisted Edwards: -x^2 + y^2 = 1 + d*x^2*y^2
// Extended coordinates: (X, Y, Z, T) where x = X/Z, y = Y/Z, T = XY/Z

#ifndef ED25519_CURVE_METAL
#define ED25519_CURVE_METAL

#include "../fields/ed25519_fp.metal"

EdPointExtended ed_point_identity() {
    EdPointExtended p;
    p.x = ed_zero();
    p.y = ed_one();
    p.z = ed_one();
    p.t = ed_zero();
    return p;
}

bool ed_point_is_identity(EdPointExtended p) {
    // Identity: X == 0 and Y == Z
    if (!ed_is_zero(p.x)) return false;
    // Check Y == Z: subtract and check zero
    uint borrow;
    EdFp diff = ed_sub_raw(p.y, p.z, borrow);
    return ed_is_zero(diff) && borrow == 0;
}

EdPointExtended ed_point_from_affine(EdPointAffine a) {
    EdPointExtended p;
    p.x = a.x;
    p.y = a.y;
    p.z = ed_one();
    p.t = ed_mul(a.x, a.y);
    return p;
}

// Pre-computed 2*d constant (in Montgomery form)
// This will be computed at initialization by the MSM engine and passed as a constant
// For now, we compute d = -121665/121666 and 2*d inline
// d_mont is passed via buffer in practice; here we use a function.
// In the MSM kernels, 2*d will be a constant buffer parameter.

// Point addition using extended coordinates (unified formula)
// For -x^2 + y^2 = 1 + d*x^2*y^2 (a = -1)
// Cost: 8M + 1D
EdPointExtended ed_point_add(EdPointExtended p, EdPointExtended q, EdFp d2) {
    EdFp a = ed_mul(p.x, q.x);
    EdFp b = ed_mul(p.y, q.y);
    EdFp c = ed_mul(ed_mul(p.t, q.t), d2);
    EdFp d_val = ed_mul(p.z, q.z);
    EdFp dd = ed_double(d_val);
    EdFp e = ed_sub(ed_mul(ed_add(p.x, p.y), ed_add(q.x, q.y)), ed_add(a, b));
    EdFp f = ed_sub(dd, c);
    EdFp g = ed_add(dd, c);
    // a_coeff = -1, so h = b + a (= b - (-a) = y1y2 + x1x2 since -x^2 term)
    EdFp h = ed_add(b, a);

    EdPointExtended r;
    r.x = ed_mul(e, f);
    r.y = ed_mul(g, h);
    r.z = ed_mul(f, g);
    r.t = ed_mul(e, h);
    return r;
}

// Point doubling (optimized, no d2 dependency)
// For -x^2 + y^2 = 1 + d*x^2*y^2 (a = -1)
EdPointExtended ed_point_double(EdPointExtended p) {
    EdFp a = ed_sqr(p.x);
    EdFp b = ed_sqr(p.y);
    EdFp c = ed_double(ed_sqr(p.z));
    EdFp d_val = ed_neg(a);  // D = -A (since curve a = -1)
    EdFp e = ed_sub(ed_sqr(ed_add(p.x, p.y)), ed_add(a, b));
    EdFp g = ed_add(d_val, b);
    EdFp f = ed_sub(g, c);
    EdFp h = ed_sub(d_val, b);

    EdPointExtended r;
    r.x = ed_mul(e, f);
    r.y = ed_mul(g, h);
    r.z = ed_mul(f, g);
    r.t = ed_mul(e, h);
    return r;
}

// Mixed addition: extended + affine
EdPointExtended ed_point_add_mixed(EdPointExtended p, EdPointAffine q, EdFp d2) {
    if (ed_point_is_identity(p)) return ed_point_from_affine(q);

    // Extended + affine (Z_q = 1)
    EdFp a = ed_mul(p.x, q.x);
    EdFp b = ed_mul(p.y, q.y);
    EdFp c = ed_mul(p.t, ed_mul(ed_mul(q.x, q.y), d2));
    EdFp dd = ed_double(p.z);
    EdFp e = ed_sub(ed_mul(ed_add(p.x, p.y), ed_add(q.x, q.y)), ed_add(a, b));
    EdFp f = ed_sub(dd, c);
    EdFp g = ed_add(dd, c);
    EdFp h = ed_add(b, a);

    EdPointExtended r;
    r.x = ed_mul(e, f);
    r.y = ed_mul(g, h);
    r.z = ed_mul(f, g);
    r.t = ed_mul(e, h);
    return r;
}

// Negate a point: -(x,y,z,t) = (-x,y,z,-t)
EdPointExtended ed_point_neg(EdPointExtended p) {
    EdPointExtended r;
    r.x = ed_neg(p.x);
    r.y = p.y;
    r.z = p.z;
    r.t = ed_neg(p.t);
    return r;
}

#endif // ED25519_CURVE_METAL
