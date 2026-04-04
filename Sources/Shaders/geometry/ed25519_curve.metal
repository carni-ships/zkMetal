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

// The curve constant d (in Montgomery form) is passed as a buffer parameter
// from the MSM engine. d = -121665/121666 mod p.

// Point addition using extended coordinates (unified formula)
// For -x^2 + y^2 = 1 + d*x^2*y^2 (a = -1)
// A=X1*X2, B=Y1*Y2, C=d*T1*T2, D=Z1*Z2
// E=(X1+Y1)*(X2+Y2)-A-B, F=D-C, G=D+C, H=B+A (since a=-1, H=B-aA=B+A)
// Cost: 8M + 1D
EdPointExtended ed_point_add(EdPointExtended p, EdPointExtended q, EdFp d_const) {
    EdFp a = ed_mul(p.x, q.x);
    EdFp b = ed_mul(p.y, q.y);
    EdFp c = ed_mul(ed_mul(p.t, q.t), d_const);
    EdFp d_val = ed_mul(p.z, q.z);
    EdFp e = ed_sub(ed_mul(ed_add(p.x, p.y), ed_add(q.x, q.y)), ed_add(a, b));
    EdFp f = ed_sub(d_val, c);
    EdFp g = ed_add(d_val, c);
    EdFp h = ed_add(b, a);  // H = B - a*A = B + A since a = -1

    EdPointExtended r;
    r.x = ed_mul(e, f);
    r.y = ed_mul(g, h);
    r.z = ed_mul(f, g);
    r.t = ed_mul(e, h);
    return r;
}

// Point doubling (optimized, no d_const dependency)
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

// Mixed addition: extended + affine (Z_q = 1, so D = Z1)
EdPointExtended ed_point_add_mixed(EdPointExtended p, EdPointAffine q, EdFp d_const) {
    if (ed_point_is_identity(p)) return ed_point_from_affine(q);

    EdFp a = ed_mul(p.x, q.x);
    EdFp b = ed_mul(p.y, q.y);
    EdFp c = ed_mul(p.t, ed_mul(ed_mul(q.x, q.y), d_const));
    EdFp d_val = p.z;  // Z_q = 1, so D = Z1 * Z_q = Z1
    EdFp e = ed_sub(ed_mul(ed_add(p.x, p.y), ed_add(q.x, q.y)), ed_add(a, b));
    EdFp f = ed_sub(d_val, c);
    EdFp g = ed_add(d_val, c);
    EdFp h = ed_add(b, a);  // H = B - a*A = B + A since a = -1

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
