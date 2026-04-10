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
PointProjective point_double(PointProjective p) {
    if (point_is_identity(p)) return p;

    Fp a = fp_mul_karatsuba(p.x, p.x);
    Fp b = fp_mul_karatsuba(p.y, p.y);
    Fp c = fp_mul_karatsuba(b, b);

    Fp d = fp_sub(fp_mul_karatsuba(fp_add(p.x, b), fp_add(p.x, b)), fp_add(a, c));
    d = fp_double(d);

    Fp e = fp_add(fp_double(a), a); // 3*X^2
    Fp f = fp_mul_karatsuba(e, e);

    PointProjective r;
    r.x = fp_sub(f, fp_double(d));
    r.y = fp_sub(fp_mul_karatsuba(e, fp_sub(d, r.x)), fp_double(fp_double(fp_double(c))));
    Fp yz = fp_add(p.y, p.z);
    r.z = fp_sub(fp_mul_karatsuba(yz, yz), fp_add(b, fp_mul_karatsuba(p.z, p.z)));
    return r;
}

// Mixed addition: projective + affine (with identity/doubling checks)
PointProjective point_add_mixed(PointProjective p, PointAffine q) {
    if (point_is_identity(p)) return point_from_affine(q);

    Fp z1z1 = fp_mul_karatsuba(p.z, p.z);
    Fp u2 = fp_mul_karatsuba(q.x, z1z1);
    Fp h = fp_sub(u2, p.x);

    if (fp_is_zero(h)) {
        Fp s2 = fp_mul_karatsuba(q.y, fp_mul_karatsuba(p.z, z1z1));
        Fp rr = fp_double(fp_sub(s2, p.y));
        if (fp_is_zero(rr)) return point_double(p);
        return point_identity();
    }

    Fp s2 = fp_mul_karatsuba(q.y, fp_mul_karatsuba(p.z, z1z1));
    PointProjective result;
    result.z = fp_double(fp_mul_karatsuba(p.z, h));
    Fp hh = fp_mul_karatsuba(h, h);
    Fp i = fp_double(fp_double(hh));
    Fp v = fp_mul_karatsuba(p.x, i);
    Fp j = fp_mul_karatsuba(h, i);
    Fp rr = fp_double(fp_sub(s2, p.y));
    result.x = fp_sub(fp_sub(fp_mul_karatsuba(rr, rr), j), fp_double(v));
    result.y = fp_sub(fp_mul_karatsuba(rr, fp_sub(v, result.x)),
                      fp_double(fp_mul_karatsuba(p.y, j)));
    return result;
}

// Fast mixed addition: no identity/doubling checks.
// Use only when p is guaranteed non-identity and p != ±q.
// Saves branch overhead in tight MSM bucket reduction loops.
PointProjective point_add_mixed_unsafe(PointProjective p, PointAffine q) {
    Fp z1z1 = fp_mul_karatsuba(p.z, p.z);
    Fp u2 = fp_mul_karatsuba(q.x, z1z1);
    Fp s2 = fp_mul_karatsuba(q.y, fp_mul_karatsuba(p.z, z1z1));
    Fp h = fp_sub(u2, p.x);
    PointProjective result;
    result.z = fp_double(fp_mul_karatsuba(p.z, h));
    Fp hh = fp_mul_karatsuba(h, h);
    Fp i = fp_double(fp_double(hh));
    Fp v = fp_mul_karatsuba(p.x, i);
    Fp j = fp_mul_karatsuba(h, i);
    Fp rr = fp_double(fp_sub(s2, p.y));
    result.x = fp_sub(fp_sub(fp_mul_karatsuba(rr, rr), j), fp_double(v));
    result.y = fp_sub(fp_mul_karatsuba(rr, fp_sub(v, result.x)),
                      fp_double(fp_mul_karatsuba(p.y, j)));
    return result;
}

// Full addition: projective + projective (with identity/doubling checks)
PointProjective point_add(PointProjective p, PointProjective q) {
    if (point_is_identity(p)) return q;
    if (point_is_identity(q)) return p;

    Fp z1z1 = fp_mul_karatsuba(p.z, p.z);
    Fp z2z2 = fp_mul_karatsuba(q.z, q.z);
    Fp u1 = fp_mul_karatsuba(p.x, z2z2);
    Fp u2 = fp_mul_karatsuba(q.x, z1z1);
    Fp s1 = fp_mul_karatsuba(p.y, fp_mul_karatsuba(q.z, z2z2));
    Fp s2 = fp_mul_karatsuba(q.y, fp_mul_karatsuba(p.z, z1z1));

    Fp h = fp_sub(u2, u1);
    Fp rr = fp_double(fp_sub(s2, s1));

    if (fp_is_zero(h)) {
        if (fp_is_zero(rr)) return point_double(p);
        return point_identity();
    }

    PointProjective result;
    result.z = fp_mul_karatsuba(fp_double(fp_mul_karatsuba(p.z, q.z)), h);

    Fp dh = fp_double(h);
    Fp i = fp_mul_karatsuba(dh, dh);
    Fp v = fp_mul_karatsuba(u1, i);
    Fp j = fp_mul_karatsuba(h, i);

    result.x = fp_sub(fp_sub(fp_mul_karatsuba(rr, rr), j), fp_double(v));
    result.y = fp_sub(fp_mul_karatsuba(rr, fp_sub(v, result.x)),
                      fp_double(fp_mul_karatsuba(s1, j)));
    return result;
}

// Fast projective + projective: no identity/doubling checks.
// Use only when both p,q are guaranteed non-identity and p != ±q.
PointProjective point_add_unsafe(PointProjective p, PointProjective q) {
    Fp z1z1 = fp_mul_karatsuba(p.z, p.z);
    Fp z2z2 = fp_mul_karatsuba(q.z, q.z);
    Fp u1 = fp_mul_karatsuba(p.x, z2z2);
    Fp u2 = fp_mul_karatsuba(q.x, z1z1);
    Fp s1 = fp_mul_karatsuba(p.y, fp_mul_karatsuba(q.z, z2z2));
    Fp s2 = fp_mul_karatsuba(q.y, fp_mul_karatsuba(p.z, z1z1));

    Fp h = fp_sub(u2, u1);
    Fp rr = fp_double(fp_sub(s2, s1));

    PointProjective result;
    result.z = fp_mul_karatsuba(fp_double(fp_mul_karatsuba(p.z, q.z)), h);

    Fp dh = fp_double(h);
    Fp i = fp_mul_karatsuba(dh, dh);
    Fp v = fp_mul_karatsuba(u1, i);
    Fp j = fp_mul_karatsuba(h, i);

    result.x = fp_sub(fp_sub(fp_mul_karatsuba(rr, rr), j), fp_double(v));
    result.y = fp_sub(fp_mul_karatsuba(rr, fp_sub(v, result.x)),
                      fp_double(fp_mul_karatsuba(s1, j)));
    return result;
}

#endif // BN254_CURVE_METAL
