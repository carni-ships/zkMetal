//go:build darwin && arm64

package zkmetal

/*
#include "NeonFieldOps.h"
*/
import "C"
import "unsafe"

// ---------------------------------------------------------------------------
// BLS12-381 Fr (scalar field, 4x64-bit Montgomery form)
// ---------------------------------------------------------------------------

// BLS12381Fr represents a BLS12-381 scalar field element in Montgomery form.
type BLS12381Fr [4]uint64

// BLS12381FrAdd returns a + b mod r.
func BLS12381FrAdd(a, b *BLS12381Fr) BLS12381Fr {
	var r BLS12381Fr
	C.bls12_381_fr_add((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381FrSub returns a - b mod r.
func BLS12381FrSub(a, b *BLS12381Fr) BLS12381Fr {
	var r BLS12381Fr
	C.bls12_381_fr_sub((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381FrMul returns a * b mod r (Montgomery multiplication).
func BLS12381FrMul(a, b *BLS12381Fr) BLS12381Fr {
	var r BLS12381Fr
	C.bls12_381_fr_mul((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381FrNeg returns -a mod r.
func BLS12381FrNeg(a *BLS12381Fr) BLS12381Fr {
	var r BLS12381Fr
	C.bls12_381_fr_neg((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381FrSqr returns a^2 mod r.
func BLS12381FrSqr(a *BLS12381Fr) BLS12381Fr {
	var r BLS12381Fr
	C.bls12_381_fr_sqr((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// ---------------------------------------------------------------------------
// BLS12-381 Fp (base field, 6x64-bit Montgomery form)
// ---------------------------------------------------------------------------

// BLS12381Fp represents a BLS12-381 base field element in Montgomery form.
type BLS12381Fp [6]uint64

// BLS12381FpMul returns a * b mod p.
func BLS12381FpMul(a, b *BLS12381Fp) BLS12381Fp {
	var r BLS12381Fp
	C.bls12_381_fp_mul((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381FpAdd returns a + b mod p.
func BLS12381FpAdd(a, b *BLS12381Fp) BLS12381Fp {
	var r BLS12381Fp
	C.bls12_381_fp_add((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381FpSub returns a - b mod p.
func BLS12381FpSub(a, b *BLS12381Fp) BLS12381Fp {
	var r BLS12381Fp
	C.bls12_381_fp_sub((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381FpNeg returns -a mod p.
func BLS12381FpNeg(a *BLS12381Fp) BLS12381Fp {
	var r BLS12381Fp
	C.bls12_381_fp_neg((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381FpSqr returns a^2 mod p.
func BLS12381FpSqr(a *BLS12381Fp) BLS12381Fp {
	var r BLS12381Fp
	C.bls12_381_fp_sqr((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381FpInv returns a^{-1} mod p.
func BLS12381FpInv(a *BLS12381Fp) BLS12381Fp {
	var r BLS12381Fp
	C.bls12_381_fp_inv_ext((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381FpSqrt returns the square root of a if it exists.
func BLS12381FpSqrt(a *BLS12381Fp) (BLS12381Fp, bool) {
	var r BLS12381Fp
	ok := C.bls12_381_fp_sqrt((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r, ok == 1
}

// ---------------------------------------------------------------------------
// BLS12-381 Fp2 tower (12 uint64s: c0[6] + c1[6]*u)
// ---------------------------------------------------------------------------

// BLS12381Fp2 represents a BLS12-381 Fp2 element.
type BLS12381Fp2 [12]uint64

// BLS12381Fp2Add returns a + b.
func BLS12381Fp2Add(a, b *BLS12381Fp2) BLS12381Fp2 {
	var r BLS12381Fp2
	C.bls12_381_fp2_add((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381Fp2Sub returns a - b.
func BLS12381Fp2Sub(a, b *BLS12381Fp2) BLS12381Fp2 {
	var r BLS12381Fp2
	C.bls12_381_fp2_sub((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381Fp2Mul returns a * b.
func BLS12381Fp2Mul(a, b *BLS12381Fp2) BLS12381Fp2 {
	var r BLS12381Fp2
	C.bls12_381_fp2_mul((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381Fp2Sqr returns a^2.
func BLS12381Fp2Sqr(a *BLS12381Fp2) BLS12381Fp2 {
	var r BLS12381Fp2
	C.bls12_381_fp2_sqr((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381Fp2Inv returns a^{-1}.
func BLS12381Fp2Inv(a *BLS12381Fp2) BLS12381Fp2 {
	var r BLS12381Fp2
	C.bls12_381_fp2_inv((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381Fp2Conj returns the conjugate of a.
func BLS12381Fp2Conj(a *BLS12381Fp2) BLS12381Fp2 {
	var r BLS12381Fp2
	C.bls12_381_fp2_conj((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// ---------------------------------------------------------------------------
// BLS12-381 G1 curve types (6-limb Fp coordinates)
// ---------------------------------------------------------------------------

// BLS12381G1Affine represents a BLS12-381 G1 affine point.
// Layout: x[6], y[6] in Montgomery form (12 uint64s).
type BLS12381G1Affine struct {
	X, Y BLS12381Fp
}

// BLS12381G1Projective represents a BLS12-381 G1 Jacobian projective point.
// Layout: x[6], y[6], z[6] in Montgomery form (18 uint64s).
type BLS12381G1Projective struct {
	X, Y, Z BLS12381Fp
}

// BLS12381G1Add returns p + q.
func BLS12381G1Add(p, q *BLS12381G1Projective) BLS12381G1Projective {
	var r BLS12381G1Projective
	C.bls12_381_g1_point_add((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&q.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BLS12381G1Double returns 2*p.
func BLS12381G1Double(p *BLS12381G1Projective) BLS12381G1Projective {
	var r BLS12381G1Projective
	C.bls12_381_g1_point_double((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BLS12381G1AddMixed returns projective p + affine q.
func BLS12381G1AddMixed(p *BLS12381G1Projective, q *BLS12381G1Affine) BLS12381G1Projective {
	var r BLS12381G1Projective
	C.bls12_381_g1_point_add_mixed((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&q.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BLS12381G1ScalarMul returns scalar * p.
func BLS12381G1ScalarMul(p *BLS12381G1Projective, scalar *BLS12381Fr) BLS12381G1Projective {
	var r BLS12381G1Projective
	C.bls12_381_g1_scalar_mul((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&scalar[0])),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// ---------------------------------------------------------------------------
// BLS12-381 G2 curve types (Fp2 coordinates)
// ---------------------------------------------------------------------------

// BLS12381G2Affine represents a BLS12-381 G2 affine point.
// Layout: x[12], y[12] in Fp2 Montgomery form (24 uint64s).
type BLS12381G2Affine struct {
	X, Y BLS12381Fp2
}

// BLS12381G2Projective represents a BLS12-381 G2 Jacobian projective point.
// Layout: x[12], y[12], z[12] in Fp2 Montgomery form (36 uint64s).
type BLS12381G2Projective struct {
	X, Y, Z BLS12381Fp2
}

// BLS12381G2Add returns p + q.
func BLS12381G2Add(p, q *BLS12381G2Projective) BLS12381G2Projective {
	var r BLS12381G2Projective
	C.bls12_381_g2_point_add((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&q.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BLS12381G2Double returns 2*p.
func BLS12381G2Double(p *BLS12381G2Projective) BLS12381G2Projective {
	var r BLS12381G2Projective
	C.bls12_381_g2_point_double((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BLS12381G2AddMixed returns projective p + affine q.
func BLS12381G2AddMixed(p *BLS12381G2Projective, q *BLS12381G2Affine) BLS12381G2Projective {
	var r BLS12381G2Projective
	C.bls12_381_g2_point_add_mixed((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&q.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BLS12381G2ScalarMul returns scalar * p.
func BLS12381G2ScalarMul(p *BLS12381G2Projective, scalar *BLS12381Fr) BLS12381G2Projective {
	var r BLS12381G2Projective
	C.bls12_381_g2_scalar_mul((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&scalar[0])),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BLS12381HashToG2 hashes a message to G2 using RFC 9380 (SSWU + 3-isogeny).
func BLS12381HashToG2(msg, dst []byte) BLS12381G2Projective {
	var r BLS12381G2Projective
	var msgPtr, dstPtr *C.uint8_t
	if len(msg) > 0 {
		msgPtr = (*C.uint8_t)(unsafe.Pointer(&msg[0]))
	}
	if len(dst) > 0 {
		dstPtr = (*C.uint8_t)(unsafe.Pointer(&dst[0]))
	}
	C.bls12_381_hash_to_g2(msgPtr, C.size_t(len(msg)),
		dstPtr, C.size_t(len(dst)),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BLS12381HashToG2Default hashes a message to G2 using the default DST.
func BLS12381HashToG2Default(msg []byte) BLS12381G2Projective {
	var r BLS12381G2Projective
	var msgPtr *C.uint8_t
	if len(msg) > 0 {
		msgPtr = (*C.uint8_t)(unsafe.Pointer(&msg[0]))
	}
	C.bls12_381_hash_to_g2_default(msgPtr, C.size_t(len(msg)),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BLS12381G2ClearCofactor applies cofactor clearing to a G2 point.
func BLS12381G2ClearCofactor(p *BLS12381G2Projective) BLS12381G2Projective {
	var r BLS12381G2Projective
	C.bls12_381_g2_clear_cofactor((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}
