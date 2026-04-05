//go:build darwin && arm64

package zkmetal

/*
#include "NeonFieldOps.h"
*/
import "C"
import "unsafe"

// ---------------------------------------------------------------------------
// BN254 Fr (scalar field, 4x64-bit Montgomery form)
// ---------------------------------------------------------------------------

// BN254Fr represents a BN254 scalar field element in Montgomery form.
// Layout: 4 little-endian uint64 limbs.
type BN254Fr [4]uint64

// BN254FrAdd returns a + b mod r.
func BN254FrAdd(a, b *BN254Fr) BN254Fr {
	var r BN254Fr
	C.bn254_fr_add((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BN254FrSub returns a - b mod r.
func BN254FrSub(a, b *BN254Fr) BN254Fr {
	var r BN254Fr
	C.bn254_fr_sub((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BN254FrMul returns a * b mod r (Montgomery multiplication).
func BN254FrMul(a, b *BN254Fr) BN254Fr {
	var r BN254Fr
	C.bn254_fr_mul((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BN254FrNeg returns -a mod r.
func BN254FrNeg(a *BN254Fr) BN254Fr {
	var r BN254Fr
	C.bn254_fr_neg((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BN254FrInv returns a^{-1} mod r via Fermat's little theorem.
func BN254FrInv(a *BN254Fr) BN254Fr {
	var r BN254Fr
	C.bn254_fr_inverse((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BN254FrSqr returns a^2 mod r (optimized squaring).
func BN254FrSqr(a *BN254Fr) BN254Fr {
	var r BN254Fr
	C.bn254_fr_sqr((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BN254FrPow returns a^exp mod r.
func BN254FrPow(a *BN254Fr, exp uint64) BN254Fr {
	var r BN254Fr
	C.bn254_fr_pow((*C.uint64_t)(unsafe.Pointer(&a[0])),
		C.uint64_t(exp),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BN254FrEqual returns true if a == b.
func BN254FrEqual(a, b *BN254Fr) bool {
	return C.bn254_fr_eq((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0]))) == 1
}

// IsZero returns true if a is the zero element.
func (a *BN254Fr) IsZero() bool {
	return a[0] == 0 && a[1] == 0 && a[2] == 0 && a[3] == 0
}

// ---------------------------------------------------------------------------
// BN254 Fr batch operations
// ---------------------------------------------------------------------------

// BN254FrBatchAdd computes result[i] = a[i] + b[i] (multi-threaded for n >= 4096).
func BN254FrBatchAdd(a, b []BN254Fr) []BN254Fr {
	n := len(a)
	if n == 0 {
		return nil
	}
	r := make([]BN254Fr, n)
	C.bn254_fr_batch_add_parallel(
		(*C.uint64_t)(unsafe.Pointer(&r[0])),
		(*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		C.int(n))
	return r
}

// BN254FrBatchSub computes result[i] = a[i] - b[i] (multi-threaded for n >= 4096).
func BN254FrBatchSub(a, b []BN254Fr) []BN254Fr {
	n := len(a)
	if n == 0 {
		return nil
	}
	r := make([]BN254Fr, n)
	C.bn254_fr_batch_sub_parallel(
		(*C.uint64_t)(unsafe.Pointer(&r[0])),
		(*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		C.int(n))
	return r
}

// BN254FrBatchInverse computes result[i] = a[i]^{-1} using Montgomery's trick.
func BN254FrBatchInverse(a []BN254Fr) []BN254Fr {
	n := len(a)
	if n == 0 {
		return nil
	}
	r := make([]BN254Fr, n)
	C.bn254_fr_batch_inverse(
		(*C.uint64_t)(unsafe.Pointer(&a[0])),
		C.int(n),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BN254FrInnerProduct computes sum(a[i] * b[i]).
func BN254FrInnerProduct(a, b []BN254Fr) BN254Fr {
	n := len(a)
	var r BN254Fr
	if n == 0 {
		return r
	}
	C.bn254_fr_inner_product(
		(*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		C.int(n),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// ---------------------------------------------------------------------------
// BN254 Fq (base field, 4x64-bit Montgomery form)
// ---------------------------------------------------------------------------

// BN254Fq represents a BN254 base field element in Montgomery form.
type BN254Fq [4]uint64

// BN254FqSqr returns a^2 mod q.
func BN254FqSqr(a *BN254Fq) BN254Fq {
	var r BN254Fq
	C.bn254_fp_sqr((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BN254FqInv returns a^{-1} mod q.
func BN254FqInv(a *BN254Fq) BN254Fq {
	var r BN254Fq
	C.bn254_fp_inv((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BN254FqSqrt returns the square root of a if it exists.
func BN254FqSqrt(a *BN254Fq) (BN254Fq, bool) {
	var r BN254Fq
	ok := C.bn254_fp_sqrt((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r, ok == 1
}

// ---------------------------------------------------------------------------
// BN254 G1 curve types
// ---------------------------------------------------------------------------

// BN254G1Affine represents a BN254 G1 point in affine coordinates.
// Layout: x[4], y[4] in Montgomery form (8 uint64s total).
type BN254G1Affine struct {
	X, Y BN254Fq
}

// BN254G1Projective represents a BN254 G1 point in Jacobian projective coordinates.
// Layout: x[4], y[4], z[4] in Montgomery form (12 uint64s total).
type BN254G1Projective struct {
	X, Y, Z BN254Fq
}

// BN254G1Add returns p + q in projective coordinates.
func BN254G1Add(p, q *BN254G1Projective) BN254G1Projective {
	var r BN254G1Projective
	C.bn254_point_add((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&q.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BN254G1AddMixed returns projective p + affine q (saves 2 muls + 1 sqr).
func BN254G1AddMixed(p *BN254G1Projective, q *BN254G1Affine) BN254G1Projective {
	var r BN254G1Projective
	C.bn254_point_add_mixed((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&q.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BN254G1ScalarMul returns scalar * p.
// scalar is 8 x uint32 in non-Montgomery integer form, little-endian.
func BN254G1ScalarMul(p *BN254G1Projective, scalar *[8]uint32) BN254G1Projective {
	var r BN254G1Projective
	C.bn254_point_scalar_mul((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint32_t)(unsafe.Pointer(&scalar[0])),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BN254G1ToAffine converts a projective point to affine.
func BN254G1ToAffine(p *BN254G1Projective) BN254G1Affine {
	var r BN254G1Affine
	C.bn254_projective_to_affine((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BN254G1BatchToAffine converts n projective points to affine using Montgomery's batch trick.
func BN254G1BatchToAffine(proj []BN254G1Projective) []BN254G1Affine {
	n := len(proj)
	if n == 0 {
		return nil
	}
	aff := make([]BN254G1Affine, n)
	C.bn254_batch_to_affine((*C.uint64_t)(unsafe.Pointer(&proj[0].X[0])),
		(*C.uint64_t)(unsafe.Pointer(&aff[0].X[0])),
		C.int(n))
	return aff
}

// ---------------------------------------------------------------------------
// BN254 G2 types (Fp2 coordinates)
// ---------------------------------------------------------------------------

// BN254Fp2 represents a BN254 Fp2 element (a0 + a1*u), 8 uint64s.
type BN254Fp2 [8]uint64

// BN254G2Affine represents a BN254 G2 point in affine coordinates.
// Layout: x[8], y[8] in Fp2 Montgomery form (16 uint64s total).
type BN254G2Affine struct {
	X, Y BN254Fp2
}

// ---------------------------------------------------------------------------
// BN254 polynomial operations
// ---------------------------------------------------------------------------

// BN254FrHornerEval evaluates a polynomial at z using Horner's method.
// coeffs[0] + coeffs[1]*z + ... + coeffs[n-1]*z^(n-1).
func BN254FrHornerEval(coeffs []BN254Fr, z *BN254Fr) BN254Fr {
	n := len(coeffs)
	var r BN254Fr
	if n == 0 {
		return r
	}
	C.bn254_fr_horner_eval(
		(*C.uint64_t)(unsafe.Pointer(&coeffs[0])),
		C.int(n),
		(*C.uint64_t)(unsafe.Pointer(&z[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BN254FrSyntheticDiv computes quotient = (p(x) - p(z)) / (x - z).
func BN254FrSyntheticDiv(coeffs []BN254Fr, z *BN254Fr) []BN254Fr {
	n := len(coeffs)
	if n <= 1 {
		return nil
	}
	q := make([]BN254Fr, n-1)
	C.bn254_fr_synthetic_div(
		(*C.uint64_t)(unsafe.Pointer(&coeffs[0])),
		(*C.uint64_t)(unsafe.Pointer(&z[0])),
		C.int(n),
		(*C.uint64_t)(unsafe.Pointer(&q[0])))
	return q
}

// BN254FrEvalAndDiv computes p(z) and (p(x)-p(z))/(x-z) in a single fused pass.
func BN254FrEvalAndDiv(coeffs []BN254Fr, z *BN254Fr) (BN254Fr, []BN254Fr) {
	n := len(coeffs)
	var eval BN254Fr
	if n < 2 {
		return eval, nil
	}
	q := make([]BN254Fr, n-1)
	C.bn254_fr_eval_and_div(
		(*C.uint64_t)(unsafe.Pointer(&coeffs[0])),
		C.int(n),
		(*C.uint64_t)(unsafe.Pointer(&z[0])),
		(*C.uint64_t)(unsafe.Pointer(&eval[0])),
		(*C.uint64_t)(unsafe.Pointer(&q[0])))
	return eval, q
}
