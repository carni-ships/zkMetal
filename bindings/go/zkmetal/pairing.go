//go:build darwin && arm64

package zkmetal

/*
#include "NeonFieldOps.h"
*/
import "C"
import "unsafe"

// ---------------------------------------------------------------------------
// BN254 Pairing (Fp12 = 48 x uint64)
// ---------------------------------------------------------------------------

// BN254Gt represents a BN254 Fp12 element (target group of the pairing).
type BN254Gt [48]uint64

// BN254MillerLoop computes the Miller loop e(P, Q) without final exponentiation.
// P is a G1 affine point (8 uint64), Q is a G2 affine point (16 uint64).
func BN254MillerLoop(p *BN254G1Affine, q *BN254G2Affine) BN254Gt {
	var r BN254Gt
	C.bn254_miller_loop((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&q.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BN254FinalExp computes the final exponentiation on a Miller loop result.
func BN254FinalExp(f *BN254Gt) BN254Gt {
	var r BN254Gt
	C.bn254_final_exp((*C.uint64_t)(unsafe.Pointer(&f[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BN254Pairing computes the full optimal ate pairing e(P, Q).
func BN254Pairing(p *BN254G1Affine, q *BN254G2Affine) BN254Gt {
	var r BN254Gt
	C.bn254_pairing((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&q.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BN254PairingPair groups a G1 and G2 point for batch pairing checks.
type BN254PairingPair struct {
	G1 BN254G1Affine
	G2 BN254G2Affine
}

// BN254PairingCheck verifies that product(e(G1[i], G2[i])) == 1.
// Returns true if the pairing check passes.
// pairs is a flat array: each pair is G1Affine (8 uint64) followed by G2Affine (16 uint64).
func BN254PairingCheck(pairs []BN254PairingPair) bool {
	n := len(pairs)
	if n == 0 {
		return true
	}
	return C.bn254_pairing_check((*C.uint64_t)(unsafe.Pointer(&pairs[0].G1.X[0])),
		C.int(n)) == 1
}

// ---------------------------------------------------------------------------
// BLS12-381 Pairing (Fp12 = 72 x uint64)
// ---------------------------------------------------------------------------

// BLS12381Gt represents a BLS12-381 Fp12 element (target group).
type BLS12381Gt [72]uint64

// BLS12381MillerLoop computes the Miller loop e(P, Q) without final exponentiation.
func BLS12381MillerLoop(p *BLS12381G1Affine, q *BLS12381G2Affine) BLS12381Gt {
	var r BLS12381Gt
	C.bls12_381_miller_loop((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&q.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381FinalExp computes the final exponentiation on a Miller loop result.
func BLS12381FinalExp(f *BLS12381Gt) BLS12381Gt {
	var r BLS12381Gt
	C.bls12_381_final_exp((*C.uint64_t)(unsafe.Pointer(&f[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381Pairing computes the full optimal ate pairing e(P, Q).
func BLS12381Pairing(p *BLS12381G1Affine, q *BLS12381G2Affine) BLS12381Gt {
	var r BLS12381Gt
	C.bls12_381_pairing((*C.uint64_t)(unsafe.Pointer(&p.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&q.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// BLS12381PairingPair groups a G1 and G2 point for batch pairing checks.
type BLS12381PairingPair struct {
	G1 BLS12381G1Affine
	G2 BLS12381G2Affine
}

// BLS12381PairingCheck verifies that product(e(G1[i], G2[i])) == 1.
// Returns true if the pairing check passes.
func BLS12381PairingCheck(pairs []BLS12381PairingPair) bool {
	n := len(pairs)
	if n == 0 {
		return true
	}
	return C.bls12_381_pairing_check((*C.uint64_t)(unsafe.Pointer(&pairs[0].G1.X[0])),
		C.int(n)) == 1
}
