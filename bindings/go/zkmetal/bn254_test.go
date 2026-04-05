//go:build darwin && arm64

package zkmetal

import (
	"testing"
	"unsafe"
)

// TestBN254FrZero verifies the zero element is its own identity.
func TestBN254FrZero(t *testing.T) {
	var zero BN254Fr
	if !zero.IsZero() {
		t.Fatal("expected zero element to be zero")
	}
}

// TestBN254FrAddSub verifies a + b - b == a for known values.
func TestBN254FrAddSub(t *testing.T) {
	a := BN254Fr{1, 0, 0, 0}
	b := BN254Fr{2, 0, 0, 0}

	sum := BN254FrAdd(&a, &b)
	diff := BN254FrSub(&sum, &b)

	if !BN254FrEqual(&diff, &a) {
		t.Fatalf("(a + b) - b != a: got %v, want %v", diff, a)
	}
}

// TestBN254FrMulInv verifies a * b * b^{-1} == a.
func TestBN254FrMulInv(t *testing.T) {
	// These are raw limb values; actual Montgomery form values would depend
	// on the field modulus, but we test the algebraic identity.
	a := BN254Fr{7, 0, 0, 0}
	b := BN254Fr{13, 0, 0, 0}

	ab := BN254FrMul(&a, &b)
	bInv := BN254FrInv(&b)
	result := BN254FrMul(&ab, &bInv)

	if !BN254FrEqual(&result, &a) {
		t.Fatalf("a * b * b^{-1} != a: got %v, want %v", result, a)
	}
}

// TestBN254FrNegDouble verifies -(-a) == a.
func TestBN254FrNegDouble(t *testing.T) {
	a := BN254Fr{42, 0, 0, 0}
	neg := BN254FrNeg(&a)
	negNeg := BN254FrNeg(&neg)

	if !BN254FrEqual(&negNeg, &a) {
		t.Fatalf("-(-a) != a: got %v, want %v", negNeg, a)
	}
}

// TestBN254FrAddZero verifies a + 0 == a.
func TestBN254FrAddZero(t *testing.T) {
	a := BN254Fr{99, 0, 0, 0}
	var zero BN254Fr
	sum := BN254FrAdd(&a, &zero)

	if !BN254FrEqual(&sum, &a) {
		t.Fatalf("a + 0 != a: got %v, want %v", sum, a)
	}
}

// TestBN254FrSqrConsistency verifies a^2 == a * a.
func TestBN254FrSqrConsistency(t *testing.T) {
	a := BN254Fr{5, 0, 0, 0}
	sqr := BN254FrSqr(&a)
	mul := BN254FrMul(&a, &a)

	if !BN254FrEqual(&sqr, &mul) {
		t.Fatalf("a^2 != a*a: sqr=%v, mul=%v", sqr, mul)
	}
}

// TestBN254G1Types verifies type sizes are correct for C interop.
func TestBN254G1Types(t *testing.T) {
	var aff BN254G1Affine
	var proj BN254G1Projective

	// Affine: 8 uint64 = 64 bytes
	if got := unsafe.Sizeof(aff); got != 64 {
		t.Fatalf("BN254G1Affine size: got %d, want 64", got)
	}
	// Projective: 12 uint64 = 96 bytes
	if got := unsafe.Sizeof(proj); got != 96 {
		t.Fatalf("BN254G1Projective size: got %d, want 96", got)
	}
}

// TestBLS12381G1Types verifies BLS12-381 type sizes for C interop.
func TestBLS12381G1Types(t *testing.T) {
	var aff BLS12381G1Affine
	var proj BLS12381G1Projective

	// Affine: 12 uint64 = 96 bytes
	if got := unsafe.Sizeof(aff); got != 96 {
		t.Fatalf("BLS12381G1Affine size: got %d, want 96", got)
	}
	// Projective: 18 uint64 = 144 bytes
	if got := unsafe.Sizeof(proj); got != 144 {
		t.Fatalf("BLS12381G1Projective size: got %d, want 144", got)
	}
}

// TestBLS12381G2Types verifies BLS12-381 G2 type sizes for C interop.
func TestBLS12381G2Types(t *testing.T) {
	var aff BLS12381G2Affine
	var proj BLS12381G2Projective

	// G2 Affine: 24 uint64 = 192 bytes
	if got := unsafe.Sizeof(aff); got != 192 {
		t.Fatalf("BLS12381G2Affine size: got %d, want 192", got)
	}
	// G2 Projective: 36 uint64 = 288 bytes
	if got := unsafe.Sizeof(proj); got != 288 {
		t.Fatalf("BLS12381G2Projective size: got %d, want 288", got)
	}
}

// TestBN254GtSize verifies the Gt element size matches C layout.
func TestBN254GtSize(t *testing.T) {
	var gt BN254Gt
	// BN254 Fp12: 48 uint64 = 384 bytes
	if got := unsafe.Sizeof(gt); got != 384 {
		t.Fatalf("BN254Gt size: got %d, want 384", got)
	}
}

// TestBLS12381GtSize verifies the BLS12-381 Gt element size.
func TestBLS12381GtSize(t *testing.T) {
	var gt BLS12381Gt
	// BLS12-381 Fp12: 72 uint64 = 576 bytes
	if got := unsafe.Sizeof(gt); got != 576 {
		t.Fatalf("BLS12381Gt size: got %d, want 576", got)
	}
}
