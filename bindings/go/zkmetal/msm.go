//go:build darwin && arm64

package zkmetal

/*
#include "NeonFieldOps.h"
*/
import "C"
import "unsafe"

// ---------------------------------------------------------------------------
// BN254 MSM (Pippenger multi-scalar multiplication)
// ---------------------------------------------------------------------------

// BN254G1MSM computes sum(scalars[i] * points[i]) using Pippenger's algorithm.
// Points are affine (8 uint64 each). Scalars are 8 x uint32 little-endian.
func BN254G1MSM(points []BN254G1Affine, scalars [][8]uint32) BN254G1Projective {
	n := len(points)
	if n == 0 {
		return BN254G1Projective{}
	}
	var r BN254G1Projective
	C.bn254_pippenger_msm(
		(*C.uint64_t)(unsafe.Pointer(&points[0].X[0])),
		(*C.uint32_t)(unsafe.Pointer(&scalars[0][0])),
		C.int(n),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BN254G1MSMProjective computes MSM from projective points (optimal for small n).
func BN254G1MSMProjective(points []BN254G1Projective, scalars [][8]uint32) BN254G1Projective {
	n := len(points)
	if n == 0 {
		return BN254G1Projective{}
	}
	var r BN254G1Projective
	C.bn254_msm_projective(
		(*C.uint64_t)(unsafe.Pointer(&points[0].X[0])),
		(*C.uint32_t)(unsafe.Pointer(&scalars[0][0])),
		C.int(n),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// BN254DualMSMProjective computes two MSMs with a shared thread pool.
func BN254DualMSMProjective(
	points1 []BN254G1Projective, scalars1 [][8]uint32,
	points2 []BN254G1Projective, scalars2 [][8]uint32,
) (BN254G1Projective, BN254G1Projective) {
	n1, n2 := len(points1), len(points2)
	var r1, r2 BN254G1Projective
	if n1 == 0 && n2 == 0 {
		return r1, r2
	}
	C.bn254_dual_msm_projective(
		(*C.uint64_t)(unsafe.Pointer(&points1[0].X[0])),
		(*C.uint32_t)(unsafe.Pointer(&scalars1[0][0])),
		C.int(n1),
		(*C.uint64_t)(unsafe.Pointer(&points2[0].X[0])),
		(*C.uint32_t)(unsafe.Pointer(&scalars2[0][0])),
		C.int(n2),
		(*C.uint64_t)(unsafe.Pointer(&r1.X[0])),
		(*C.uint64_t)(unsafe.Pointer(&r2.X[0])))
	return r1, r2
}

// ---------------------------------------------------------------------------
// BLS12-381 MSM
// ---------------------------------------------------------------------------

// BLS12381G1MSM computes sum(scalars[i] * points[i]) using Pippenger's algorithm.
func BLS12381G1MSM(points []BLS12381G1Affine, scalars [][8]uint32) BLS12381G1Projective {
	n := len(points)
	if n == 0 {
		return BLS12381G1Projective{}
	}
	var r BLS12381G1Projective
	C.bls12_381_g1_pippenger_msm(
		(*C.uint64_t)(unsafe.Pointer(&points[0].X[0])),
		(*C.uint32_t)(unsafe.Pointer(&scalars[0][0])),
		C.int(n),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// ---------------------------------------------------------------------------
// Grumpkin MSM
// ---------------------------------------------------------------------------

// GrumpkinAffine represents a Grumpkin affine point (over BN254 Fr).
type GrumpkinAffine struct {
	X, Y BN254Fr
}

// GrumpkinProjective represents a Grumpkin projective point.
type GrumpkinProjective struct {
	X, Y, Z BN254Fr
}

// GrumpkinMSM computes Pippenger MSM on the Grumpkin curve.
func GrumpkinMSM(points []GrumpkinAffine, scalars [][8]uint32) GrumpkinProjective {
	n := len(points)
	if n == 0 {
		return GrumpkinProjective{}
	}
	var r GrumpkinProjective
	C.grumpkin_pippenger_msm(
		(*C.uint64_t)(unsafe.Pointer(&points[0].X[0])),
		(*C.uint32_t)(unsafe.Pointer(&scalars[0][0])),
		C.int(n),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// ---------------------------------------------------------------------------
// secp256k1 MSM
// ---------------------------------------------------------------------------

// Secp256k1Affine represents a secp256k1 affine point.
type Secp256k1Affine struct {
	X, Y [4]uint64
}

// Secp256k1Projective represents a secp256k1 projective point.
type Secp256k1Projective struct {
	X, Y, Z [4]uint64
}

// Secp256k1MSM computes Pippenger MSM on secp256k1.
func Secp256k1MSM(points []Secp256k1Affine, scalars [][8]uint32) Secp256k1Projective {
	n := len(points)
	if n == 0 {
		return Secp256k1Projective{}
	}
	var r Secp256k1Projective
	C.secp256k1_pippenger_msm(
		(*C.uint64_t)(unsafe.Pointer(&points[0].X[0])),
		(*C.uint32_t)(unsafe.Pointer(&scalars[0][0])),
		C.int(n),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// ---------------------------------------------------------------------------
// Pallas / Vesta MSM
// ---------------------------------------------------------------------------

// PallasAffine represents a Pallas affine point.
type PallasAffine struct {
	X, Y [4]uint64
}

// PallasProjective represents a Pallas projective point.
type PallasProjective struct {
	X, Y, Z [4]uint64
}

// PallasMSM computes Pippenger MSM on Pallas.
func PallasMSM(points []PallasAffine, scalars [][8]uint32) PallasProjective {
	n := len(points)
	if n == 0 {
		return PallasProjective{}
	}
	var r PallasProjective
	C.pallas_pippenger_msm(
		(*C.uint64_t)(unsafe.Pointer(&points[0].X[0])),
		(*C.uint32_t)(unsafe.Pointer(&scalars[0][0])),
		C.int(n),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// VestaAffine represents a Vesta affine point.
type VestaAffine struct {
	X, Y [4]uint64
}

// VestaProjective represents a Vesta projective point.
type VestaProjective struct {
	X, Y, Z [4]uint64
}

// VestaMSM computes Pippenger MSM on Vesta.
func VestaMSM(points []VestaAffine, scalars [][8]uint32) VestaProjective {
	n := len(points)
	if n == 0 {
		return VestaProjective{}
	}
	var r VestaProjective
	C.vesta_pippenger_msm(
		(*C.uint64_t)(unsafe.Pointer(&points[0].X[0])),
		(*C.uint32_t)(unsafe.Pointer(&scalars[0][0])),
		C.int(n),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}

// ---------------------------------------------------------------------------
// BGMW fixed-base MSM
// ---------------------------------------------------------------------------

// BGMWTable holds precomputed lookup tables for fixed-base MSM.
type BGMWTable struct {
	data       []uint64
	n          int
	windowBits int
}

// BGMWPrecompute creates a lookup table for fixed-base MSM.
func BGMWPrecompute(generators []BN254G1Affine, windowBits int) *BGMWTable {
	n := len(generators)
	if n == 0 {
		return nil
	}
	numWindows := (256 + windowBits - 1) / windowBits
	tableSize := n * numWindows * ((1 << windowBits) - 1) * 8
	data := make([]uint64, tableSize)
	C.bgmw_precompute(
		(*C.uint64_t)(unsafe.Pointer(&generators[0].X[0])),
		C.int(n),
		C.int(windowBits),
		(*C.uint64_t)(unsafe.Pointer(&data[0])))
	return &BGMWTable{data: data, n: n, windowBits: windowBits}
}

// Eval computes the MSM using precomputed tables (additions only).
func (t *BGMWTable) Eval(scalars [][8]uint32) BN254G1Projective {
	var r BN254G1Projective
	if t == nil || len(scalars) == 0 {
		return r
	}
	C.bgmw_msm(
		(*C.uint64_t)(unsafe.Pointer(&t.data[0])),
		C.int(t.n),
		C.int(t.windowBits),
		(*C.uint32_t)(unsafe.Pointer(&scalars[0][0])),
		(*C.uint64_t)(unsafe.Pointer(&r.X[0])))
	return r
}
