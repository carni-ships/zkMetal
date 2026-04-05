//go:build darwin && arm64

package zkmetal

/*
#include "NeonFieldOps.h"
*/
import "C"
import "unsafe"

// BN254FrNTT performs a forward NTT on BN254 Fr elements in-place.
// data must have exactly 2^logN elements.
func BN254FrNTT(data []BN254Fr, logN int) {
	if len(data) == 0 {
		return
	}
	C.bn254_fr_ntt((*C.uint64_t)(unsafe.Pointer(&data[0])), C.int(logN))
}

// BN254FrINTT performs an inverse NTT on BN254 Fr elements in-place.
func BN254FrINTT(data []BN254Fr, logN int) {
	if len(data) == 0 {
		return
	}
	C.bn254_fr_intt((*C.uint64_t)(unsafe.Pointer(&data[0])), C.int(logN))
}

// BLS12377FrNTT performs a forward NTT on BLS12-377 Fr elements in-place.
func BLS12377FrNTT(data [][4]uint64, logN int) {
	if len(data) == 0 {
		return
	}
	C.bls12_377_fr_ntt((*C.uint64_t)(unsafe.Pointer(&data[0][0])), C.int(logN))
}

// BLS12377FrINTT performs an inverse NTT on BLS12-377 Fr elements in-place.
func BLS12377FrINTT(data [][4]uint64, logN int) {
	if len(data) == 0 {
		return
	}
	C.bls12_377_fr_intt((*C.uint64_t)(unsafe.Pointer(&data[0][0])), C.int(logN))
}

// Stark252NTT performs a forward NTT on Stark252 field elements in-place.
func Stark252NTT(data [][4]uint64, logN int) {
	if len(data) == 0 {
		return
	}
	C.stark252_ntt((*C.uint64_t)(unsafe.Pointer(&data[0][0])), C.int(logN))
}

// Stark252INTT performs an inverse NTT on Stark252 field elements in-place.
func Stark252INTT(data [][4]uint64, logN int) {
	if len(data) == 0 {
		return
	}
	C.stark252_intt((*C.uint64_t)(unsafe.Pointer(&data[0][0])), C.int(logN))
}

// BabyBearNTT performs a forward NTT on BabyBear elements using ARM NEON.
func BabyBearNTT(data []uint32, logN int) {
	if len(data) == 0 {
		return
	}
	C.babybear_ntt_neon((*C.uint32_t)(unsafe.Pointer(&data[0])), C.int(logN))
}

// BabyBearINTT performs an inverse NTT on BabyBear elements.
func BabyBearINTT(data []uint32, logN int) {
	if len(data) == 0 {
		return
	}
	C.babybear_intt_neon((*C.uint32_t)(unsafe.Pointer(&data[0])), C.int(logN))
}

// GoldilocksNTT performs a forward NTT on Goldilocks elements.
func GoldilocksNTT(data []uint64, logN int) {
	if len(data) == 0 {
		return
	}
	C.goldilocks_ntt((*C.uint64_t)(unsafe.Pointer(&data[0])), C.int(logN))
}

// GoldilocksINTT performs an inverse NTT on Goldilocks elements.
func GoldilocksINTT(data []uint64, logN int) {
	if len(data) == 0 {
		return
	}
	C.goldilocks_intt((*C.uint64_t)(unsafe.Pointer(&data[0])), C.int(logN))
}

// GoldilocksNTTNeon performs a forward NTT using ARM NEON vectorized butterflies.
func GoldilocksNTTNeon(data []uint64, logN int) {
	if len(data) == 0 {
		return
	}
	C.goldilocks_ntt_neon((*C.uint64_t)(unsafe.Pointer(&data[0])), C.int(logN))
}

// GoldilocksINTTNeon performs an inverse NTT using ARM NEON vectorized butterflies.
func GoldilocksINTTNeon(data []uint64, logN int) {
	if len(data) == 0 {
		return
	}
	C.goldilocks_intt_neon((*C.uint64_t)(unsafe.Pointer(&data[0])), C.int(logN))
}
