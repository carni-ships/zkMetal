//go:build darwin && arm64

package zkmetal

/*
#include "zkmetal.h"
*/
import "C"
import "unsafe"

// ---------------------------------------------------------------------------
// GPU Batch Pairing (BN254, Metal-accelerated Miller loops)
// ---------------------------------------------------------------------------
//
// For CPU-only pairing, use BN254Pairing / BN254PairingCheck in pairing.go.
// GPU pairing uses parallel Metal-accelerated Miller loops + CPU final exp.

// GPUBatchPairing computes product of e(g1[i], g2[i]) for i in 0..nPairs.
//
// g1Points: nPairs affine G1 points, each 64 bytes (x,y in Montgomery form).
// g2Points: nPairs affine G2 points, each 128 bytes (x0,x1,y0,y1 Montgomery Fp2).
// Returns: Fp12 element (384 bytes = 12 x 32B, Montgomery form).
func (e *PairingEngine) GPUBatchPairing(g1Points, g2Points []byte, nPairs uint32) ([]byte, error) {
	if nPairs == 0 {
		return nil, nil
	}
	if uint32(len(g1Points)) < nPairs*64 {
		return nil, ErrInvalidInput
	}
	if uint32(len(g2Points)) < nPairs*128 {
		return nil, ErrInvalidInput
	}

	result := make([]byte, 384)
	status := C.zkmetal_bn254_batch_pairing(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&g1Points[0])),
		(*C.uint8_t)(unsafe.Pointer(&g2Points[0])),
		C.uint32_t(nPairs),
		(*C.uint8_t)(unsafe.Pointer(&result[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return result, nil
}

// GPUBatchPairingAuto uses a lazy singleton GPU engine.
func GPUBatchPairingAuto(g1Points, g2Points []byte, nPairs uint32) ([]byte, error) {
	if nPairs == 0 {
		return nil, nil
	}
	if uint32(len(g1Points)) < nPairs*64 {
		return nil, ErrInvalidInput
	}
	if uint32(len(g2Points)) < nPairs*128 {
		return nil, ErrInvalidInput
	}

	result := make([]byte, 384)
	status := C.zkmetal_bn254_batch_pairing_auto(
		(*C.uint8_t)(unsafe.Pointer(&g1Points[0])),
		(*C.uint8_t)(unsafe.Pointer(&g2Points[0])),
		C.uint32_t(nPairs),
		(*C.uint8_t)(unsafe.Pointer(&result[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return result, nil
}

// ---------------------------------------------------------------------------
// GPU Pairing Check
// ---------------------------------------------------------------------------

// GPUPairingCheck verifies that product of e(g1[i], g2[i]) == 1 (Gt identity).
// Returns nil if the check passes, ErrInvalidInput if it fails.
func (e *PairingEngine) GPUPairingCheck(g1Points, g2Points []byte, nPairs uint32) error {
	if nPairs == 0 {
		return nil
	}
	if uint32(len(g1Points)) < nPairs*64 {
		return ErrInvalidInput
	}
	if uint32(len(g2Points)) < nPairs*128 {
		return ErrInvalidInput
	}

	status := C.zkmetal_bn254_pairing_check(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&g1Points[0])),
		(*C.uint8_t)(unsafe.Pointer(&g2Points[0])),
		C.uint32_t(nPairs),
	)
	return statusToError(status)
}

// GPUPairingCheckAuto uses a lazy singleton GPU engine.
func GPUPairingCheckAuto(g1Points, g2Points []byte, nPairs uint32) error {
	if nPairs == 0 {
		return nil
	}
	if uint32(len(g1Points)) < nPairs*64 {
		return ErrInvalidInput
	}
	if uint32(len(g2Points)) < nPairs*128 {
		return ErrInvalidInput
	}

	status := C.zkmetal_bn254_pairing_check_auto(
		(*C.uint8_t)(unsafe.Pointer(&g1Points[0])),
		(*C.uint8_t)(unsafe.Pointer(&g2Points[0])),
		C.uint32_t(nPairs),
	)
	return statusToError(status)
}

// ---------------------------------------------------------------------------
// Typed GPU Pairing variants
// ---------------------------------------------------------------------------

// GPUBatchPairingTyped computes batch pairing from typed Go structs.
// Returns the BN254Gt (Fp12) result element.
func (e *PairingEngine) GPUBatchPairingTyped(g1 []BN254G1Affine, g2 []BN254G2Affine) (BN254Gt, error) {
	var result BN254Gt
	n := len(g1)
	if n == 0 {
		return result, nil
	}
	if len(g2) != n {
		return result, ErrInvalidInput
	}

	status := C.zkmetal_bn254_batch_pairing(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&g1[0].X[0])),
		(*C.uint8_t)(unsafe.Pointer(&g2[0].X[0])),
		C.uint32_t(n),
		(*C.uint8_t)(unsafe.Pointer(&result[0])),
	)
	if err := statusToError(status); err != nil {
		return result, err
	}
	return result, nil
}

// GPUPairingCheckTyped verifies product of e(g1[i], g2[i]) == 1.
// Returns nil if check passes, error otherwise.
func (e *PairingEngine) GPUPairingCheckTyped(g1 []BN254G1Affine, g2 []BN254G2Affine) error {
	n := len(g1)
	if n == 0 {
		return nil
	}
	if len(g2) != n {
		return ErrInvalidInput
	}

	status := C.zkmetal_bn254_pairing_check(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&g1[0].X[0])),
		(*C.uint8_t)(unsafe.Pointer(&g2[0].X[0])),
		C.uint32_t(n),
	)
	return statusToError(status)
}
