//go:build darwin && arm64

package zkmetal

/*
#include "zkmetal.h"
*/
import "C"
import "unsafe"

// ---------------------------------------------------------------------------
// GPU FRI Fold (BN254 Fr, Metal-accelerated)
// ---------------------------------------------------------------------------
//
// Single FRI fold round: fold n=2^logN evaluations with challenge beta.
// Output has n/2 elements.

// GPUFRIFold performs one FRI fold round on the Metal GPU.
//
// evals: n = 2^logN field elements (each 32 bytes, Montgomery form).
// logN: log2 of the number of evaluation points (must be >= 1).
// beta: fold challenge (32 bytes, Montgomery form).
// Returns: n/2 folded field elements.
func (e *FRIEngine) GPUFRIFold(evals []byte, logN uint32, beta [32]byte) ([]byte, error) {
	if logN < 1 {
		return nil, ErrInvalidInput
	}
	n := uint32(1) << logN
	if uint32(len(evals)) < n*32 {
		return nil, ErrInvalidInput
	}

	result := make([]byte, (n/2)*32)
	status := C.zkmetal_fri_fold(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&evals[0])),
		C.uint32_t(logN),
		(*C.uint8_t)(unsafe.Pointer(&beta[0])),
		(*C.uint8_t)(unsafe.Pointer(&result[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return result, nil
}

// GPUFRIFoldAuto uses a lazy singleton GPU engine.
func GPUFRIFoldAuto(evals []byte, logN uint32, beta [32]byte) ([]byte, error) {
	if logN < 1 {
		return nil, ErrInvalidInput
	}
	n := uint32(1) << logN
	if uint32(len(evals)) < n*32 {
		return nil, ErrInvalidInput
	}

	result := make([]byte, (n/2)*32)
	status := C.zkmetal_fri_fold_auto(
		(*C.uint8_t)(unsafe.Pointer(&evals[0])),
		C.uint32_t(logN),
		(*C.uint8_t)(unsafe.Pointer(&beta[0])),
		(*C.uint8_t)(unsafe.Pointer(&result[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return result, nil
}

// ---------------------------------------------------------------------------
// Typed GPU FRI Fold variants
// ---------------------------------------------------------------------------

// GPUFRIFoldTyped performs one FRI fold round from typed BN254Fr slices.
// evals must have 2^logN elements. Returns 2^(logN-1) folded elements.
func (e *FRIEngine) GPUFRIFoldTyped(evals []BN254Fr, logN uint32, beta *BN254Fr) ([]BN254Fr, error) {
	if logN < 1 {
		return nil, ErrInvalidInput
	}
	n := 1 << logN
	if len(evals) < n {
		return nil, ErrInvalidInput
	}

	result := make([]BN254Fr, n/2)
	status := C.zkmetal_fri_fold(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&evals[0])),
		C.uint32_t(logN),
		(*C.uint8_t)(unsafe.Pointer(&beta[0])),
		(*C.uint8_t)(unsafe.Pointer(&result[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return result, nil
}

// GPUFRIFoldTypedAuto is the singleton convenience variant.
func GPUFRIFoldTypedAuto(evals []BN254Fr, logN uint32, beta *BN254Fr) ([]BN254Fr, error) {
	if logN < 1 {
		return nil, ErrInvalidInput
	}
	n := 1 << logN
	if len(evals) < n {
		return nil, ErrInvalidInput
	}

	result := make([]BN254Fr, n/2)
	status := C.zkmetal_fri_fold_auto(
		(*C.uint8_t)(unsafe.Pointer(&evals[0])),
		C.uint32_t(logN),
		(*C.uint8_t)(unsafe.Pointer(&beta[0])),
		(*C.uint8_t)(unsafe.Pointer(&result[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return result, nil
}
