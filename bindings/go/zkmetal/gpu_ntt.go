//go:build darwin && arm64

package zkmetal

/*
#include "zkmetal.h"
*/
import "C"
import "unsafe"

// ---------------------------------------------------------------------------
// GPU NTT — Metal-accelerated Number Theoretic Transform (BN254 Fr)
// ---------------------------------------------------------------------------
//
// For CPU-only NTT via ARM NEON, use BN254FrNTT/BN254FrINTT in ntt.go.
// The GPU variants are faster for large transforms (n >= 2^16 or so).
//
// Elements are 32 bytes each in Montgomery form (8x u32 LE).
// Data is transformed in-place. n = 2^logN elements.

// GPUNTT performs a forward NTT on BN254 Fr elements in-place using the Metal GPU.
// data must contain exactly 2^logN elements, each 32 bytes in Montgomery form.
func (e *NTTEngine) GPUNTT(data []byte, logN uint32) error {
	if len(data) == 0 {
		return nil
	}
	n := uint32(1) << logN
	if uint32(len(data)) < n*32 {
		return ErrInvalidInput
	}

	status := C.zkmetal_bn254_ntt(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.uint32_t(logN),
	)
	return statusToError(status)
}

// GPUINTT performs an inverse NTT on BN254 Fr elements in-place using the Metal GPU.
func (e *NTTEngine) GPUINTT(data []byte, logN uint32) error {
	if len(data) == 0 {
		return nil
	}
	n := uint32(1) << logN
	if uint32(len(data)) < n*32 {
		return ErrInvalidInput
	}

	status := C.zkmetal_bn254_intt(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.uint32_t(logN),
	)
	return statusToError(status)
}

// GPUNTTAuto performs a forward NTT using a lazy singleton GPU engine.
func GPUNTTAuto(data []byte, logN uint32) error {
	if len(data) == 0 {
		return nil
	}
	n := uint32(1) << logN
	if uint32(len(data)) < n*32 {
		return ErrInvalidInput
	}

	status := C.zkmetal_bn254_ntt_auto(
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.uint32_t(logN),
	)
	return statusToError(status)
}

// GPUINTTAuto performs an inverse NTT using a lazy singleton GPU engine.
func GPUINTTAuto(data []byte, logN uint32) error {
	if len(data) == 0 {
		return nil
	}
	n := uint32(1) << logN
	if uint32(len(data)) < n*32 {
		return ErrInvalidInput
	}

	status := C.zkmetal_bn254_intt_auto(
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.uint32_t(logN),
	)
	return statusToError(status)
}

// ---------------------------------------------------------------------------
// Typed GPU NTT variants (using BN254Fr slices)
// ---------------------------------------------------------------------------

// GPUNTTTyped performs forward NTT on a slice of BN254Fr elements in-place.
// The slice must have exactly 2^logN elements.
func (e *NTTEngine) GPUNTTTyped(data []BN254Fr, logN uint32) error {
	n := len(data)
	if n == 0 {
		return nil
	}
	expected := 1 << logN
	if n < expected {
		return ErrInvalidInput
	}

	status := C.zkmetal_bn254_ntt(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.uint32_t(logN),
	)
	return statusToError(status)
}

// GPUINTTTyped performs inverse NTT on a slice of BN254Fr elements in-place.
func (e *NTTEngine) GPUINTTTyped(data []BN254Fr, logN uint32) error {
	n := len(data)
	if n == 0 {
		return nil
	}
	expected := 1 << logN
	if n < expected {
		return ErrInvalidInput
	}

	status := C.zkmetal_bn254_intt(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.uint32_t(logN),
	)
	return statusToError(status)
}

// GPUNTTTypedAuto performs forward NTT on BN254Fr elements using a singleton engine.
func GPUNTTTypedAuto(data []BN254Fr, logN uint32) error {
	if len(data) == 0 {
		return nil
	}
	status := C.zkmetal_bn254_ntt_auto(
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.uint32_t(logN),
	)
	return statusToError(status)
}

// GPUINTTTypedAuto performs inverse NTT on BN254Fr elements using a singleton engine.
func GPUINTTTypedAuto(data []BN254Fr, logN uint32) error {
	if len(data) == 0 {
		return nil
	}
	status := C.zkmetal_bn254_intt_auto(
		(*C.uint8_t)(unsafe.Pointer(&data[0])),
		C.uint32_t(logN),
	)
	return statusToError(status)
}
