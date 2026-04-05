//go:build darwin && arm64

package zkmetal

/*
#include "zkmetal.h"
*/
import "C"
import "unsafe"

// ---------------------------------------------------------------------------
// GPU MSM — Metal-accelerated multi-scalar multiplication (BN254 G1)
// ---------------------------------------------------------------------------
//
// These functions use the zkMetal Metal GPU backend for MSM computation.
// For CPU-only Pippenger MSM, use the BN254G1MSM function in msm.go.
//
// The GPU FFI uses byte-oriented buffers:
//   - Points: 64 bytes each (x: 32B, y: 32B, Montgomery form, 8x u32 LE)
//   - Scalars: 32 bytes each (8x u32 LE, standard form, NOT Montgomery)
//   - Result: projective X, Y, Z each 32 bytes

// GPUMSM computes sum(scalars[i] * points[i]) on the Metal GPU.
// Uses the engine-based API for repeated calls.
//
// points: n affine points, each 64 bytes (x,y in Montgomery form).
// scalars: n scalars, each 32 bytes (standard form, little-endian).
// Returns the projective result as (X, Y, Z) each 32 bytes.
func (e *MSMEngine) GPUMSM(points, scalars []byte, nPoints uint32) (x, y, z [32]byte, err error) {
	if nPoints == 0 {
		return x, y, z, nil
	}
	if uint32(len(points)) < nPoints*64 {
		return x, y, z, ErrInvalidInput
	}
	if uint32(len(scalars)) < nPoints*32 {
		return x, y, z, ErrInvalidInput
	}

	status := C.zkmetal_bn254_msm(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&points[0])),
		(*C.uint8_t)(unsafe.Pointer(&scalars[0])),
		C.uint32_t(nPoints),
		(*C.uint8_t)(unsafe.Pointer(&x[0])),
		(*C.uint8_t)(unsafe.Pointer(&y[0])),
		(*C.uint8_t)(unsafe.Pointer(&z[0])),
	)
	err = statusToError(status)
	return
}

// GPUMSMAuto computes MSM using a lazy singleton GPU engine.
// Simpler API — no engine handle needed. Engine created on first call.
func GPUMSMAuto(points, scalars []byte, nPoints uint32) (x, y, z [32]byte, err error) {
	if nPoints == 0 {
		return x, y, z, nil
	}
	if uint32(len(points)) < nPoints*64 {
		return x, y, z, ErrInvalidInput
	}
	if uint32(len(scalars)) < nPoints*32 {
		return x, y, z, ErrInvalidInput
	}

	status := C.zkmetal_bn254_msm_auto(
		(*C.uint8_t)(unsafe.Pointer(&points[0])),
		(*C.uint8_t)(unsafe.Pointer(&scalars[0])),
		C.uint32_t(nPoints),
		(*C.uint8_t)(unsafe.Pointer(&x[0])),
		(*C.uint8_t)(unsafe.Pointer(&y[0])),
		(*C.uint8_t)(unsafe.Pointer(&z[0])),
	)
	err = statusToError(status)
	return
}

// ---------------------------------------------------------------------------
// GPU MSM with typed Go structs
// ---------------------------------------------------------------------------

// GPUMSMTyped computes MSM from typed Go affine points and scalar arrays.
// Points are BN254G1Affine (8 uint64 = 64 bytes). Scalars are [8]uint32 = 32 bytes.
// Returns a BN254G1Projective result.
func (e *MSMEngine) GPUMSMTyped(points []BN254G1Affine, scalars [][8]uint32) (BN254G1Projective, error) {
	var result BN254G1Projective
	n := len(points)
	if n == 0 {
		return result, nil
	}
	if len(scalars) != n {
		return result, ErrInvalidInput
	}

	// BN254G1Affine is 64 bytes (8 x uint64), BN254G1Projective is 96 bytes (12 x uint64).
	// The C FFI expects uint8_t* pointers with 32-byte Montgomery fields,
	// but the memory layout is identical since Go structs are packed.
	var rx, ry, rz [32]byte
	status := C.zkmetal_bn254_msm(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&points[0].X[0])),
		(*C.uint8_t)(unsafe.Pointer(&scalars[0][0])),
		C.uint32_t(n),
		(*C.uint8_t)(unsafe.Pointer(&rx[0])),
		(*C.uint8_t)(unsafe.Pointer(&ry[0])),
		(*C.uint8_t)(unsafe.Pointer(&rz[0])),
	)
	if err := statusToError(status); err != nil {
		return result, err
	}

	// Copy 32-byte results into the [4]uint64 fields.
	result.X = *(*BN254Fq)(unsafe.Pointer(&rx[0]))
	result.Y = *(*BN254Fq)(unsafe.Pointer(&ry[0]))
	result.Z = *(*BN254Fq)(unsafe.Pointer(&rz[0]))
	return result, nil
}

// GPUMSMTypedAuto is the convenience version of GPUMSMTyped using a singleton engine.
func GPUMSMTypedAuto(points []BN254G1Affine, scalars [][8]uint32) (BN254G1Projective, error) {
	var result BN254G1Projective
	n := len(points)
	if n == 0 {
		return result, nil
	}
	if len(scalars) != n {
		return result, ErrInvalidInput
	}

	var rx, ry, rz [32]byte
	status := C.zkmetal_bn254_msm_auto(
		(*C.uint8_t)(unsafe.Pointer(&points[0].X[0])),
		(*C.uint8_t)(unsafe.Pointer(&scalars[0][0])),
		C.uint32_t(n),
		(*C.uint8_t)(unsafe.Pointer(&rx[0])),
		(*C.uint8_t)(unsafe.Pointer(&ry[0])),
		(*C.uint8_t)(unsafe.Pointer(&rz[0])),
	)
	if err := statusToError(status); err != nil {
		return result, err
	}

	result.X = *(*BN254Fq)(unsafe.Pointer(&rx[0]))
	result.Y = *(*BN254Fq)(unsafe.Pointer(&ry[0]))
	result.Z = *(*BN254Fq)(unsafe.Pointer(&rz[0]))
	return result, nil
}

// ---------------------------------------------------------------------------
// Small-scalar MSM variants (GPU)
// ---------------------------------------------------------------------------

// GPUMSMU8 computes MSM with 1-byte scalars. More efficient for small scalar ranges.
func (e *MSMEngine) GPUMSMU8(points []byte, scalars []byte, nPoints uint32) (x, y, z [32]byte, err error) {
	if nPoints == 0 {
		return x, y, z, nil
	}
	status := C.zkmetal_bn254_msm_u8(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&points[0])),
		(*C.uint8_t)(unsafe.Pointer(&scalars[0])),
		C.uint32_t(nPoints),
		(*C.uint8_t)(unsafe.Pointer(&x[0])),
		(*C.uint8_t)(unsafe.Pointer(&y[0])),
		(*C.uint8_t)(unsafe.Pointer(&z[0])),
	)
	err = statusToError(status)
	return
}

// GPUMSMU16 computes MSM with 2-byte scalars (little-endian).
func (e *MSMEngine) GPUMSMU16(points []byte, scalars []byte, nPoints uint32) (x, y, z [32]byte, err error) {
	if nPoints == 0 {
		return x, y, z, nil
	}
	status := C.zkmetal_bn254_msm_u16(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&points[0])),
		(*C.uint8_t)(unsafe.Pointer(&scalars[0])),
		C.uint32_t(nPoints),
		(*C.uint8_t)(unsafe.Pointer(&x[0])),
		(*C.uint8_t)(unsafe.Pointer(&y[0])),
		(*C.uint8_t)(unsafe.Pointer(&z[0])),
	)
	err = statusToError(status)
	return
}

// GPUMSMU32 computes MSM with 4-byte scalars (little-endian).
func (e *MSMEngine) GPUMSMU32(points []byte, scalars []byte, nPoints uint32) (x, y, z [32]byte, err error) {
	if nPoints == 0 {
		return x, y, z, nil
	}
	status := C.zkmetal_bn254_msm_u32(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&points[0])),
		(*C.uint8_t)(unsafe.Pointer(&scalars[0])),
		C.uint32_t(nPoints),
		(*C.uint8_t)(unsafe.Pointer(&x[0])),
		(*C.uint8_t)(unsafe.Pointer(&y[0])),
		(*C.uint8_t)(unsafe.Pointer(&z[0])),
	)
	err = statusToError(status)
	return
}

// ---------------------------------------------------------------------------
// Batch MSM (GPU)
// ---------------------------------------------------------------------------

// GPUMSMBatch computes multiple independent MSMs in one GPU dispatch.
// Amortizes Swift/Metal dispatch overhead for many small MSMs.
//
// allPoints: concatenated affine points for all MSMs (64 bytes each).
// allScalars: concatenated 32-byte scalars for all MSMs.
// counts: number of points in each MSM.
// Returns: one projective result (96 bytes: X,Y,Z each 32 bytes) per MSM.
func (e *MSMEngine) GPUMSMBatch(allPoints, allScalars []byte, counts []uint32) ([]byte, error) {
	nMSMs := len(counts)
	if nMSMs == 0 {
		return nil, nil
	}

	results := make([]byte, nMSMs*96)
	status := C.zkmetal_bn254_msm_batch(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&allPoints[0])),
		(*C.uint8_t)(unsafe.Pointer(&allScalars[0])),
		(*C.uint32_t)(unsafe.Pointer(&counts[0])),
		C.uint32_t(nMSMs),
		(*C.uint8_t)(unsafe.Pointer(&results[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return results, nil
}

// GPUMSMBatchAuto is the convenience version of GPUMSMBatch.
func GPUMSMBatchAuto(allPoints, allScalars []byte, counts []uint32) ([]byte, error) {
	nMSMs := len(counts)
	if nMSMs == 0 {
		return nil, nil
	}

	results := make([]byte, nMSMs*96)
	status := C.zkmetal_bn254_msm_batch_auto(
		(*C.uint8_t)(unsafe.Pointer(&allPoints[0])),
		(*C.uint8_t)(unsafe.Pointer(&allScalars[0])),
		(*C.uint32_t)(unsafe.Pointer(&counts[0])),
		C.uint32_t(nMSMs),
		(*C.uint8_t)(unsafe.Pointer(&results[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return results, nil
}
