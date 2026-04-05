//go:build darwin && arm64

package zkmetal

/*
#include "zkmetal.h"
*/
import "C"
import "unsafe"

// ---------------------------------------------------------------------------
// GPU Poseidon2 Hash (BN254 Fr, Metal-accelerated)
// ---------------------------------------------------------------------------
//
// For CPU-only Poseidon2 (ARM NEON), use the functions in hash.go.
// GPU Poseidon2 is faster for large batch sizes (thousands of pairs).

// GPUPoseidon2HashPairs computes batch Poseidon2 2-to-1 compression hashes on the GPU.
// input: 2*nPairs field elements (each 32 bytes, Montgomery form).
// output: nPairs field elements (each 32 bytes, Montgomery form).
func (e *Poseidon2Engine) GPUPoseidon2HashPairs(input []byte, nPairs uint32) ([]byte, error) {
	if nPairs == 0 {
		return nil, nil
	}
	if uint32(len(input)) < nPairs*2*32 {
		return nil, ErrInvalidInput
	}

	output := make([]byte, nPairs*32)
	status := C.zkmetal_bn254_poseidon2_hash_pairs(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&input[0])),
		C.uint32_t(nPairs),
		(*C.uint8_t)(unsafe.Pointer(&output[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return output, nil
}

// GPUPoseidon2HashPairsAuto uses a lazy singleton GPU engine.
func GPUPoseidon2HashPairsAuto(input []byte, nPairs uint32) ([]byte, error) {
	if nPairs == 0 {
		return nil, nil
	}
	if uint32(len(input)) < nPairs*2*32 {
		return nil, ErrInvalidInput
	}

	output := make([]byte, nPairs*32)
	status := C.zkmetal_bn254_poseidon2_hash_pairs_auto(
		(*C.uint8_t)(unsafe.Pointer(&input[0])),
		C.uint32_t(nPairs),
		(*C.uint8_t)(unsafe.Pointer(&output[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return output, nil
}

// ---------------------------------------------------------------------------
// Typed Poseidon2 GPU variants (using BN254Fr slices)
// ---------------------------------------------------------------------------

// GPUPoseidon2HashPairsTyped hashes pairs of BN254Fr elements on the GPU.
// input must have 2*nPairs elements. Returns nPairs hash results.
func (e *Poseidon2Engine) GPUPoseidon2HashPairsTyped(input []BN254Fr, nPairs int) ([]BN254Fr, error) {
	if nPairs == 0 {
		return nil, nil
	}
	if len(input) < nPairs*2 {
		return nil, ErrInvalidInput
	}

	output := make([]BN254Fr, nPairs)
	status := C.zkmetal_bn254_poseidon2_hash_pairs(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&input[0])),
		C.uint32_t(nPairs),
		(*C.uint8_t)(unsafe.Pointer(&output[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return output, nil
}

// GPUPoseidon2HashPairsTypedAuto is the singleton convenience variant.
func GPUPoseidon2HashPairsTypedAuto(input []BN254Fr, nPairs int) ([]BN254Fr, error) {
	if nPairs == 0 {
		return nil, nil
	}
	if len(input) < nPairs*2 {
		return nil, ErrInvalidInput
	}

	output := make([]BN254Fr, nPairs)
	status := C.zkmetal_bn254_poseidon2_hash_pairs_auto(
		(*C.uint8_t)(unsafe.Pointer(&input[0])),
		C.uint32_t(nPairs),
		(*C.uint8_t)(unsafe.Pointer(&output[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return output, nil
}

// ---------------------------------------------------------------------------
// GPU Keccak-256 Hash (Metal-accelerated)
// ---------------------------------------------------------------------------
//
// For CPU-only Keccak-256 (ARM NEON), use Keccak256 in hash.go.
// GPU Keccak is faster for large batches of 64-byte inputs.

// GPUKeccak256Hash computes batch Keccak-256 hashes of 64-byte inputs on the GPU.
// input: nInputs * 64 bytes (each input is 64 bytes).
// Returns: nInputs * 32 bytes (each output is a 32-byte hash).
func (e *KeccakEngine) GPUKeccak256Hash(input []byte, nInputs uint32) ([]byte, error) {
	if nInputs == 0 {
		return nil, nil
	}
	if uint32(len(input)) < nInputs*64 {
		return nil, ErrInvalidInput
	}

	output := make([]byte, nInputs*32)
	status := C.zkmetal_keccak256_hash(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&input[0])),
		C.uint32_t(nInputs),
		(*C.uint8_t)(unsafe.Pointer(&output[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return output, nil
}

// GPUKeccak256HashAuto uses a lazy singleton GPU engine.
func GPUKeccak256HashAuto(input []byte, nInputs uint32) ([]byte, error) {
	if nInputs == 0 {
		return nil, nil
	}
	if uint32(len(input)) < nInputs*64 {
		return nil, ErrInvalidInput
	}

	output := make([]byte, nInputs*32)
	status := C.zkmetal_keccak256_hash_auto(
		(*C.uint8_t)(unsafe.Pointer(&input[0])),
		C.uint32_t(nInputs),
		(*C.uint8_t)(unsafe.Pointer(&output[0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return output, nil
}

// ---------------------------------------------------------------------------
// Typed Keccak-256 GPU variants
// ---------------------------------------------------------------------------

// GPUKeccak256HashTyped hashes pairs of 32-byte inputs on the GPU.
// Each pair (a[i], b[i]) is concatenated to form a 64-byte input.
func (e *KeccakEngine) GPUKeccak256HashTyped(inputs [][64]byte) ([][32]byte, error) {
	n := len(inputs)
	if n == 0 {
		return nil, nil
	}

	outputs := make([][32]byte, n)
	status := C.zkmetal_keccak256_hash(
		e.handle,
		(*C.uint8_t)(unsafe.Pointer(&inputs[0][0])),
		C.uint32_t(n),
		(*C.uint8_t)(unsafe.Pointer(&outputs[0][0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return outputs, nil
}

// GPUKeccak256HashTypedAuto is the singleton convenience variant.
func GPUKeccak256HashTypedAuto(inputs [][64]byte) ([][32]byte, error) {
	n := len(inputs)
	if n == 0 {
		return nil, nil
	}

	outputs := make([][32]byte, n)
	status := C.zkmetal_keccak256_hash_auto(
		(*C.uint8_t)(unsafe.Pointer(&inputs[0][0])),
		C.uint32_t(n),
		(*C.uint8_t)(unsafe.Pointer(&outputs[0][0])),
	)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	return outputs, nil
}
