//go:build darwin && arm64

// GPU engine lifecycle management for zkMetal Metal-accelerated operations.
//
// The zkMetal FFI provides two API styles:
//   - Engine-based: create/destroy handles, pass to compute functions.
//     Best for repeated calls (avoids re-initialization overhead).
//   - Auto (convenience): functions use a lazy singleton engine.
//     Simpler API, engine created on first call, persists for process lifetime.
//
// This file provides Go wrappers for both styles with idiomatic error handling.

package zkmetal

/*
#cgo CFLAGS: -I../../../Sources/zkMetal-ffi/include
#cgo LDFLAGS: -L../../../bindings/lib -lzkmetal_ffi -framework Metal -framework Foundation -framework CoreGraphics
#include "zkmetal.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

var (
	// ErrNoGPU indicates no Metal GPU is available on this system.
	ErrNoGPU = errors.New("zkmetal: no Metal GPU available")

	// ErrInvalidInput indicates invalid parameters were passed to a function.
	ErrInvalidInput = errors.New("zkmetal: invalid input")

	// ErrGPU indicates a Metal GPU execution error.
	ErrGPU = errors.New("zkmetal: GPU error")

	// ErrAllocFailed indicates a memory allocation failure.
	ErrAllocFailed = errors.New("zkmetal: allocation failed")
)

// statusToError converts a C ZkMetalStatus code to a Go error.
func statusToError(status C.ZkMetalStatus) error {
	switch status {
	case C.ZKMETAL_SUCCESS:
		return nil
	case C.ZKMETAL_ERR_NO_GPU:
		return ErrNoGPU
	case C.ZKMETAL_ERR_INVALID_INPUT:
		return ErrInvalidInput
	case C.ZKMETAL_ERR_GPU_ERROR:
		return ErrGPU
	case C.ZKMETAL_ERR_ALLOC_FAILED:
		return ErrAllocFailed
	default:
		return fmt.Errorf("zkmetal: unknown error code %d", int(status))
	}
}

// ---------------------------------------------------------------------------
// Capability queries
// ---------------------------------------------------------------------------

// GPUAvailable returns true if a Metal GPU is available on this system.
func GPUAvailable() bool {
	return C.zkmetal_gpu_available() == 1
}

// Version returns the zkMetal library version string.
func Version() string {
	return C.GoString(C.zkmetal_version())
}

// SetShaderDir sets the path to the Shaders directory containing Metal shader
// sources. Call this before creating any engines if the binary is not run from
// the zkMetal source tree. Pass empty string to clear.
func SetShaderDir(path string) {
	if path == "" {
		C.zkmetal_set_shader_dir(nil)
		return
	}
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	C.zkmetal_set_shader_dir(cpath)
}

// ---------------------------------------------------------------------------
// MSM Engine
// ---------------------------------------------------------------------------

// MSMEngine wraps an opaque GPU MSM engine handle.
// Use NewMSMEngine to create, and Close to release resources.
type MSMEngine struct {
	handle C.ZkMetalMSMEngine
}

// NewMSMEngine creates a new GPU-accelerated MSM engine for BN254 G1.
// The caller must call Close when done to release GPU resources.
func NewMSMEngine() (*MSMEngine, error) {
	var handle C.ZkMetalMSMEngine
	status := C.zkmetal_msm_engine_create(&handle)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	eng := &MSMEngine{handle: handle}
	runtime.SetFinalizer(eng, func(e *MSMEngine) { e.Close() })
	return eng, nil
}

// Close releases the GPU resources held by this engine.
func (e *MSMEngine) Close() {
	if e.handle != nil {
		C.zkmetal_msm_engine_destroy(e.handle)
		e.handle = nil
	}
}

// ---------------------------------------------------------------------------
// NTT Engine
// ---------------------------------------------------------------------------

// NTTEngine wraps an opaque GPU NTT engine handle.
type NTTEngine struct {
	handle C.ZkMetalNTTEngine
}

// NewNTTEngine creates a new GPU-accelerated NTT engine for BN254 Fr.
func NewNTTEngine() (*NTTEngine, error) {
	var handle C.ZkMetalNTTEngine
	status := C.zkmetal_ntt_engine_create(&handle)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	eng := &NTTEngine{handle: handle}
	runtime.SetFinalizer(eng, func(e *NTTEngine) { e.Close() })
	return eng, nil
}

// Close releases the GPU resources held by this engine.
func (e *NTTEngine) Close() {
	if e.handle != nil {
		C.zkmetal_ntt_engine_destroy(e.handle)
		e.handle = nil
	}
}

// ---------------------------------------------------------------------------
// Poseidon2 Engine
// ---------------------------------------------------------------------------

// Poseidon2Engine wraps an opaque GPU Poseidon2 engine handle.
type Poseidon2Engine struct {
	handle C.ZkMetalPoseidon2Engine
}

// NewPoseidon2Engine creates a new GPU-accelerated Poseidon2 engine.
func NewPoseidon2Engine() (*Poseidon2Engine, error) {
	var handle C.ZkMetalPoseidon2Engine
	status := C.zkmetal_poseidon2_engine_create(&handle)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	eng := &Poseidon2Engine{handle: handle}
	runtime.SetFinalizer(eng, func(e *Poseidon2Engine) { e.Close() })
	return eng, nil
}

// Close releases the GPU resources held by this engine.
func (e *Poseidon2Engine) Close() {
	if e.handle != nil {
		C.zkmetal_poseidon2_engine_destroy(e.handle)
		e.handle = nil
	}
}

// ---------------------------------------------------------------------------
// Keccak Engine
// ---------------------------------------------------------------------------

// KeccakEngine wraps an opaque GPU Keccak-256 engine handle.
type KeccakEngine struct {
	handle C.ZkMetalKeccakEngine
}

// NewKeccakEngine creates a new GPU-accelerated Keccak-256 engine.
func NewKeccakEngine() (*KeccakEngine, error) {
	var handle C.ZkMetalKeccakEngine
	status := C.zkmetal_keccak_engine_create(&handle)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	eng := &KeccakEngine{handle: handle}
	runtime.SetFinalizer(eng, func(e *KeccakEngine) { e.Close() })
	return eng, nil
}

// Close releases the GPU resources held by this engine.
func (e *KeccakEngine) Close() {
	if e.handle != nil {
		C.zkmetal_keccak_engine_destroy(e.handle)
		e.handle = nil
	}
}

// ---------------------------------------------------------------------------
// FRI Engine
// ---------------------------------------------------------------------------

// FRIEngine wraps an opaque GPU FRI folding engine handle.
type FRIEngine struct {
	handle C.ZkMetalFRIEngine
}

// NewFRIEngine creates a new GPU-accelerated FRI folding engine.
func NewFRIEngine() (*FRIEngine, error) {
	var handle C.ZkMetalFRIEngine
	status := C.zkmetal_fri_engine_create(&handle)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	eng := &FRIEngine{handle: handle}
	runtime.SetFinalizer(eng, func(e *FRIEngine) { e.Close() })
	return eng, nil
}

// Close releases the GPU resources held by this engine.
func (e *FRIEngine) Close() {
	if e.handle != nil {
		C.zkmetal_fri_engine_destroy(e.handle)
		e.handle = nil
	}
}

// ---------------------------------------------------------------------------
// Pairing Engine
// ---------------------------------------------------------------------------

// PairingEngine wraps an opaque GPU BN254 pairing engine handle.
type PairingEngine struct {
	handle C.ZkMetalPairingEngine
}

// NewPairingEngine creates a new GPU-accelerated BN254 pairing engine.
func NewPairingEngine() (*PairingEngine, error) {
	var handle C.ZkMetalPairingEngine
	status := C.zkmetal_pairing_engine_create(&handle)
	if err := statusToError(status); err != nil {
		return nil, err
	}
	eng := &PairingEngine{handle: handle}
	runtime.SetFinalizer(eng, func(e *PairingEngine) { e.Close() })
	return eng, nil
}

// Close releases the GPU resources held by this engine.
func (e *PairingEngine) Close() {
	if e.handle != nil {
		C.zkmetal_pairing_engine_destroy(e.handle)
		e.handle = nil
	}
}
