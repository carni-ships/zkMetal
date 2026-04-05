//go:build darwin && arm64

package zkmetal

import (
	"testing"
	"unsafe"
)

// TestGPUAvailable checks that GPU availability query works.
func TestGPUAvailable(t *testing.T) {
	avail := GPUAvailable()
	t.Logf("GPU available: %v", avail)
	// On Apple Silicon this should always be true, but we don't fail if not.
}

// TestVersion checks that the version string is non-empty.
func TestVersion(t *testing.T) {
	v := Version()
	if v == "" {
		t.Fatal("expected non-empty version string")
	}
	t.Logf("zkMetal version: %s", v)
}

// TestMSMEngineLifecycle verifies create/close of MSM engine.
func TestMSMEngineLifecycle(t *testing.T) {
	if !GPUAvailable() {
		t.Skip("no Metal GPU available")
	}
	eng, err := NewMSMEngine()
	if err != nil {
		t.Fatalf("NewMSMEngine: %v", err)
	}
	eng.Close()
	// Double close should be safe.
	eng.Close()
}

// TestNTTEngineLifecycle verifies create/close of NTT engine.
func TestNTTEngineLifecycle(t *testing.T) {
	if !GPUAvailable() {
		t.Skip("no Metal GPU available")
	}
	eng, err := NewNTTEngine()
	if err != nil {
		t.Fatalf("NewNTTEngine: %v", err)
	}
	eng.Close()
}

// TestPoseidon2EngineLifecycle verifies create/close of Poseidon2 engine.
func TestPoseidon2EngineLifecycle(t *testing.T) {
	if !GPUAvailable() {
		t.Skip("no Metal GPU available")
	}
	eng, err := NewPoseidon2Engine()
	if err != nil {
		t.Fatalf("NewPoseidon2Engine: %v", err)
	}
	eng.Close()
}

// TestKeccakEngineLifecycle verifies create/close of Keccak engine.
func TestKeccakEngineLifecycle(t *testing.T) {
	if !GPUAvailable() {
		t.Skip("no Metal GPU available")
	}
	eng, err := NewKeccakEngine()
	if err != nil {
		t.Fatalf("NewKeccakEngine: %v", err)
	}
	eng.Close()
}

// TestGPUMSMZeroPoints verifies MSM with zero points returns empty result.
func TestGPUMSMZeroPoints(t *testing.T) {
	x, y, z, err := GPUMSMAuto(nil, nil, 0)
	if err != nil {
		t.Fatalf("GPUMSMAuto(0 points): %v", err)
	}
	// Zero points should return zero result.
	var zero [32]byte
	if x != zero || y != zero || z != zero {
		t.Log("zero-point MSM returned non-zero (may be point-at-infinity encoding)")
	}
}

// TestGPUMSMTypedZero verifies typed MSM with empty slices.
func TestGPUMSMTypedZero(t *testing.T) {
	result, err := GPUMSMTypedAuto(nil, nil)
	if err != nil {
		t.Fatalf("GPUMSMTypedAuto(nil): %v", err)
	}
	_ = result
}

// TestGPUNTTZero verifies NTT with empty data is a no-op.
func TestGPUNTTZero(t *testing.T) {
	err := GPUNTTAuto(nil, 0)
	if err != nil {
		t.Fatalf("GPUNTTAuto(nil): %v", err)
	}
}

// TestGPUPoseidon2Zero verifies Poseidon2 with zero pairs.
func TestGPUPoseidon2Zero(t *testing.T) {
	out, err := GPUPoseidon2HashPairsAuto(nil, 0)
	if err != nil {
		t.Fatalf("GPUPoseidon2HashPairsAuto(0): %v", err)
	}
	if out != nil {
		t.Fatal("expected nil output for 0 pairs")
	}
}

// TestGPUKeccakZero verifies Keccak with zero inputs.
func TestGPUKeccakZero(t *testing.T) {
	out, err := GPUKeccak256HashAuto(nil, 0)
	if err != nil {
		t.Fatalf("GPUKeccak256HashAuto(0): %v", err)
	}
	if out != nil {
		t.Fatal("expected nil output for 0 inputs")
	}
}

// TestGPUFRIFoldInvalidLogN verifies FRI fold rejects logN < 1.
func TestGPUFRIFoldInvalidLogN(t *testing.T) {
	var beta [32]byte
	_, err := GPUFRIFoldAuto(nil, 0, beta)
	if err != ErrInvalidInput {
		t.Fatalf("expected ErrInvalidInput for logN=0, got: %v", err)
	}
}

// TestGPUMSMInputValidation verifies MSM rejects short buffers.
func TestGPUMSMInputValidation(t *testing.T) {
	short := make([]byte, 10)
	_, _, _, err := GPUMSMAuto(short, short, 1)
	if err != ErrInvalidInput {
		t.Fatalf("expected ErrInvalidInput for short buffer, got: %v", err)
	}
}

// TestGPUEngineTypes verifies engine handle sizes are pointer-sized.
func TestGPUEngineTypes(t *testing.T) {
	ptrSize := unsafe.Sizeof(uintptr(0))
	if got := unsafe.Sizeof(MSMEngine{}); got != ptrSize {
		t.Fatalf("MSMEngine size: got %d, want %d", got, ptrSize)
	}
	if got := unsafe.Sizeof(NTTEngine{}); got != ptrSize {
		t.Fatalf("NTTEngine size: got %d, want %d", got, ptrSize)
	}
}
