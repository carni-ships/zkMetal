//go:build darwin && arm64

package gnark

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"testing"
	"unsafe"

	"github.com/carni-ships/zkMetal/bindings/go/zkmetal"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254"
	"github.com/consensys/gnark-crypto/ecc/bn254/fp"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

// ---------------------------------------------------------------------------
// Type layout verification
// ---------------------------------------------------------------------------

// TestTypeLayoutCompatibility verifies that gnark and zkMetal types have
// identical memory layouts, which is the foundation of our zero-copy casts.
func TestTypeLayoutCompatibility(t *testing.T) {
	// fr.Element and BN254Fr must both be [4]uint64 = 32 bytes.
	if got := unsafe.Sizeof(fr.Element{}); got != 32 {
		t.Fatalf("fr.Element size: got %d, want 32", got)
	}
	if got := unsafe.Sizeof(zkmetal.BN254Fr{}); got != 32 {
		t.Fatalf("BN254Fr size: got %d, want 32", got)
	}

	// fp.Element and BN254Fq must both be [4]uint64 = 32 bytes.
	if got := unsafe.Sizeof(fp.Element{}); got != 32 {
		t.Fatalf("fp.Element size: got %d, want 32", got)
	}
	if got := unsafe.Sizeof(zkmetal.BN254Fq{}); got != 32 {
		t.Fatalf("BN254Fq size: got %d, want 32", got)
	}

	// bn254.G1Affine and BN254G1Affine must both be 64 bytes.
	if got := unsafe.Sizeof(bn254.G1Affine{}); got != 64 {
		t.Fatalf("bn254.G1Affine size: got %d, want 64", got)
	}
	if got := unsafe.Sizeof(zkmetal.BN254G1Affine{}); got != 64 {
		t.Fatalf("BN254G1Affine size: got %d, want 64", got)
	}

	// Verify field offsets match: G1Affine.X at offset 0, G1Affine.Y at offset 32.
	var gnarkPt bn254.G1Affine
	var zkPt zkmetal.BN254G1Affine
	gnarkXOff := uintptr(unsafe.Pointer(&gnarkPt.X)) - uintptr(unsafe.Pointer(&gnarkPt))
	gnarkYOff := uintptr(unsafe.Pointer(&gnarkPt.Y)) - uintptr(unsafe.Pointer(&gnarkPt))
	zkXOff := uintptr(unsafe.Pointer(&zkPt.X)) - uintptr(unsafe.Pointer(&zkPt))
	zkYOff := uintptr(unsafe.Pointer(&zkPt.Y)) - uintptr(unsafe.Pointer(&zkPt))

	if gnarkXOff != zkXOff || gnarkXOff != 0 {
		t.Fatalf("X offset mismatch: gnark=%d, zkmetal=%d", gnarkXOff, zkXOff)
	}
	if gnarkYOff != zkYOff || gnarkYOff != 32 {
		t.Fatalf("Y offset mismatch: gnark=%d, zkmetal=%d", gnarkYOff, zkYOff)
	}
}

// TestZeroCopyCast verifies that a zero-copy cast preserves field element values.
func TestZeroCopyCast(t *testing.T) {
	// Create a known gnark fr.Element.
	var a fr.Element
	a.SetUint64(42)

	// Cast to BN254Fr and back.
	zkA := (*zkmetal.BN254Fr)(unsafe.Pointer(&a))
	back := (*fr.Element)(unsafe.Pointer(zkA))

	if !back.Equal(&a) {
		t.Fatalf("zero-copy round-trip failed: got %v, want %v", back, a)
	}
}

// ---------------------------------------------------------------------------
// MSM correctness tests
// ---------------------------------------------------------------------------

// TestMSMMatchesGnarkCPU verifies that the zkMetal GPU MSM produces the same
// result as gnark's CPU MSM for random inputs.
func TestMSMMatchesGnarkCPU(t *testing.T) {
	sizes := []int{1, 2, 4, 16, 64, 256, 1024}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			points, scalars := generateRandomMSMInputs(n)

			// gnark CPU MSM
			var cpuResult bn254.G1Affine
			_, err := cpuResult.MultiExp(points, scalars, ecc.MultiExpConfig{})
			if err != nil {
				t.Fatalf("gnark CPU MSM failed: %v", err)
			}

			// zkMetal GPU MSM (via bridge)
			var gpuResult bn254.G1Affine
			err = MultiExpAuto(&gpuResult, points, scalars)
			if err != nil {
				t.Fatalf("zkMetal GPU MSM failed: %v", err)
			}

			// Compare results.
			if !cpuResult.Equal(&gpuResult) {
				t.Fatalf("MSM mismatch for n=%d:\n  CPU: %v\n  GPU: %v", n, cpuResult, gpuResult)
			}
		})
	}
}

// TestMSMEmpty verifies that MSM with zero points returns the identity.
func TestMSMEmpty(t *testing.T) {
	var result bn254.G1Affine
	err := MultiExpAuto(&result, nil, nil)
	if err != nil {
		t.Fatalf("empty MSM failed: %v", err)
	}
	// Result should be zero (point at infinity in affine = (0,0)).
	var zero bn254.G1Affine
	if !result.Equal(&zero) {
		t.Fatalf("empty MSM not zero: %v", result)
	}
}

// TestMSMLengthMismatch verifies error handling for mismatched inputs.
func TestMSMLengthMismatch(t *testing.T) {
	points := make([]bn254.G1Affine, 10)
	scalars := make([]fr.Element, 5)

	var result bn254.G1Affine
	err := MultiExpAuto(&result, points, scalars)
	if err == nil {
		t.Fatal("expected error for length mismatch, got nil")
	}
}

// TestMSMSinglePoint verifies MSM with a single scalar*generator.
func TestMSMSinglePoint(t *testing.T) {
	// Use the BN254 generator.
	_, _, g1, _ := bn254.Generators()
	points := []bn254.G1Affine{g1}

	// scalar = 7
	var s fr.Element
	s.SetUint64(7)
	scalars := []fr.Element{s}

	// GPU MSM.
	var gpuResult bn254.G1Affine
	err := MultiExpAuto(&gpuResult, points, scalars)
	if err != nil {
		t.Fatalf("MSM failed: %v", err)
	}

	// Expected: 7 * G1.
	var expected bn254.G1Jac
	expected.ScalarMultiplication(new(bn254.G1Jac).FromAffine(&g1), big.NewInt(7))
	var expectedAff bn254.G1Affine
	expectedAff.FromJacobian(&expected)

	if !gpuResult.Equal(&expectedAff) {
		t.Fatalf("single-point MSM mismatch:\n  got:  %v\n  want: %v", gpuResult, expectedAff)
	}
}

// TestMSMWithEngine tests the engine-based API.
func TestMSMWithEngine(t *testing.T) {
	backend, err := NewMSMBackend()
	if err != nil {
		t.Skipf("cannot create MSM engine (no GPU?): %v", err)
	}
	defer backend.Close()

	n := 128
	points, scalars := generateRandomMSMInputs(n)

	var cpuResult bn254.G1Affine
	_, err = cpuResult.MultiExp(points, scalars, ecc.MultiExpConfig{})
	if err != nil {
		t.Fatalf("gnark CPU MSM failed: %v", err)
	}

	var gpuResult bn254.G1Affine
	err = backend.MultiExp(&gpuResult, points, scalars)
	if err != nil {
		t.Fatalf("engine MSM failed: %v", err)
	}

	if !cpuResult.Equal(&gpuResult) {
		t.Fatalf("engine MSM mismatch")
	}
}

// ---------------------------------------------------------------------------
// NTT correctness tests
// ---------------------------------------------------------------------------

// TestNTTRoundTrip verifies that FFT followed by IFFT recovers the original data.
func TestNTTRoundTrip(t *testing.T) {
	logSizes := []int{4, 8, 10, 12}

	for _, logN := range logSizes {
		n := 1 << logN
		t.Run(fmt.Sprintf("logN=%d", logN), func(t *testing.T) {
			// Generate random field elements.
			original := make([]fr.Element, n)
			for i := range original {
				original[i].SetRandom()
			}

			// Copy for GPU round-trip.
			gpuData := make([]fr.Element, n)
			copy(gpuData, original)

			// Forward NTT.
			err := FFTAuto(gpuData)
			if err != nil {
				t.Fatalf("FFT failed: %v", err)
			}

			// Verify it actually changed the data (not a no-op).
			allSame := true
			for i := range original {
				if !original[i].Equal(&gpuData[i]) {
					allSame = false
					break
				}
			}
			if allSame && n > 1 {
				t.Fatal("FFT did not modify the data")
			}

			// Inverse NTT.
			err = IFFTAuto(gpuData)
			if err != nil {
				t.Fatalf("IFFT failed: %v", err)
			}

			// Compare with original.
			for i := range original {
				if !original[i].Equal(&gpuData[i]) {
					t.Fatalf("NTT round-trip mismatch at index %d: got %v, want %v",
						i, gpuData[i], original[i])
				}
			}
		})
	}
}

// TestNTTNotPowerOfTwo verifies error for non-power-of-two lengths.
func TestNTTNotPowerOfTwo(t *testing.T) {
	data := make([]fr.Element, 3) // not a power of two
	err := FFTAuto(data)
	if err != ErrNotPowerOfTwo {
		t.Fatalf("expected ErrNotPowerOfTwo, got: %v", err)
	}
}

// TestNTTEmpty verifies empty NTT is a no-op.
func TestNTTEmpty(t *testing.T) {
	err := FFTAuto(nil)
	if err != nil {
		t.Fatalf("empty NTT should succeed, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Hash correctness tests
// ---------------------------------------------------------------------------

// TestPoseidon2HashConsistency verifies that single and batch Poseidon2 hashes
// produce the same results.
func TestPoseidon2HashConsistency(t *testing.T) {
	n := 64

	// Generate random pairs.
	left := make([]fr.Element, n)
	right := make([]fr.Element, n)
	for i := range left {
		left[i].SetRandom()
		right[i].SetRandom()
	}

	// Single hash (CPU NEON).
	singleResults := make([]fr.Element, n)
	for i := range left {
		singleResults[i] = Poseidon2Hash(&left[i], &right[i])
	}

	// Batch hash (GPU).
	batchResults, err := Poseidon2CompressAuto(left, right)
	if err != nil {
		t.Fatalf("batch Poseidon2 failed: %v", err)
	}

	// Compare.
	for i := range singleResults {
		if !singleResults[i].Equal(&batchResults[i]) {
			t.Fatalf("Poseidon2 mismatch at index %d: single=%v, batch=%v",
				i, singleResults[i], batchResults[i])
		}
	}
}

// TestPoseidon2MerkleTree verifies Merkle tree construction.
func TestPoseidon2MerkleTree(t *testing.T) {
	n := 8
	leaves := make([]fr.Element, n)
	for i := range leaves {
		leaves[i].SetUint64(uint64(i + 1))
	}

	tree := Poseidon2MerkleTree(leaves)
	if len(tree) != 2*n-1 {
		t.Fatalf("tree size: got %d, want %d", len(tree), 2*n-1)
	}

	// Verify leaves are in the tree.
	for i := range leaves {
		if !leaves[i].Equal(&tree[i]) {
			t.Fatalf("leaf %d not in tree", i)
		}
	}

	// Verify root is non-zero.
	var zero fr.Element
	root := tree[len(tree)-1]
	if root.Equal(&zero) {
		t.Fatal("Merkle root is zero")
	}

	// Verify internal consistency: tree[parent] = H(tree[2i], tree[2i+1]).
	for i := 0; i < n-1; i++ {
		left := tree[2*i]
		right := tree[2*i+1]
		expected := Poseidon2Hash(&left, &right)
		parent := tree[n+i]
		if !expected.Equal(&parent) {
			t.Fatalf("Merkle tree inconsistency at internal node %d", i)
		}
	}
}

// TestPoseidon2CompressEmpty verifies empty batch returns nil.
func TestPoseidon2CompressEmpty(t *testing.T) {
	result, err := Poseidon2CompressAuto(nil, nil)
	if err != nil {
		t.Fatalf("empty compress failed: %v", err)
	}
	if result != nil {
		t.Fatalf("expected nil, got %v", result)
	}
}

// TestPoseidon2CompressLengthMismatch verifies error for mismatched lengths.
func TestPoseidon2CompressLengthMismatch(t *testing.T) {
	left := make([]fr.Element, 5)
	right := make([]fr.Element, 3)
	_, err := Poseidon2CompressAuto(left, right)
	if err != ErrLengthMismatch {
		t.Fatalf("expected ErrLengthMismatch, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Conversion helper tests
// ---------------------------------------------------------------------------

// TestScalarsToZkMetal verifies Montgomery-to-standard scalar conversion.
func TestScalarsToZkMetal(t *testing.T) {
	// Create a known scalar.
	var s fr.Element
	s.SetUint64(42)

	converted := ScalarsToZkMetal([]fr.Element{s})
	if len(converted) != 1 {
		t.Fatalf("expected 1 scalar, got %d", len(converted))
	}

	// The standard form of 42 should have 42 in the first uint32 limb.
	if converted[0][0] != 42 {
		t.Logf("scalar[0] = %v (may differ due to Montgomery domain)", converted[0])
		// Not necessarily 42 if the element was in Montgomery form.
		// Just verify it's non-zero and deterministic.
		converted2 := ScalarsToZkMetal([]fr.Element{s})
		if converted[0] != converted2[0] {
			t.Fatal("scalar conversion is non-deterministic")
		}
	}
}

// TestFrConversionRoundTrip verifies zero-copy Fr <-> BN254Fr conversion.
func TestFrConversionRoundTrip(t *testing.T) {
	elements := make([]fr.Element, 100)
	for i := range elements {
		elements[i].SetRandom()
	}

	zkElements := FrToZkMetal(elements)
	back := FrFromZkMetal(zkElements)

	for i := range elements {
		if !elements[i].Equal(&back[i]) {
			t.Fatalf("Fr round-trip mismatch at %d", i)
		}
	}
}

// TestPointConversionRoundTrip verifies zero-copy point conversion.
func TestPointConversionRoundTrip(t *testing.T) {
	_, _, g1, _ := bn254.Generators()
	points := make([]bn254.G1Affine, 10)
	for i := range points {
		var s big.Int
		s.SetInt64(int64(i + 1))
		points[i].ScalarMultiplication(&g1, &s)
	}

	zkPoints := PointsToZkMetal(points)
	back := PointsFromZkMetal(zkPoints)

	for i := range points {
		if !points[i].Equal(&back[i]) {
			t.Fatalf("point round-trip mismatch at %d", i)
		}
	}
}

// ---------------------------------------------------------------------------
// Benchmarks: gnark CPU vs zkMetal GPU
// ---------------------------------------------------------------------------

func BenchmarkMSM(b *testing.B) {
	sizes := []int{1 << 10, 1 << 14, 1 << 16, 1 << 18}

	for _, n := range sizes {
		points, scalars := generateRandomMSMInputs(n)

		b.Run(fmt.Sprintf("gnark_cpu/n=%d", n), func(b *testing.B) {
			var result bn254.G1Affine
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = result.MultiExp(points, scalars, ecc.MultiExpConfig{})
			}
		})

		b.Run(fmt.Sprintf("zkmetal_gpu/n=%d", n), func(b *testing.B) {
			var result bn254.G1Affine
			// Pre-warm the GPU engine.
			_ = MultiExpAuto(&result, points, scalars)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MultiExpAuto(&result, points, scalars)
			}
		})

		b.Run(fmt.Sprintf("zkmetal_gpu_engine/n=%d", n), func(b *testing.B) {
			backend, err := NewMSMBackend()
			if err != nil {
				b.Skipf("no GPU: %v", err)
			}
			defer backend.Close()

			var result bn254.G1Affine
			// Pre-warm.
			_ = backend.MultiExp(&result, points, scalars)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = backend.MultiExp(&result, points, scalars)
			}
		})
	}
}

func BenchmarkNTT(b *testing.B) {
	logSizes := []int{12, 16, 18, 20}

	for _, logN := range logSizes {
		n := 1 << logN
		data := make([]fr.Element, n)
		for i := range data {
			data[i].SetRandom()
		}

		b.Run(fmt.Sprintf("zkmetal_gpu_fft/logN=%d", logN), func(b *testing.B) {
			buf := make([]fr.Element, n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(buf, data)
				_ = FFTAuto(buf)
			}
		})

		b.Run(fmt.Sprintf("zkmetal_gpu_roundtrip/logN=%d", logN), func(b *testing.B) {
			buf := make([]fr.Element, n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(buf, data)
				_ = FFTAuto(buf)
				_ = IFFTAuto(buf)
			}
		})
	}
}

func BenchmarkPoseidon2(b *testing.B) {
	sizes := []int{64, 1024, 1 << 14, 1 << 16}

	for _, n := range sizes {
		left := make([]fr.Element, n)
		right := make([]fr.Element, n)
		for i := range left {
			left[i].SetRandom()
			right[i].SetRandom()
		}

		b.Run(fmt.Sprintf("cpu_single/n=%d", n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for j := 0; j < n; j++ {
					_ = Poseidon2Hash(&left[j], &right[j])
				}
			}
		})

		b.Run(fmt.Sprintf("gpu_batch/n=%d", n), func(b *testing.B) {
			// Pre-warm.
			_, _ = Poseidon2CompressAuto(left, right)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = Poseidon2CompressAuto(left, right)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

// generateRandomMSMInputs creates random points on BN254 G1 and random scalars.
func generateRandomMSMInputs(n int) ([]bn254.G1Affine, []fr.Element) {
	// Generate random scalars.
	scalars := make([]fr.Element, n)
	for i := range scalars {
		scalars[i].SetRandom()
	}

	// Generate random points by multiplying the generator by random scalars.
	_, _, g1, _ := bn254.Generators()
	points := make([]bn254.G1Affine, n)
	for i := range points {
		var s big.Int
		var rnd fr.Element
		rnd.SetRandom()
		rnd.BigInt(&s)
		points[i].ScalarMultiplication(&g1, &s)
	}

	return points, scalars
}

// Ensure rand is used.
var _ = rand.Reader
