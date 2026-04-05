# zkMetal iOS/macOS Demo

Minimal demo showing GPU-accelerated ZK proving on Apple Silicon.

## What it does

1. **BN254 MSM** -- Multi-scalar multiplication of 2^14 (16,384) points on Metal GPU
2. **Poseidon2 Hash** -- Batch hashing 1,024 pairs on Metal GPU
3. **Groth16 Proof** -- Generates and verifies a SNARK proof for `x^3 + x + 5 = y`

## Build & Run (macOS)

```bash
cd demo/ios
swift build -c release
swift run -c release ZKMetalDemo
```

## Build for iOS

Add the zkMetal package to your Xcode project:

1. File > Add Package Dependencies
2. Point to the local zkMetal repo (or a git URL)
3. Import `zkMetal` in your Swift code

The library targets iOS 16+ and requires Metal GPU access (A-series chip or later).

## Requirements

- macOS 13+ or iOS 16+
- Apple Silicon (M1/M2/M3/M4) or A-series GPU
- Swift 5.9+

## Expected Output

```
zkMetal Demo v0.3.0
============================================

=== BN254 MSM Benchmark ===
  GPU: Apple M3 Pro
  MSM 2^14 (16384 points):                       X.XX ms (best of 3)

=== Poseidon2 Hash Benchmark ===
  Poseidon2 hash 1024 pairs (GPU)                 X.XX ms
  Output[0] non-zero: true
  Hashes computed: 1024

=== Groth16 Prove & Verify ===
  Circuit: x^3 + x + 5 = y
  Public inputs: x=3, y=35
  Constraints: 3, Variables: 5
  Trusted setup                                    X.XX ms
  Prove                                            X.XX ms
  Verify                                           X.XX ms
  Proof valid: true
  Reject bad input: true

============================================
All demos completed successfully.
```
