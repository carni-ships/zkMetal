//go:build darwin && arm64

// Package zkmetal provides Go bindings for the zkMetal C library (NeonFieldOps),
// a high-performance cryptographic primitives library optimized for Apple Silicon
// using ARM NEON intrinsics, Metal GPU compute shaders, and hand-tuned assembly.
//
// Supported curves: BN254, BLS12-381, BLS12-377, secp256k1, Grumpkin,
// Pallas/Vesta, Ed25519, BabyJubjub, Jubjub.
//
// Supported operations: field arithmetic (Montgomery CIOS), elliptic curve
// point operations, multi-scalar multiplication (Pippenger), NTT/INTT,
// pairings (Miller loop + final exponentiation), hash functions
// (Poseidon2, Keccak-256, Blake3).
//
// This package targets integration with gnark (Go ZK framework) and
// Ethereum clients (Geth, Prysm) on Apple Silicon hardware.
package zkmetal

/*
#cgo CFLAGS: -I../../../Sources/NeonFieldOps/include
#cgo LDFLAGS: -L../../../build -lNeonFieldOps
#include "NeonFieldOps.h"
*/
import "C"
