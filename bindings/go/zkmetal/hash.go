//go:build darwin && arm64

package zkmetal

/*
#include "NeonFieldOps.h"
*/
import "C"
import "unsafe"

// ---------------------------------------------------------------------------
// Poseidon2 (BN254 Fr, NEON-optimized)
// ---------------------------------------------------------------------------

// Poseidon2Permutation applies the full Poseidon2 permutation on 3 Fr elements.
func Poseidon2Permutation(state [3]BN254Fr) [3]BN254Fr {
	var r [3]BN254Fr
	C.poseidon2_permutation_cpu((*C.uint64_t)(unsafe.Pointer(&state[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// Poseidon2Hash computes the 2-to-1 Poseidon2 compression hash.
func Poseidon2Hash(a, b *BN254Fr) BN254Fr {
	var r BN254Fr
	C.poseidon2_hash_cpu((*C.uint64_t)(unsafe.Pointer(&a[0])),
		(*C.uint64_t)(unsafe.Pointer(&b[0])),
		(*C.uint64_t)(unsafe.Pointer(&r[0])))
	return r
}

// Poseidon2BatchHash computes count 2-to-1 hashes in parallel.
// input has 2*count Fr elements (pairs), output has count Fr elements.
func Poseidon2BatchHash(input []BN254Fr, count int) []BN254Fr {
	if count == 0 {
		return nil
	}
	output := make([]BN254Fr, count)
	C.poseidon2_hash_batch_cpu((*C.uint64_t)(unsafe.Pointer(&input[0])),
		C.int(count),
		(*C.uint64_t)(unsafe.Pointer(&output[0])))
	return output
}

// Poseidon2MerkleTree builds a complete binary Merkle tree from leaves.
// Returns a tree of 2n-1 elements: tree[0..n-1] = leaves, tree[2n-2] = root.
func Poseidon2MerkleTree(leaves []BN254Fr) []BN254Fr {
	n := len(leaves)
	if n == 0 {
		return nil
	}
	tree := make([]BN254Fr, 2*n-1)
	C.poseidon2_merkle_tree_cpu((*C.uint64_t)(unsafe.Pointer(&leaves[0])),
		C.int(n),
		(*C.uint64_t)(unsafe.Pointer(&tree[0])))
	return tree
}

// ---------------------------------------------------------------------------
// Keccak-256 (ARM NEON optimized)
// ---------------------------------------------------------------------------

// Keccak256 computes the Keccak-256 hash of arbitrary input.
func Keccak256(data []byte) [32]byte {
	var out [32]byte
	var ptr *C.uint8_t
	if len(data) > 0 {
		ptr = (*C.uint8_t)(unsafe.Pointer(&data[0]))
	}
	C.keccak256_hash_neon(ptr, C.size_t(len(data)),
		(*C.uint8_t)(unsafe.Pointer(&out[0])))
	return out
}

// Keccak256Pair computes Keccak-256 of two concatenated 32-byte inputs.
// Optimized: 64 bytes < 136-byte rate, so only one f1600 call needed.
func Keccak256Pair(a, b [32]byte) [32]byte {
	var out [32]byte
	C.keccak256_hash_pair_neon((*C.uint8_t)(unsafe.Pointer(&a[0])),
		(*C.uint8_t)(unsafe.Pointer(&b[0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0])))
	return out
}

// Keccak256BatchPairs computes n Keccak-256 hashes of 64-byte pairs.
func Keccak256BatchPairs(inputs [][64]byte) [][32]byte {
	n := len(inputs)
	if n == 0 {
		return nil
	}
	outputs := make([][32]byte, n)
	C.keccak256_batch_hash_pairs_neon(
		(*C.uint8_t)(unsafe.Pointer(&inputs[0][0])),
		(*C.uint8_t)(unsafe.Pointer(&outputs[0][0])),
		C.size_t(n))
	return outputs
}

// ---------------------------------------------------------------------------
// Blake3 (ARM NEON optimized)
// ---------------------------------------------------------------------------

// Blake3Hash computes the Blake3 hash of input (up to 64 bytes -> 32 bytes).
func Blake3Hash(data []byte) [32]byte {
	var out [32]byte
	var ptr *C.uint8_t
	if len(data) > 0 {
		ptr = (*C.uint8_t)(unsafe.Pointer(&data[0]))
	}
	C.blake3_hash_neon(ptr, C.size_t(len(data)),
		(*C.uint8_t)(unsafe.Pointer(&out[0])))
	return out
}

// Blake3HashPair computes Blake3 parent node hash: left(32B) || right(32B) -> 32B.
func Blake3HashPair(left, right [32]byte) [32]byte {
	var out [32]byte
	C.blake3_hash_pair_neon((*C.uint8_t)(unsafe.Pointer(&left[0])),
		(*C.uint8_t)(unsafe.Pointer(&right[0])),
		(*C.uint8_t)(unsafe.Pointer(&out[0])))
	return out
}

// Blake3BatchPairs computes n Blake3 parent hashes from 64-byte pairs.
func Blake3BatchPairs(inputs [][64]byte) [][32]byte {
	n := len(inputs)
	if n == 0 {
		return nil
	}
	outputs := make([][32]byte, n)
	C.blake3_batch_hash_pairs_neon(
		(*C.uint8_t)(unsafe.Pointer(&inputs[0][0])),
		(*C.uint8_t)(unsafe.Pointer(&outputs[0][0])),
		C.size_t(n))
	return outputs
}
