// Persistia Incremental State Proof -- Noir Circuit
//
// Alternative to the full-tree circuit that uses sparse Merkle tree
// inclusion/update proofs instead of recomputing the entire tree.
//
// Proves that a state transition is valid:
// 1. A BFT quorum of nodes signed this block (Schnorr on Grumpkin)
// 2. Applying the mutations to the sparse Merkle tree transforms
//    prev_state_root into new_state_root
// 3. (Recursive) The previous proof in the chain is valid
//
// Tradeoffs vs full-tree circuit:
// - Gate count: O(MAX_MUTATIONS * TREE_DEPTH) vs O(MAX_MUTATIONS^2)
//   With M=64, D=20: ~3.6M gates vs ~9.1M gates (2.5x smaller)
// - State tree: persistent across blocks, supports 2^TREE_DEPTH keys
// - Witness: requires Merkle sibling paths from the data source
// - MAX_MUTATIONS: lower per proof (64 vs 1024), batch across proofs if needed

use std::embedded_curve_ops::EmbeddedCurvePoint;
use poseidon::poseidon2::Poseidon2;

// --- Constants ---

// Maximum mutations per proof. Lower than full-tree circuit but
// each mutation is cheaper (2*TREE_DEPTH hashes vs full tree rebuild).
global MAX_MUTATIONS: u32 = 64;

// Sparse Merkle tree depth. Supports 2^TREE_DEPTH unique keys.
// Depth 20 = ~1M keys, depth 24 = ~16M keys.
global TREE_DEPTH: u32 = 20;

// BFT validator limit (same as full-tree circuit).
global MAX_VALIDATORS: u32 = 4;

// Recursive proof sizes (same as full-tree circuit).
global ENABLE_RECURSIVE: bool = false;
global VK_SIZE: u32 = 115;
global PROOF_SIZE: u32 = 449;
global PUBLIC_INPUTS_SIZE: u32 = 8;

// --- Types ---

struct NodeSignature {
    pubkey_x: Field,
    pubkey_y: Field,
    signature: [u8; 64],
    msg: [u8; 32],
    enabled: bool,
}

struct MerkleUpdate {
    key: Field,
    old_value: Field,
    new_value: Field,
    siblings: [Field; TREE_DEPTH],
    is_delete: bool,
    enabled: bool,
}

struct StateTransitionOutput {
    state_root: Field,
    block_number: u64,
    proven_blocks: u64,
    genesis_root: Field,
}

// --- Poseidon2 Hashing ---

fn poseidon2_leaf_hash(key: Field, value: Field) -> Field {
    Poseidon2::hash([1, key, value], 3)
}

fn poseidon2_node_hash(left: Field, right: Field) -> Field {
    Poseidon2::hash([2, left, right], 3)
}

fn poseidon2_empty_hash() -> Field {
    Poseidon2::hash([0], 1)
}

// --- Sparse Merkle Tree Operations ---

/// Compute the Merkle root from a leaf and its sibling path.
/// The key determines the path direction at each level (bit 0 = left/right at level 0).
fn compute_root_from_path(
    leaf: Field,
    key_bits: [u1; TREE_DEPTH],
    siblings: [Field; TREE_DEPTH],
) -> Field {
    let mut current = leaf;
    for i in 0..TREE_DEPTH {
        if key_bits[i] == 0 {
            current = poseidon2_node_hash(current, siblings[i]);
        } else {
            current = poseidon2_node_hash(siblings[i], current);
        }
    }
    current
}

/// Apply a sequence of Merkle updates to a root.
/// Each update:
/// 1. Verifies the old value exists at the key position (against current root)
/// 2. Computes the new root after replacing the old leaf with the new leaf
///
/// Updates are applied sequentially. Each update's sibling path must be valid
/// against the root produced by all previous updates.
fn apply_merkle_updates(
    updates: [MerkleUpdate; MAX_MUTATIONS],
    initial_root: Field,
) -> Field {
    let empty_leaf = poseidon2_empty_hash();
    let mut current_root = initial_root;

    for i in 0..MAX_MUTATIONS {
        if updates[i].enabled {
            let key_bits: [u1; TREE_DEPTH] = updates[i].key.to_le_bits();

            // Verify old state: old leaf must exist at this position
            let old_leaf = if updates[i].old_value == 0 {
                empty_leaf
            } else {
                poseidon2_leaf_hash(updates[i].key, updates[i].old_value)
            };
            let computed_old_root = compute_root_from_path(
                old_leaf, key_bits, updates[i].siblings,
            );
            assert(computed_old_root == current_root, "Invalid Merkle inclusion proof");

            // Compute new state: replace leaf and recompute root
            let new_leaf = if updates[i].is_delete {
                empty_leaf
            } else {
                poseidon2_leaf_hash(updates[i].key, updates[i].new_value)
            };
            current_root = compute_root_from_path(
                new_leaf, key_bits, updates[i].siblings,
            );
        }
    }

    current_root
}

// --- BFT Quorum ---

fn required_quorum(active_nodes: u32) -> u32 {
    if active_nodes <= 1 { 1 } else {
        let f = (active_nodes - 1) / 3;
        2 * f + 1
    }
}

fn verify_quorum(
    signatures: [NodeSignature; MAX_VALIDATORS],
    _sig_count: u32,
    active_nodes: u32,
) {
    let quorum = required_quorum(active_nodes);
    let mut valid: u32 = 0;

    for i in 0..MAX_VALIDATORS {
        if signatures[i].enabled {
            let pk = EmbeddedCurvePoint {
                x: signatures[i].pubkey_x,
                y: signatures[i].pubkey_y,
                is_infinite: false,
            };
            let ok = schnorr::verify_signature(pk, signatures[i].signature, signatures[i].msg);
            if ok {
                valid += 1;
            }
        }
    }

    assert(valid >= quorum, "Insufficient quorum");
}

// --- Main Circuit ---

fn main(
    // Private inputs
    updates: [MerkleUpdate; MAX_MUTATIONS],
    mutation_count: u32,
    signatures: [NodeSignature; MAX_VALIDATORS],
    sig_count: u32,

    prev_proven_blocks: u64,
    prev_genesis_root: Field,

    // Recursive proof inputs
    prev_proof: [Field; PROOF_SIZE],
    prev_vk: [Field; VK_SIZE],
    prev_key_hash: Field,
    prev_public_inputs: [Field; PUBLIC_INPUTS_SIZE],

    // Public inputs
    prev_state_root: pub Field,
    new_state_root: pub Field,
    block_number: pub u64,
    active_nodes: pub u32,
) -> pub StateTransitionOutput {

    // 1. Verify BFT quorum
    verify_quorum(signatures, sig_count, active_nodes);

    // 2. Apply Merkle updates and verify state transition
    assert(mutation_count as u64 <= MAX_MUTATIONS as u64, "Too many mutations");
    let computed_root = apply_merkle_updates(updates, prev_state_root);
    assert(computed_root == new_state_root, "State root mismatch after applying updates");

    // 3. Recursive IVC chain
    let mut proven_blocks: u64 = 0;
    let mut genesis_root: Field = 0;

    if ENABLE_RECURSIVE {
        std::verify_proof_with_type(
            prev_vk,
            prev_proof,
            prev_public_inputs,
            prev_key_hash,
            0,
        );
        assert(prev_state_root == prev_public_inputs[1], "State root chain broken");
        proven_blocks = prev_proven_blocks + 1;
        genesis_root = prev_genesis_root;
    } else {
        proven_blocks = 1;
        genesis_root = prev_state_root;
    }

    StateTransitionOutput {
        state_root: new_state_root,
        block_number,
        proven_blocks,
        genesis_root,
    }
}

// --- Tests ---

#[test]
fn test_empty_tree_no_updates() {
    let empty = poseidon2_empty_hash();
    // Build empty tree root: hash empty leaves up TREE_DEPTH levels
    let mut root = empty;
    for _ in 0..TREE_DEPTH {
        root = poseidon2_node_hash(root, root);
    }

    let updates: [MerkleUpdate; MAX_MUTATIONS] = [MerkleUpdate {
        key: 0, old_value: 0, new_value: 0,
        siblings: [0; TREE_DEPTH], is_delete: false, enabled: false,
    }; MAX_MUTATIONS];

    let result = apply_merkle_updates(updates, root);
    assert(result == root);
}

#[test]
fn test_single_insertion() {
    let empty = poseidon2_empty_hash();
    // Build empty tree: all siblings are empty subtree hashes at each level
    let mut empty_subtree: [Field; TREE_DEPTH] = [0; TREE_DEPTH];
    empty_subtree[0] = empty;
    for i in 1..TREE_DEPTH {
        empty_subtree[i] = poseidon2_node_hash(empty_subtree[i - 1], empty_subtree[i - 1]);
    }

    // Empty tree root
    let mut root = empty;
    for i in 0..TREE_DEPTH {
        root = poseidon2_node_hash(root, empty_subtree[i]);
    }

    // Insert key=0 (all left turns), value=42
    let mut updates: [MerkleUpdate; MAX_MUTATIONS] = [MerkleUpdate {
        key: 0, old_value: 0, new_value: 0,
        siblings: [0; TREE_DEPTH], is_delete: false, enabled: false,
    }; MAX_MUTATIONS];

    updates[0] = MerkleUpdate {
        key: 0,
        old_value: 0,  // empty slot
        new_value: 42,
        siblings: empty_subtree,
        is_delete: false,
        enabled: true,
    };

    let new_root = apply_merkle_updates(updates, root);

    // Verify: recompute expected root with key=0 inserted
    let new_leaf = poseidon2_leaf_hash(0, 42);
    let mut expected = new_leaf;
    for i in 0..TREE_DEPTH {
        expected = poseidon2_node_hash(expected, empty_subtree[i]);
    }
    assert(new_root == expected);
}

#[test]
fn test_update_existing_value() {
    let empty = poseidon2_empty_hash();
    let mut empty_subtree: [Field; TREE_DEPTH] = [0; TREE_DEPTH];
    empty_subtree[0] = empty;
    for i in 1..TREE_DEPTH {
        empty_subtree[i] = poseidon2_node_hash(empty_subtree[i - 1], empty_subtree[i - 1]);
    }

    // Tree with key=0, value=42
    let leaf = poseidon2_leaf_hash(0, 42);
    let mut root = leaf;
    for i in 0..TREE_DEPTH {
        root = poseidon2_node_hash(root, empty_subtree[i]);
    }

    // Update key=0 from 42 to 99
    let mut updates: [MerkleUpdate; MAX_MUTATIONS] = [MerkleUpdate {
        key: 0, old_value: 0, new_value: 0,
        siblings: [0; TREE_DEPTH], is_delete: false, enabled: false,
    }; MAX_MUTATIONS];

    updates[0] = MerkleUpdate {
        key: 0,
        old_value: 42,
        new_value: 99,
        siblings: empty_subtree,
        is_delete: false,
        enabled: true,
    };

    let new_root = apply_merkle_updates(updates, root);

    let new_leaf = poseidon2_leaf_hash(0, 99);
    let mut expected = new_leaf;
    for i in 0..TREE_DEPTH {
        expected = poseidon2_node_hash(expected, empty_subtree[i]);
    }
    assert(new_root == expected);
}
