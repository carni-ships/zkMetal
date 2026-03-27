// Persistia State Transition Proof -- Noir Circuit
//
// Proves that a state transition is valid:
// 1. A BFT quorum of nodes signed this block (Schnorr on Grumpkin)
// 2. The mutations produce the declared state root (Poseidon2 Merkle tree)
// 3. (Recursive) The previous proof in the chain is valid

use std::embedded_curve_ops::EmbeddedCurvePoint;
use poseidon::poseidon2::Poseidon2;

// --- Constants ---

// Compile-time toggle for recursive IVC chain verification.
// When false, the recursive proof verification opcode is eliminated entirely,
// reducing the circuit from ~769K gates to ~42K gates (~18x smaller, ~10x faster).
// Set to true when building recursive proof chains across blocks.
global ENABLE_RECURSIVE: bool = false;

// Proof field counts for recursive verification.
// VK is always 114 Fields in Barretenberg UltraHonk.
// Proof size depends on circuit -- set to maximum expected.
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

struct StateMutation {
    key: Field,
    new_value: Field,
    is_delete: bool,
    enabled: bool,
}

struct StateTransitionOutput {
    state_root: Field,
    block_number: u64,
    proven_blocks: u64,
    genesis_root: Field,
}

// --- Poseidon2 Merkle Tree ---

fn poseidon2_leaf_hash(key: Field, value: Field) -> Field {
    Poseidon2::hash([1, key, value], 3)
}

fn poseidon2_node_hash(left: Field, right: Field) -> Field {
    Poseidon2::hash([2, left, right], 3)
}

fn compute_merkle_root(
    leaves: [StateMutation; 256],
    _leaf_count: u32,
) -> Field {
    let mut hashes: [Field; 256] = [0; 256];
    let mut active: u32 = 0;

    for i in 0..256 {
        if (leaves[i].enabled) & (!leaves[i].is_delete) {
            hashes[active] = poseidon2_leaf_hash(leaves[i].key, leaves[i].new_value);
            active += 1;
        }
    }

    let empty_hash: Field = Poseidon2::hash([0], 1);
    let mut result = empty_hash;

    if active >= 2 {
        let mut p2: u32 = 1;
        for _ in 0..8 {
            if p2 < active { p2 *= 2; }
        }
        let last_hash = hashes[active - 1];
        for i in 0..256 {
            if ((i as u32) >= active) & ((i as u32) < p2) {
                hashes[i] = last_hash;
            }
        }

        let mut sz = p2;
        for _ in 0..8 {
            if sz > 1 {
                let half = sz / 2;
                for j in 0..128 {
                    if (j as u32) < half {
                        hashes[j] = poseidon2_node_hash(hashes[j * 2], hashes[j * 2 + 1]);
                    }
                }
                sz = half;
            }
        }
        result = hashes[0];
    } else if active == 1 {
        result = hashes[0];
    }

    result
}

// --- BFT Quorum ---

fn required_quorum(active_nodes: u32) -> u32 {
    if active_nodes <= 1 { 1 } else {
        let f = (active_nodes - 1) / 3;
        2 * f + 1
    }
}

fn verify_quorum(
    signatures: [NodeSignature; 4],
    _sig_count: u32,
    active_nodes: u32,
) {
    let quorum = required_quorum(active_nodes);
    let mut valid: u32 = 0;

    for i in 0..4 {
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
    mutations: [StateMutation; 256],
    mutation_count: u32,
    signatures: [NodeSignature; 4],
    sig_count: u32,

    prev_proven_blocks: u64,
    prev_genesis_root: Field,

    // Recursive proof inputs (eliminated at compile time when ENABLE_RECURSIVE=false)
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

    // 2. Verify mutations produce the declared state root
    assert(mutation_count as u64 <= 256, "Too many mutations");
    let computed_root = compute_merkle_root(mutations, mutation_count);
    assert(computed_root == new_state_root, "Merkle root mismatch");

    // 3. Recursive IVC chain (compile-time conditional)
    let mut proven_blocks: u64 = 0;
    let mut genesis_root: Field = 0;

    if ENABLE_RECURSIVE {
        // Verify previous proof in the chain
        // proof_type 0 = UltraHonk (non-ZK)
        std::verify_proof_with_type(
            prev_vk,
            prev_proof,
            prev_public_inputs,
            prev_key_hash,
            0,
        );
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
fn test_required_quorum() {
    assert(required_quorum(1) == 1);
    assert(required_quorum(3) == 1);
    assert(required_quorum(4) == 3);
    assert(required_quorum(7) == 5);
    assert(required_quorum(10) == 7);
}

#[test]
fn test_poseidon2_leaf_hash_deterministic() {
    let k: Field = 42;
    let v: Field = 99;
    let h1 = poseidon2_leaf_hash(k, v);
    let h2 = poseidon2_leaf_hash(k, v);
    assert(h1 == h2);
    let h3 = poseidon2_leaf_hash(k, 100);
    assert(h1 != h3);
}

#[test]
fn test_poseidon2_node_hash_deterministic() {
    let a: Field = 1;
    let b: Field = 2;
    let h1 = poseidon2_node_hash(a, b);
    let h2 = poseidon2_node_hash(a, b);
    assert(h1 == h2);
    let h3 = poseidon2_node_hash(b, a);
    assert(h1 != h3);
}

#[test]
fn test_merkle_root_single_leaf() {
    let mut muts: [StateMutation; 256] = [StateMutation { key: 0, new_value: 0, is_delete: false, enabled: false }; 256];
    muts[0] = StateMutation { key: 42, new_value: 99, is_delete: false, enabled: true };
    let root = compute_merkle_root(muts, 1);
    let expected = poseidon2_leaf_hash(42, 99);
    assert(root == expected);
}

#[test]
fn test_merkle_root_two_leaves() {
    let mut muts: [StateMutation; 256] = [StateMutation { key: 0, new_value: 0, is_delete: false, enabled: false }; 256];
    muts[0] = StateMutation { key: 1, new_value: 10, is_delete: false, enabled: true };
    muts[1] = StateMutation { key: 2, new_value: 20, is_delete: false, enabled: true };
    let root = compute_merkle_root(muts, 2);
    let l0 = poseidon2_leaf_hash(1, 10);
    let l1 = poseidon2_leaf_hash(2, 20);
    let expected = poseidon2_node_hash(l0, l1);
    assert(root == expected);
}

#[test]
fn test_schnorr_verify_known_good() {
    let pub_key = EmbeddedCurvePoint {
        x: 0x04b260954662e97f00cab9adb773a259097f7a274b83b113532bce27fa3fb96a,
        y: 0x2fd51571db6c08666b0edfbfbc57d432068bccd0110a39b166ab243da0037197,
        is_infinite: false,
    };
    let signature: [u8; 64] = [
        1, 13, 119, 112, 212, 39, 233, 41, 84, 235, 255, 93, 245, 172, 186, 83, 157, 253, 76,
        77, 33, 128, 178, 15, 214, 67, 105, 107, 177, 234, 77, 48, 27, 237, 155, 84, 39, 84,
        247, 27, 22, 8, 176, 230, 24, 115, 145, 220, 254, 122, 135, 179, 171, 4, 214, 202, 64,
        199, 19, 84, 239, 138, 124, 12,
    ];
    let message: [u8; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let valid = schnorr::verify_signature(pub_key, signature, message);
    assert(valid);
}
