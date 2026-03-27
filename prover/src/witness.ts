// Witness generation for the Persistia Noir circuit.
//
// Transforms Persistia node API responses into the fixed-size witness
// format that the Noir circuit expects. All arrays are padded to their
// compile-time maximums with disabled sentinel entries.
//
// Schnorr signatures on Grumpkin curve via @aztec/bb.js.
// Poseidon2 Merkle tree with Field-typed keys/values.

import { createHash } from "crypto";
import { Barretenberg } from "@aztec/bb.js";

const MAX_VALIDATORS = 4;
const MAX_MUTATIONS = 1024;
const VK_SIZE = 115;
const PROOF_SIZE = 449;
const PUBLIC_INPUTS_SIZE = 8;

// --- Types matching the Noir circuit structs ---

interface NodeSignatureWitness {
  pubkey_x: string;      // Field as hex string
  pubkey_y: string;      // Field as hex string
  signature: number[];   // [u8; 64] -- [s(32) || e(32)]
  msg: number[];         // [u8; 32] -- SHA-256 hash of block data
  enabled: boolean;
}

interface StateMutationWitness {
  key: string;           // Field as hex/decimal string
  new_value: string;     // Field as hex/decimal string
  is_delete: boolean;
  enabled: boolean;
}

export interface CircuitWitness {
  mutations: StateMutationWitness[];
  mutation_count: number;
  signatures: NodeSignatureWitness[];
  sig_count: number;
  prev_proven_blocks: number;
  prev_genesis_root: string;
  // Recursive proof inputs (zeroed when ENABLE_RECURSIVE=false in circuit)
  prev_proof: string[];
  prev_vk: string[];
  prev_key_hash: string;
  prev_public_inputs: string[];
  // Public inputs
  prev_state_root: string;
  new_state_root: string;
  block_number: number;
  active_nodes: number;
}

// --- Helpers ---

function hexToBytes(hex: string): number[] {
  const bytes: number[] = [];
  const clean = hex.startsWith("0x") ? hex.slice(2) : hex;
  for (let i = 0; i < clean.length; i += 2) {
    bytes.push(parseInt(clean.substring(i, i + 2), 16));
  }
  return bytes;
}

function bytesToHex(bytes: Uint8Array): string {
  return "0x" + Array.from(bytes).map(b => b.toString(16).padStart(2, "0")).join("");
}

function sha256Hash(data: Buffer | Uint8Array): number[] {
  return Array.from(createHash("sha256").update(data).digest());
}

function padArray(arr: number[], targetLen: number): number[] {
  const result = [...arr];
  while (result.length < targetLen) result.push(0);
  return result.slice(0, targetLen);
}

// --- Poseidon2 Merkle Root (matches Noir circuit) ---

function fieldToBytes(field: string | bigint): Uint8Array {
  const n = typeof field === "string" ? BigInt(field) : field;
  const hex = n.toString(16).padStart(64, "0");
  const bytes = new Uint8Array(32);
  for (let i = 0; i < 32; i++) {
    bytes[i] = parseInt(hex.substring(i * 2, i * 2 + 2), 16);
  }
  return bytes;
}

async function poseidon2Hash(inputs: (string | bigint)[]): Promise<string> {
  const bb = await getBb();
  const fieldInputs = inputs.map(fieldToBytes);
  const { hash } = await bb.poseidon2Hash({ inputs: fieldInputs });
  return bytesToHex(hash);
}

/** Batch multiple Poseidon2 hashes in parallel to reduce WASM FFI round-trip overhead. */
async function poseidon2HashBatch(inputSets: (string | bigint)[][]): Promise<string[]> {
  const bb = await getBb();
  const results = await Promise.all(
    inputSets.map(async (inputs) => {
      const fieldInputs = inputs.map(fieldToBytes);
      const { hash } = await bb.poseidon2Hash({ inputs: fieldInputs });
      return bytesToHex(hash);
    }),
  );
  return results;
}

async function poseidon2LeafHash(key: string, value: string): Promise<string> {
  return poseidon2Hash(["1", key, value]);
}

async function poseidon2NodeHash(left: string, right: string): Promise<string> {
  return poseidon2Hash(["2", left, right]);
}

/** Compute Poseidon2 Merkle root matching the Noir circuit's compute_merkle_root(). */
export async function computePoseidon2MerkleRoot(
  mutations: StateMutationWitness[],
): Promise<string> {
  // Collect active (non-delete) mutations
  const active = mutations.filter((m) => m.enabled && !m.is_delete);

  if (active.length === 0) {
    return poseidon2Hash(["0"]);
  }

  // Batch all leaf hashes in a single parallel call
  const leafInputs = active.map((m) => ["1" as string | bigint, m.key, m.new_value]);
  const hashes = await poseidon2HashBatch(leafInputs);

  if (hashes.length === 1) {
    return hashes[0];
  }

  // Pad to next power of 2 with last hash (matches Noir circuit)
  let p2 = 1;
  while (p2 < hashes.length) p2 *= 2;
  const lastHash = hashes[hashes.length - 1];
  while (hashes.length < p2) hashes.push(lastHash);

  // Binary tree reduction — batch each level
  let sz = p2;
  while (sz > 1) {
    const half = sz / 2;
    const levelInputs: (string | bigint)[][] = [];
    for (let j = 0; j < half; j++) {
      levelInputs.push(["2", hashes[j * 2], hashes[j * 2 + 1]]);
    }
    const levelHashes = await poseidon2HashBatch(levelInputs);
    for (let j = 0; j < half; j++) {
      hashes[j] = levelHashes[j];
    }
    sz = half;
  }

  return hashes[0];
}

// --- Schnorr Signing ---

let _bb: Barretenberg | null = null;

async function getBb(): Promise<Barretenberg> {
  if (!_bb) {
    _bb = await Barretenberg.new();
  }
  return _bb;
}

// Noir evaluates all branches at the circuit level. Disabled signature slots
// still run through schnorr::verify_signature, so we need valid dummy sigs.
let _dummySig: NodeSignatureWitness | null = null;

async function getDummySignature(): Promise<NodeSignatureWitness> {
  if (_dummySig) return _dummySig;
  const bb = await getBb();
  const dummyKey = new Uint8Array(32);
  dummyKey[31] = 0xff;
  const dummyMsg = new Uint8Array(32);
  const { publicKey } = await bb.schnorrComputePublicKey({ privateKey: dummyKey });
  const { s, e } = await bb.schnorrConstructSignature({ message: dummyMsg, privateKey: dummyKey });
  _dummySig = {
    pubkey_x: bytesToHex(publicKey.x),
    pubkey_y: bytesToHex(publicKey.y),
    signature: [...Array.from(s), ...Array.from(e)],
    msg: Array.from(dummyMsg),
    enabled: false,
  };
  return _dummySig;
}

export function emptyMutation(): StateMutationWitness {
  return {
    key: "0",
    new_value: "0",
    is_delete: false,
    enabled: false,
  };
}

/** Sign a message with a Grumpkin private key using Barretenberg's Schnorr. */
export async function schnorrSign(
  privateKey: Uint8Array,
  message: Uint8Array,
): Promise<{ s: Uint8Array; e: Uint8Array; pubkey: { x: Uint8Array; y: Uint8Array } }> {
  const bb = await getBb();
  const { publicKey } = await bb.schnorrComputePublicKey({ privateKey });
  const { s, e } = await bb.schnorrConstructSignature({ message, privateKey });
  return { s, e, pubkey: publicKey };
}

// --- API Types (from Persistia node) ---

interface ApiSignature {
  pubkey: string;
  signature: string;
  message: string;
  schnorr_s?: string;
  schnorr_e?: string;
  grumpkin_x?: string;
  grumpkin_y?: string;
}

interface ApiMutation {
  key: string;
  old_value?: string;
  new_value?: string;
}

interface ApiBlock {
  block_number: number;
  state_root?: string;
  mutations: ApiMutation[];
  signatures: ApiSignature[];
  active_nodes: number;
}

// --- Witness Builders ---

async function buildSignatureWitness(sig: ApiSignature): Promise<NodeSignatureWitness> {
  const msgBytes = Buffer.from(sig.message, "utf-8");
  const msgHash = sha256Hash(msgBytes);

  if (sig.grumpkin_x && sig.grumpkin_y && sig.schnorr_s && sig.schnorr_e) {
    const s = hexToBytes(sig.schnorr_s);
    const e = hexToBytes(sig.schnorr_e);
    return {
      pubkey_x: sig.grumpkin_x,
      pubkey_y: sig.grumpkin_y,
      signature: [...padArray(s, 32), ...padArray(e, 32)],
      msg: msgHash,
      enabled: true,
    };
  }

  throw new Error(
    "Node must provide Schnorr signature fields (grumpkin_x, grumpkin_y, schnorr_s, schnorr_e). " +
    "Use buildTestWitness() for testing with generated keys."
  );
}

export function buildMutationWitness(mut: ApiMutation): StateMutationWitness {
  // Convert string key/value to Field by hashing with SHA-256 and interpreting
  // the first 31 bytes as a BN254 field element (to stay within field range)
  const keyHash = createHash("sha256").update(mut.key).digest();
  const keyField = "0x00" + keyHash.subarray(0, 31).toString("hex");

  const isDelete = mut.new_value == null;
  const valueField = isDelete
    ? "0"
    : "0x00" + createHash("sha256").update(mut.new_value!).digest().subarray(0, 31).toString("hex");

  return {
    key: keyField,
    new_value: valueField,
    is_delete: isDelete,
    enabled: true,
  };
}

/** Build witness for a single-block proof. */
export async function buildSingleBlockWitness(
  block: ApiBlock,
  prevStateRoot: string,
  opts?: {
    prevProvenBlocks?: number;
    prevGenesisRoot?: string;
    prevProof?: string[];
    prevVk?: string[];
    prevKeyHash?: string;
    prevPublicInputs?: string[];
  },
): Promise<CircuitWitness> {
  const sigs: NodeSignatureWitness[] = [];
  for (const sig of block.signatures) {
    // Skip signatures without Schnorr fields (peers whose keys aren't available)
    if (sig.grumpkin_x && sig.grumpkin_y && sig.schnorr_s && sig.schnorr_e) {
      sigs.push(await buildSignatureWitness(sig));
    }
  }
  const dummy = await getDummySignature();
  while (sigs.length < MAX_VALIDATORS) sigs.push({ ...dummy });

  const muts = block.mutations.map(buildMutationWitness);
  while (muts.length < MAX_MUTATIONS) muts.push(emptyMutation());

  const mutSlice = muts.slice(0, MAX_MUTATIONS);

  // Compute Poseidon2 Merkle root to match circuit's computed root
  const computedRoot = await computePoseidon2MerkleRoot(mutSlice);

  return {
    mutations: mutSlice,
    mutation_count: block.mutations.length,
    signatures: sigs.slice(0, MAX_VALIDATORS),
    sig_count: sigs.filter(s => s.enabled).length,
    prev_proven_blocks: opts?.prevProvenBlocks ?? 0,
    prev_genesis_root: opts?.prevGenesisRoot ?? "0",
    prev_proof: opts?.prevProof ?? new Array(PROOF_SIZE).fill("0"),
    prev_vk: opts?.prevVk ?? new Array(VK_SIZE).fill("0"),
    prev_key_hash: opts?.prevKeyHash ?? "0",
    prev_public_inputs: opts?.prevPublicInputs ?? new Array(PUBLIC_INPUTS_SIZE).fill("0"),
    prev_state_root: prevStateRoot,
    new_state_root: computedRoot,
    block_number: block.block_number,
    active_nodes: block.active_nodes,
  };
}

/**
 * Build a test witness with generated Schnorr signatures.
 * Uses random Grumpkin keys -- for benchmarking and testing only.
 */
export async function buildTestWitness(opts?: {
  numValidators?: number;
  numMutations?: number;
  blockNumber?: number;
}): Promise<CircuitWitness> {
  const bb = await getBb();
  const numValidators = opts?.numValidators ?? 1;
  const numMutations = opts?.numMutations ?? 1;
  const blockNumber = opts?.blockNumber ?? 1;

  const sigs: NodeSignatureWitness[] = [];
  const blockMsg = Buffer.from(`block:${blockNumber}`, "utf-8");
  const msgHash = sha256Hash(blockMsg);

  for (let i = 0; i < numValidators; i++) {
    const privateKey = new Uint8Array(32);
    privateKey[31] = i + 1;

    const { publicKey } = await bb.schnorrComputePublicKey({ privateKey });
    const { s, e } = await bb.schnorrConstructSignature({
      message: new Uint8Array(msgHash),
      privateKey,
    });

    sigs.push({
      pubkey_x: bytesToHex(publicKey.x),
      pubkey_y: bytesToHex(publicKey.y),
      signature: [...Array.from(s), ...Array.from(e)],
      msg: msgHash,
      enabled: true,
    });
  }
  const dummy = await getDummySignature();
  while (sigs.length < MAX_VALIDATORS) sigs.push({ ...dummy });

  const muts: StateMutationWitness[] = [];
  for (let i = 0; i < numMutations; i++) {
    muts.push({
      key: `${i + 1}`,
      new_value: `${(i + 1) * 100}`,
      is_delete: false,
      enabled: true,
    });
  }
  while (muts.length < MAX_MUTATIONS) muts.push(emptyMutation());

  const mutSlice = muts.slice(0, MAX_MUTATIONS);

  // Compute Poseidon2 Merkle root to match circuit's computed root
  const computedRoot = await computePoseidon2MerkleRoot(mutSlice);

  return {
    mutations: mutSlice,
    mutation_count: numMutations,
    signatures: sigs.slice(0, MAX_VALIDATORS),
    sig_count: numValidators,
    prev_proven_blocks: 0,
    prev_genesis_root: "0",
    prev_proof: new Array(PROOF_SIZE).fill("0"),
    prev_vk: new Array(VK_SIZE).fill("0"),
    prev_key_hash: "0",
    prev_public_inputs: new Array(PUBLIC_INPUTS_SIZE).fill("0"),
    prev_state_root: "0xaa",
    new_state_root: computedRoot,
    block_number: blockNumber,
    active_nodes: numValidators,
  };
}

/** Clean up Barretenberg instance. */
export async function destroyBb(): Promise<void> {
  if (_bb) {
    await _bb.destroy();
    _bb = null;
  }
}
