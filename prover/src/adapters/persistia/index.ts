// Persistia adapter for zkMetal — reference implementation.
//
// Connects the generic zkMetal proving pipeline to a Persistia node,
// fetching committed blocks and submitting proofs via the node API.

import type { DataSource, WitnessBuilder, ProofSink, ProofOutput, RecursiveProofInputs } from "../../types.js";
import {
  buildSingleBlockWitness,
  buildMutationWitness,
  emptyMutation,
  computePoseidon2MerkleRoot,
  type CircuitWitness,
} from "../../witness.js";

// --- API Types ---

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

export interface PersistiaBlock {
  block_number: number;
  state_root?: string;
  mutations: ApiMutation[];
  signatures: ApiSignature[];
  active_nodes: number;
}

// --- URL helper ---

function nodeUrl(base: string, path: string): string {
  try {
    const u = new URL(base);
    const [pathname, query] = path.split("?", 2);
    u.pathname = u.pathname.replace(/\/$/, "") + pathname;
    if (query) {
      for (const param of query.split("&")) {
        const [k, v] = param.split("=", 2);
        u.searchParams.set(k, v ?? "");
      }
    }
    return u.toString();
  } catch {
    return `${base}${path}`;
  }
}

// --- State root computation ---

const GENESIS_STATE_ROOT = "0";

async function computeBlockStateRoot(block: PersistiaBlock): Promise<string> {
  if (!block.mutations || block.mutations.length === 0) return GENESIS_STATE_ROOT;
  const muts = block.mutations.map(buildMutationWitness);
  while (muts.length < 32) muts.push(emptyMutation());
  return computePoseidon2MerkleRoot(muts.slice(0, 32));
}

// --- Persistia DataSource ---

export class PersistiaDataSource implements DataSource<PersistiaBlock> {
  constructor(private nodeBase: string) {}

  async fetchBlock(blockNumber: number): Promise<PersistiaBlock> {
    const url = nodeUrl(this.nodeBase, `/proof/block?block=${blockNumber}`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch block ${blockNumber}: ${res.status}`);
    return res.json() as Promise<PersistiaBlock>;
  }

  async fetchLatestBlockNumber(): Promise<number> {
    const url = nodeUrl(this.nodeBase, "/proof/zk/status");
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch ZK status: ${res.status}`);
    const data = await res.json() as any;
    return data.last_committed_round ?? data.latest_block ?? data.latestBlock ?? 0;
  }
}

// --- Persistia WitnessBuilder ---

export class PersistiaWitnessBuilder implements WitnessBuilder<PersistiaBlock> {
  private prevStateRoots = new Map<number, string>();

  constructor(private nodeBase: string) {}

  async buildWitness(
    block: PersistiaBlock,
    blockNumber: number,
    recursiveOpts?: RecursiveProofInputs,
  ): Promise<Record<string, unknown>> {
    // In recursive mode, extract prev_state_root from the previous proof's
    // public inputs (index 1 = new_state_root) to avoid re-fetching/recomputing.
    // The circuit enforces this matches via the state root continuity assertion.
    let prevStateRoot: string;
    if (recursiveOpts?.prevPublicInputs && recursiveOpts.prevPublicInputs.length > 1) {
      prevStateRoot = recursiveOpts.prevPublicInputs[1];
      this.prevStateRoots.set(blockNumber - 1, prevStateRoot);
    } else {
      prevStateRoot = await this.fetchPrevStateRoot(blockNumber);
    }

    const opts = recursiveOpts ? {
      prevProvenBlocks: recursiveOpts.prevProvenBlocks,
      prevGenesisRoot: recursiveOpts.prevGenesisRoot,
      prevProof: recursiveOpts.prevProof,
      prevVk: recursiveOpts.prevVk,
      prevKeyHash: recursiveOpts.prevKeyHash,
      prevPublicInputs: recursiveOpts.prevPublicInputs,
    } : undefined;

    const witness = await buildSingleBlockWitness(block as any, prevStateRoot, opts);

    // Cache this block's state root for the next block
    this.prevStateRoots.set(blockNumber, witness.new_state_root);

    return witness as unknown as Record<string, unknown>;
  }

  private async fetchPrevStateRoot(blockNumber: number): Promise<string> {
    if (blockNumber <= 1) return GENESIS_STATE_ROOT;

    // Check cache first
    const cached = this.prevStateRoots.get(blockNumber - 1);
    if (cached) return cached;

    // Fetch and compute from previous block
    try {
      const url = nodeUrl(this.nodeBase, `/proof/block?block=${blockNumber - 1}`);
      const res = await fetch(url);
      if (!res.ok) return GENESIS_STATE_ROOT;
      const prevBlock = await res.json() as PersistiaBlock;
      return computeBlockStateRoot(prevBlock);
    } catch {
      return GENESIS_STATE_ROOT;
    }
  }
}

// --- Persistia ProofSink ---

export class PersistiaProofSink implements ProofSink {
  constructor(private nodeBase: string) {}

  async submitProof(proof: ProofOutput): Promise<void> {
    const url = nodeUrl(this.nodeBase, "/proof/zk/submit");
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        proof: proof.proof,
        publicInputs: proof.publicInputs,
        vkAsFields: proof.vkAsFields,
        vkHash: proof.vkHash,
        vk: proof.vk,
        block_number: proof.blockNumber,
        proven_blocks: proof.provenBlocks,
        state_root: proof.meta?.state_root,
        genesis_root: proof.meta?.genesis_root,
        proof_type: "noir-ultrahonk",
        prover: proof.prover,
        timestamp: proof.timestamp,
      }),
    });
    if (!res.ok) throw new Error(`Failed to submit proof: ${res.status}`);
  }
}

/** Create all three Persistia adapter components from a node URL. */
export function createPersistiaAdapter(nodeBase: string) {
  return {
    dataSource: new PersistiaDataSource(nodeBase),
    witnessBuilder: new PersistiaWitnessBuilder(nodeBase),
    proofSink: new PersistiaProofSink(nodeBase),
  };
}
