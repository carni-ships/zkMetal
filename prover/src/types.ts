// Core interfaces for the zkMetal proving SDK.
//
// Implement these to integrate any Noir circuit with the proving pipeline.

/** Configuration for the proving engine. */
export interface ProverConfig {
  /** Path to compiled Noir circuit JSON (nargo compile output). */
  circuitPath: string;
  /** Number of WASM threads for Barretenberg (default: 8). */
  threads?: number;
  /** Path to native bb CLI binary (default: ~/.bb/bb). */
  bbPath?: string;
  /** Directory for caching verification keys. */
  vkCacheDir?: string;
}

/** A block or unit of work to be proven. Generic over the block type. */
export interface DataSource<B = unknown> {
  /** Fetch a specific block/unit by number. */
  fetchBlock(blockNumber: number): Promise<B>;
  /** Get the latest available block number. */
  fetchLatestBlockNumber(): Promise<number>;
}

/** Transforms a raw block into the witness map the circuit expects. */
export interface WitnessBuilder<B = unknown> {
  /**
   * Build the witness inputs for the circuit from a block.
   * Returns a plain object that will be passed to `noir.execute(witness)`.
   */
  buildWitness(
    block: B,
    blockNumber: number,
    opts?: RecursiveProofInputs,
  ): Promise<Record<string, unknown>>;
}

/** Optional: where to send proofs after generation. */
export interface ProofSink {
  /** Submit a generated proof. */
  submitProof(proof: ProofOutput): Promise<void>;
}

/** Inputs for recursive IVC chaining. */
export interface RecursiveProofInputs {
  prevProvenBlocks: number;
  prevGenesisRoot: string;
  prevProof: string[];
  prevVk: string[];
  prevKeyHash: string;
  prevPublicInputs: string[];
}

/** Output from proof generation. */
export interface ProofOutput {
  /** Base64-encoded proof bytes. */
  proof: string;
  /** Public inputs as hex field strings. */
  publicInputs: string[];
  /** VK fields for recursive chaining (optional). */
  vkAsFields?: string[];
  /** VK hash for recursive chaining (optional). */
  vkHash?: string;
  /** Base64-encoded verification key (optional, from native bb). */
  vk?: string;
  /** Block number this proof covers. */
  blockNumber: number;
  /** Number of blocks in the recursive chain. */
  provenBlocks: number;
  /** Prover identifier string. */
  prover: string;
  /** ISO timestamp. */
  timestamp: string;
  /** Application-specific metadata. */
  meta?: Record<string, unknown>;
}

/** Options for the watch loop. */
export interface WatchOptions {
  /** Poll interval in seconds (default: 10). */
  intervalSec?: number;
  /** Starting block number (default: latest from data source). */
  startBlock?: number;
  /** Use native bb CLI for proving (default: false). */
  useNative?: boolean;
  /** Enable recursive IVC chaining (default: false). */
  recursive?: boolean;
  /** Directory to write proof JSON files. */
  proofDir?: string;
}
