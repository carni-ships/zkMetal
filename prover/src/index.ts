// zkMetal — Generic Noir prover SDK.
//
// Use this to integrate any Noir circuit with a production proving pipeline.
//
// Example:
//   import { ProverEngine, watchSequential } from "zkmetal";
//
//   const engine = new ProverEngine({ circuitPath: "./target/my_circuit.json" });
//   await watchSequential(engine, myDataSource, myWitnessBuilder, myProofSink);

// Core engine
export { ProverEngine, proofToFields, extractInnerProof } from "./engine.js";
export type { VerifierTarget } from "./engine.js";

// Worker pool
export { PersistentBb } from "./persistent-bb.js";

// Watch loops
export { watchSequential, watchPipelined, watchParallelMsgpack } from "./watcher.js";

// Types
export type {
  ProverConfig,
  DataSource,
  WitnessBuilder,
  ProofSink,
  ProofOutput,
  RecursiveProofInputs,
  WatchOptions,
} from "./types.js";
