// PersistentBb — keeps a bb process resident via the msgpack API.
//
// Eliminates ~40ms startup overhead per prove by reusing a long-lived
// bb process. Communicates via stdin/stdout with length-prefixed msgpack.

import { spawn, type ChildProcess } from "child_process";
import { Encoder, Decoder } from "msgpackr";
import { join } from "path";

const DEFAULT_BB_PATH = join(process.env.HOME ?? "~", ".bb", "bb");

const msgpackEncoder = new Encoder({ useRecords: false });
const msgpackDecoder = new Decoder({ useRecords: false });

export class PersistentBb {
  private proc: ChildProcess;
  private pendingResolves: Array<{ resolve: (v: any) => void; reject: (e: Error) => void }> = [];
  private buffer = Buffer.alloc(0);
  private readingLength = true;
  private expectedLength = 0;

  constructor(threads: number, bbPath = DEFAULT_BB_PATH) {
    this.proc = spawn(bbPath, ["msgpack", "run"], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, OMP_NUM_THREADS: String(threads) },
    });
    this.proc.stdout!.on("data", (d: Buffer) => this.handleData(d));
    this.proc.on("error", (e) => {
      if (this.pendingResolves.length > 0) this.pendingResolves.shift()!.reject(e);
    });
  }

  private handleData(data: Buffer) {
    this.buffer = Buffer.concat([this.buffer, data]);
    while (true) {
      if (this.readingLength) {
        if (this.buffer.length >= 4) {
          this.expectedLength = this.buffer.readUInt32LE(0);
          this.buffer = this.buffer.subarray(4);
          this.readingLength = false;
        } else break;
      } else {
        if (this.buffer.length >= this.expectedLength) {
          const payload = this.buffer.subarray(0, this.expectedLength);
          this.buffer = this.buffer.subarray(this.expectedLength);
          this.readingLength = true;
          const resp = msgpackDecoder.unpack(payload);
          if (this.pendingResolves.length > 0) this.pendingResolves.shift()!.resolve(resp);
        } else break;
      }
    }
  }

  /** Prove a circuit via the persistent bb process. */
  prove(
    bytecode: Buffer,
    vk: Buffer,
    witness: Buffer,
    circuitName = "circuit",
  ): Promise<any> {
    return new Promise((resolve, reject) => {
      this.pendingResolves.push({ resolve, reject });
      const cmd = [["CircuitProve", {
        circuit: { name: circuitName, bytecode, verification_key: vk },
        witness,
        settings: {
          ipa_accumulation: false,
          oracle_hash_type: "poseidon2",
          disable_zk: true,
          optimized_solidity_verifier: false,
        },
      }]];
      const packed = msgpackEncoder.pack(cmd);
      const lenBuf = Buffer.alloc(4);
      lenBuf.writeUInt32LE(packed.length, 0);
      this.proc.stdin!.write(lenBuf);
      this.proc.stdin!.write(packed);
    });
  }

  /** Gracefully shut down the bb process. */
  shutdown(): Promise<void> {
    const packed = msgpackEncoder.pack([["Shutdown", {}]]);
    const lenBuf = Buffer.alloc(4);
    lenBuf.writeUInt32LE(packed.length, 0);
    this.proc.stdin!.write(lenBuf);
    this.proc.stdin!.write(packed);
    this.proc.stdin!.end();
    return new Promise((r) => this.proc.on("close", r));
  }
}
