// ProverEngine — generic Noir circuit proving engine.
//
// Loads any compiled Noir circuit, executes witnesses, generates and verifies
// UltraHonk proofs via Barretenberg (WASM or native bb CLI).

import { Noir } from "@noir-lang/noir_js";
import { Barretenberg, UltraHonkBackend } from "@aztec/bb.js";
import { readFileSync, writeFileSync, existsSync, mkdirSync, rmSync } from "fs";
import { join, resolve } from "path";
import { execSync, exec } from "child_process";
import type { ProverConfig, ProofOutput, RecursiveProofInputs } from "./types.js";

const PROOF_OVERHEAD_FIELDS = 51;
const DEFAULT_BB_PATH = join(process.env.HOME ?? "~", ".bb", "bb");
const DEFAULT_METAL_MSM_PATH = join(process.env.HOME ?? "~", ".zkmsm", "zkmsm");

/** Convert raw proof bytes to array of hex field strings. */
export function proofToFields(proofBytes: Uint8Array): string[] {
  const fields: string[] = [];
  for (let i = 0; i < proofBytes.length; i += 32) {
    const chunk = proofBytes.slice(i, i + 32);
    fields.push("0x" + Array.from(chunk).map(b => b.toString(16).padStart(2, "0")).join(""));
  }
  return fields;
}

/** Extract inner proof fields (skip overhead) for recursive verification. */
export function extractInnerProof(proofBytes: Uint8Array): string[] {
  return proofToFields(proofBytes).slice(PROOF_OVERHEAD_FIELDS);
}

export type VerifierTarget = "noir-recursive-no-zk" | "evm-no-zk";

export class ProverEngine {
  private config: Required<ProverConfig>;
  private circuit: any;
  private noir: Noir | null = null;
  private backend: UltraHonkBackend | null = null;
  private api: Barretenberg | null = null;

  constructor(config: ProverConfig) {
    this.config = {
      circuitPath: config.circuitPath,
      threads: config.threads ?? 8,
      bbPath: config.bbPath ?? DEFAULT_BB_PATH,
      vkCacheDir: config.vkCacheDir ?? resolve(config.circuitPath, "../../target/bb_vk"),
      metalMsmPath: config.metalMsmPath ?? DEFAULT_METAL_MSM_PATH,
    };
    this.circuit = this.loadCircuit();
  }

  private loadCircuit() {
    if (!existsSync(this.config.circuitPath)) {
      throw new Error(`Circuit not found at ${this.config.circuitPath}. Run 'nargo compile' first.`);
    }
    return JSON.parse(readFileSync(this.config.circuitPath, "utf-8"));
  }

  /** Initialize WASM backend (lazy, cached). */
  async init(): Promise<void> {
    if (this.noir) return;
    this.api = await Barretenberg.new({ threads: this.config.threads });
    this.backend = new UltraHonkBackend(this.circuit.bytecode, this.api);
    this.noir = new Noir(this.circuit);
  }

  /** Execute witness (solve constraints without generating proof). */
  async execute(witness: Record<string, unknown>): Promise<{ witness: Uint8Array; returnValue: any }> {
    await this.init();
    return this.noir!.execute(witness as any);
  }

  /** Generate a proof from a solved witness (WASM backend). */
  async prove(solvedWitness: Uint8Array): Promise<{ proof: Uint8Array; publicInputs: string[] }> {
    await this.init();
    return this.backend!.generateProof(solvedWitness);
  }

  /** Verify a proof (WASM backend). */
  async verify(proof: { proof: Uint8Array; publicInputs: string[] }): Promise<boolean> {
    await this.init();
    return this.backend!.verifyProof(proof);
  }

  /** Generate recursive proof artifacts (VK fields + hash) for IVC chaining. */
  async generateRecursiveArtifacts(
    proof: { proof: Uint8Array; publicInputs: string[] },
  ): Promise<{ vkAsFields: string[]; vkHash: string }> {
    await this.init();
    return this.backend!.generateRecursiveProofArtifacts(proof, proof.publicInputs.length);
  }

  // --- Native bb CLI ---

  /** Check if native bb binary is available. */
  nativeBbAvailable(): boolean {
    try {
      execSync(`${this.config.bbPath} --version`, { stdio: "pipe" });
      return true;
    } catch {
      return false;
    }
  }

  /** Check if Metal GPU MSM binary is available (Apple Silicon only). */
  metalMsmAvailable(): boolean {
    try {
      execSync(`${this.config.metalMsmPath} --info`, { stdio: "pipe" });
      return true;
    } catch {
      return false;
    }
  }

  /** Get Metal GPU information. Returns null if not available. */
  metalGpuInfo(): { gpu: string; unified_memory: boolean } | null {
    try {
      const output = execSync(`${this.config.metalMsmPath} --info`, { stdio: "pipe" }).toString();
      return JSON.parse(output);
    } catch {
      return null;
    }
  }

  /**
   * Compute a multi-scalar multiplication on the Metal GPU.
   * Points are [x, y] hex pairs (affine BN254), scalars are 256-bit hex strings.
   * Returns the result point as {x, y} hex strings, or null if Metal is unavailable.
   */
  metalMsm(
    points: [string, string][],
    scalars: string[],
  ): { x: string; y: string; infinity: boolean; time_ms: number } | null {
    if (!this.metalMsmAvailable()) return null;
    const input = JSON.stringify({ points, scalars });
    try {
      const output = execSync(`${this.config.metalMsmPath} --msm`, {
        input,
        stdio: ["pipe", "pipe", "pipe"],
        timeout: 60_000,
      }).toString();
      return JSON.parse(output);
    } catch {
      return null;
    }
  }

  /** Ensure VK is cached for native bb proving. Returns VK file path or null. */
  ensureVkCached(target: VerifierTarget = "noir-recursive-no-zk"): string | null {
    const cacheDir = target === "evm-no-zk"
      ? resolve(this.config.vkCacheDir, "../bb_vk_evm")
      : this.config.vkCacheDir;
    const cacheFile = join(cacheDir, "vk");
    if (existsSync(cacheFile)) return cacheFile;
    try {
      mkdirSync(cacheDir, { recursive: true });
      execSync(
        `${this.config.bbPath} write_vk -b ${this.config.circuitPath} -o ${cacheDir} -t ${target}`,
        { stdio: "pipe" },
      );
      if (existsSync(cacheFile)) return cacheFile;
    } catch {}
    return null;
  }

  /** Generate proof using native bb CLI (faster than WASM). */
  nativeProve(
    solvedWitness: Uint8Array,
    target: VerifierTarget = "noir-recursive-no-zk",
    tmpPrefix = "/tmp/zkmetal_bb",
  ): { proof: Uint8Array; publicInputs: string[]; vk?: Uint8Array } {
    const tmpDir = `${tmpPrefix}_prove`;
    rmSync(tmpDir, { recursive: true, force: true });

    const vkPath = this.ensureVkCached(target);
    const vkFlag = vkPath ? ` -k ${vkPath}` : " --write_vk";

    const witnessPath = `${tmpPrefix}_witness.gz`;
    writeFileSync(witnessPath, solvedWitness);
    execSync(
      `${this.config.bbPath} prove -b ${this.config.circuitPath} -w ${witnessPath} -o ${tmpDir}${vkFlag} -t ${target}`,
      { stdio: "pipe" },
    );

    const proof = readFileSync(join(tmpDir, "proof"));
    const piRaw = readFileSync(join(tmpDir, "public_inputs"));
    const vk = existsSync(join(tmpDir, "vk")) ? readFileSync(join(tmpDir, "vk")) : undefined;

    const publicInputs: string[] = [];
    for (let i = 0; i < piRaw.length; i += 32) {
      publicInputs.push(
        "0x" + Array.from(piRaw.subarray(i, i + 32)).map(b => b.toString(16).padStart(2, "0")).join(""),
      );
    }

    // Cleanup
    rmSync(witnessPath, { force: true });
    rmSync(tmpDir, { recursive: true, force: true });

    return { proof: new Uint8Array(proof), publicInputs, vk: vk ? new Uint8Array(vk) : undefined };
  }

  /** Verify proof using native bb CLI. */
  nativeVerify(proof: Uint8Array, publicInputs: string[], vk: Uint8Array): boolean {
    const tmpDir = "/tmp/zkmetal_bb_verify";
    rmSync(tmpDir, { recursive: true, force: true });
    mkdirSync(tmpDir, { recursive: true });

    writeFileSync(join(tmpDir, "proof"), proof);
    writeFileSync(join(tmpDir, "vk"), vk);

    const piBytes = Buffer.alloc(publicInputs.length * 32);
    for (let i = 0; i < publicInputs.length; i++) {
      const val = BigInt(publicInputs[i]);
      const hex = val.toString(16).padStart(64, "0");
      for (let j = 0; j < 32; j++) {
        piBytes[i * 32 + j] = parseInt(hex.substring(j * 2, j * 2 + 2), 16);
      }
    }
    writeFileSync(join(tmpDir, "public_inputs"), piBytes);

    try {
      execSync(
        `${this.config.bbPath} verify -p ${tmpDir}/proof -k ${tmpDir}/vk -i ${tmpDir}/public_inputs -t noir-recursive-no-zk`,
        { stdio: "pipe" },
      );
      return true;
    } catch {
      return false;
    }
  }

  /** Start a native bb prove as a non-blocking child process. Returns a promise. */
  nativeProveAsync(
    solvedWitness: Uint8Array,
    target: VerifierTarget = "noir-recursive-no-zk",
    tmpPrefix = "/tmp/zkmetal_bb",
  ): Promise<{ proof: Uint8Array; publicInputs: string[]; vk?: Uint8Array }> {
    const tmpDir = `${tmpPrefix}_prove`;
    rmSync(tmpDir, { recursive: true, force: true });

    const vkPath = this.ensureVkCached(target);
    const vkFlag = vkPath ? ` -k ${vkPath}` : " --write_vk";

    const witnessPath = `${tmpPrefix}_witness.gz`;
    writeFileSync(witnessPath, solvedWitness);

    return new Promise((resolve, reject) => {
      exec(
        `${this.config.bbPath} prove -b ${this.config.circuitPath} -w ${witnessPath} -o ${tmpDir}${vkFlag} -t ${target}`,
        { env: process.env },
        (err) => {
          if (err) return reject(err);

          const proof = readFileSync(join(tmpDir, "proof"));
          const piRaw = readFileSync(join(tmpDir, "public_inputs"));
          const vk = existsSync(join(tmpDir, "vk")) ? readFileSync(join(tmpDir, "vk")) : undefined;

          const publicInputs: string[] = [];
          for (let i = 0; i < piRaw.length; i += 32) {
            publicInputs.push(
              "0x" + Array.from(piRaw.subarray(i, i + 32)).map(b => b.toString(16).padStart(2, "0")).join(""),
            );
          }

          rmSync(witnessPath, { force: true });
          rmSync(tmpDir, { recursive: true, force: true });

          resolve({
            proof: new Uint8Array(proof),
            publicInputs,
            vk: vk ? new Uint8Array(vk) : undefined,
          });
        },
      );
    });
  }

  /** Get the raw circuit object (for Noir instance creation in workers). */
  getCircuit(): any {
    return this.circuit;
  }

  /** Get config. */
  getConfig(): Required<ProverConfig> {
    return this.config;
  }

  /** Clean up WASM resources. */
  async destroy(): Promise<void> {
    if (this.backend && typeof this.backend.destroy === "function") await this.backend.destroy();
    if (this.api && typeof this.api.destroy === "function") await this.api.destroy();
    this.backend = null;
    this.api = null;
    this.noir = null;
  }
}
