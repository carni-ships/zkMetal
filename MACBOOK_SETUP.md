# Running the Optimized Barretenberg Prover on MacBooks

This guide covers building and running the GPU-accelerated Barretenberg UltraHonk prover on Apple Silicon and Intel MacBooks.

## What You Get

The optimized build includes Metal GPU acceleration for:
- **Multi-Scalar Multiplication (MSM)**: ~55ms for 201K points (vs ~267ms CPU-only)
- **Polynomial operations**: Gemini folds, partial evaluation, batch operations
- **Overall proof time**: ~855ms per 428K-gate circuit (vs ~3,850ms without GPU)

Metal GPU is **automatically enabled** when building on macOS — no flags needed.

## Prerequisites

```bash
# Required
brew install cmake ninja llvm@20

# Verify versions
cmake --version    # >= 3.24
ninja --version
clang --version    # >= 16 (system Xcode clang works)

# Node.js (for the prover SDK)
brew install node  # >= 18
```

## Building the `bb` Binary

```bash
cd barretenberg/cpp

# Option 1: Bootstrap script (recommended)
AVM=0 ./bootstrap.sh build_native

# Option 2: CMake directly (more control)
cmake --preset default
cd build && ninja bb

# The binary is at:
# barretenberg/cpp/build/bin/bb
```

### Build Presets

| Preset | Use Case |
|--------|----------|
| `default` | Native build with system Clang |
| `homebrew` | Build with Homebrew LLVM 20 |
| `arm64-macos` | Cross-compile for Apple Silicon |
| `amd64-macos` | Cross-compile for Intel Mac |

### Verify Metal GPU is Working

```bash
# Check the binary links against Metal framework
otool -L build/bin/bb | grep Metal
# Should show: /System/Library/Frameworks/Metal.framework/...

# Verify Metal shaders are in the build directory
ls build/bin/bn254.metal build/bin/prover_ops.metal
```

## Installing the Binary

The prover SDK expects `bb` at `~/.bb/bb`:

```bash
mkdir -p ~/.bb
cp barretenberg/cpp/build/bin/bb ~/.bb/bb

# Also copy the Metal shader files (required at runtime)
cp barretenberg/cpp/build/bin/bn254.metal ~/.bb/
cp barretenberg/cpp/build/bin/prover_ops.metal ~/.bb/
```

Alternatively, set a custom path when using the SDK:

```typescript
const engine = new ProverEngine({
  circuitPath: "./target/my_circuit.json",
  bbPath: "/path/to/your/bb",
});
```

## Metal Shader Runtime

The `bb` binary loads Metal shaders at runtime. It searches for `.metal` files in this order:

1. Same directory as the `bb` binary
2. Current working directory
3. Path specified by `BB_METAL_SHADER_PATH` environment variable

If shaders aren't found, GPU acceleration silently falls back to CPU. To force a specific path:

```bash
export BB_METAL_SHADER_PATH=/path/to/shaders
```

## Using the Prover

### Quick Prove (CLI)

```bash
cd prover
npm install

# Single block proof
npx tsx src/cli.ts prove --node https://your-node.com --block 100

# Continuous proving
npx tsx src/cli.ts watch --node https://your-node.com --mode sequential --recursive
```

### SDK Usage

```typescript
import { ProverEngine } from "zkmetal";

const engine = new ProverEngine({
  circuitPath: "./target/my_circuit.json",
  threads: 8,  // CPU threads (GPU uses all cores automatically)
});

// Check GPU is available
const hasNative = engine.hasNativeBb();
console.log(`Native bb available: ${hasNative}`);
```

### Parallel Proving (Maximum Throughput)

```bash
# 6 persistent bb workers with msgpack IPC
npx tsx src/cli.ts watch --node https://your-node.com --mode parallel --workers 6
```

## Performance Expectations

On Apple M3 Pro (12-core CPU, 18-core GPU):

| Metric | Value |
|--------|-------|
| Single proof (cold) | ~855ms |
| Single proof (cached VK) | ~795ms |
| Sustained throughput | ~70 proofs/min |
| GPU MSM (201K points) | ~55ms |
| Peak memory | ~308 MiB |

### GPU vs CPU Breakdown

| Component | Time | Bound By |
|-----------|------|----------|
| OinkProver (GPU MSMs) | 316ms | GPU |
| Sumcheck | 145ms | CPU |
| Gemini folds (GPU) | 84ms | GPU |
| Shplonk quotient | 62ms | CPU |
| CommitmentKey (GPU) | 62ms | GPU |
| KZG commit (GPU) | 52ms | GPU |

**GPU handles ~54% of total proof time.**

## Troubleshooting

### Build fails with missing Metal framework

This shouldn't happen on macOS, but if it does:

```bash
# Ensure Xcode command line tools are installed
xcode-select --install

# Check Metal is available
xcrun metal --version
```

### GPU acceleration not activating

```bash
# Check Metal device is available
system_profiler SPDisplaysDataType | grep "Metal"

# The bb binary dispatches to GPU only when:
# - MSM has >= 32,768 points
# - Polynomial operations have >= 8,192 elements
# Smaller operations stay on CPU (faster due to dispatch overhead)
```

### Thermal throttling

On sustained proving, MacBook GPUs thermally throttle. For benchmarking:

```bash
# Allow 30-60 seconds cooldown between benchmark runs
# Use external cooling for sustained proving workloads
# Monitor GPU temperature:
sudo powermetrics --samplers gpu_power -i 1000
```

### Intel Mac Notes

Metal GPU acceleration works on Intel Macs with discrete AMD GPUs. Performance will differ from Apple Silicon numbers above. The build process is identical.

## zkMetal GPU Primitives Library

The `metal/` directory contains a standalone Swift GPU library with optimized primitives:

```bash
cd metal
swift build
.build/debug/zkbench all  # Run all benchmarks
```

Available primitives: NTT, MSM, Poseidon2, Keccak-256, FRI, Sumcheck, Polynomial operations, Radix sort — all GPU-accelerated for BN254, BabyBear, and Goldilocks fields.
