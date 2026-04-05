# zkMetal Barretenberg Integration

Drop-in Metal GPU acceleration for [Aztec Barretenberg](https://github.com/AztecProtocol/barretenberg) on Apple Silicon.

## What gets accelerated

| Primitive | BB function | zkMetal replacement | Speedup |
|-----------|------------|---------------------|---------|
| MSM (BN254 G1) | `scalar_multiplication::pippenger` | `bn254_pippenger_msm` (Pippenger + Metal GPU) | 3-10x |
| MSM (Grumpkin) | `scalar_multiplication::pippenger` | `grumpkin_pippenger_msm` | 3-10x |
| NTT/iNTT | `polynomial_arithmetic::fft` | `bn254_fr_ntt` / `bn254_fr_intt` | 2-5x |
| Poseidon2 | `poseidon2::hash` | `poseidon2_hash_cpu` (NEON + multi-threaded) | 2-4x |
| Field arithmetic | `fr::mul`, `fr::sqr`, etc. | ARM64 CIOS Montgomery asm | 1.3-2x |
| KZG pairing | `pairing::reduced_ate_pairing` | `bn254_pairing` (C tower arithmetic) | 2-5x |

## Prerequisites

1. Apple Silicon Mac (M1/M2/M3/M4)
2. zkMetal built and installed:
   ```bash
   cd zkMetal/bindings/c
   ./build.sh install          # installs to /usr/local by default
   # or: PREFIX=/opt/zkmetal ./build.sh install
   ```

## Integration (two options)

### Option A: CMake find_package

Copy `FindZkMetal.cmake` into your BB build's `cmake/` directory, then add to your root `CMakeLists.txt`:

```cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(ZkMetal)

if(ZkMetal_FOUND)
    target_link_libraries(barretenberg PRIVATE ZkMetal::ZkMetal)
endif()
```

### Option B: Apply the integration patch

```bash
cd barretenberg
git apply /path/to/zkMetal/bindings/barretenberg/integration_patch.diff
```

This modifies:
- Root `CMakeLists.txt` -- adds `find_package(ZkMetal)`
- `cpp/src/barretenberg/ecc/scalar_multiplication/scalar_multiplication.cpp` -- MSM gate
- `cpp/src/barretenberg/polynomials/polynomial_arithmetic.cpp` -- NTT gate
- `cpp/src/barretenberg/crypto/poseidon2/poseidon2.cpp` -- hash gate

All changes are gated behind `#ifdef HAS_ZKMETAL` so BB builds cleanly without zkMetal.

## Verifying it works

```bash
cd barretenberg/cpp
cmake --preset default -DCMAKE_PREFIX_PATH=/usr/local
cmake --build --preset default
./bin/ultra_honk_bench  # should show GPU MSM in timing output
```

## Files in this directory

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | CMake target definition for `ZkMetal::ZkMetal` |
| `FindZkMetal.cmake` | Standard CMake find module |
| `zkmetal_bb_bridge.h` | C++ wrapper header over zkMetal's C API |
| `zkmetal_bb_bridge.cpp` | Bridge implementation (GPU init, format conversion) |
| `zkmetal_bb_msm.hpp` | Inline MSM integration matching BB's Pippenger signature |
| `zkmetal_bb_ntt.hpp` | Inline NTT integration matching BB's FFT signature |
| `integration_patch.diff` | Minimal diff to wire into BB source tree |
