#!/bin/bash
# build_circle_stark.sh — Build and benchmark Circle STARK with Poseidon2 ANE modes
#
# This builds only the ANE-enabled Circle STARK components:
#   - ANEOps (ANE Poseidon2 kernels)
#   - NeonFieldOps (SIMD field arithmetic)
#   - zkMetal (Circle STARK prover with Poseidon2)
#   - Minimal benchmark tool
#
# Usage:
#   ./build_circle_stark.sh          # build only
#   ./build_circle_stark.sh bench    # build and run benchmark
#   ./build_circle_stark.sh test     # build and run tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Circle STARK Poseidon2 Build ==="
echo ""

# Use a separate build directory for incremental builds
BUILD_DIR=".build/circlestark"
mkdir -p "$BUILD_DIR"

# Build only the targets we need with separate build dir for isolation
echo "Building ANEOps, NeonFieldOps, zkMetal, and benchmark..."
swift build \
    --scratch-path "$BUILD_DIR" \
    --target NeonFieldOps \
    --target ANEOps \
    --target zkMetal \
    --target zkbench 2>&1

BUILD_STATUS=${PIPESTATUS[0]}

if [ $BUILD_STATUS -ne 0 ]; then
    echo "Build failed!"
    exit $BUILD_STATUS
fi

echo ""
echo "Build successful!"

# If argument provided, run that command
if [ $# -gt 0 ]; then
    case "$1" in
        bench|benchmark)
            echo ""
            echo "=== Running Circle STARK Poseidon2 Benchmark ==="
            "$BUILD_DIR/debug/zkbench" cstark-all
            ;;
        test|tests)
            echo ""
            echo "=== Running Circle STARK Tests ==="
            swift build --scratch-path "$BUILD_DIR" --target zkMetalTests
            "$BUILD_DIR/debug/zkMetalTests" circlestark
            ;;
        clean)
            echo "Cleaning build artifacts..."
            rm -rf "$BUILD_DIR"
            echo "Clean complete."
            ;;
        *)
            echo "Unknown command: $1"
            echo "Usage: $0 [bench|test|clean]"
            exit 1
            ;;
    esac
fi
