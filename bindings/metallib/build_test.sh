#!/bin/bash
# build_test.sh — Build and run the metallib test
#
# Usage:
#   ./build_metallib.sh   # first, build the metallib
#   ./build_test.sh       # then, build and run the test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

METALLIB_PATH="$SCRIPT_DIR/zkmetal.metallib"
if [ ! -f "$METALLIB_PATH" ]; then
    echo "zkmetal.metallib not found. Building it first..."
    bash "$SCRIPT_DIR/build_metallib.sh"
fi

echo "Compiling MetalLib loader + test..."
clang -O2 -fobjc-arc \
    -framework Metal -framework Foundation \
    -o "$SCRIPT_DIR/test_metallib" \
    "$SCRIPT_DIR/MetalLibLoader.m" \
    "$SCRIPT_DIR/test_metallib.m"

echo "Running test..."
echo ""
"$SCRIPT_DIR/test_metallib" "$METALLIB_PATH"
