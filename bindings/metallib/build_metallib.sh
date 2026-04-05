#!/bin/bash
# build_metallib.sh — Compile all zkMetal GPU shaders into a single zkmetal.metallib
#
# Produces a precompiled Metal library that FFI consumers (Rust, Go, Python)
# can load at runtime via MTLDevice.makeLibrary(filepath:) without needing
# Xcode's runtime shader compiler.
#
# Requirements: Xcode Command Line Tools (provides xcrun, metal, metallib)
# Usage: ./build_metallib.sh [--output PATH] [--metal-version VERSION]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHADER_DIR="$ROOT_DIR/Sources/Shaders"
OUTPUT="$SCRIPT_DIR/zkmetal.metallib"
METAL_STD="metal3.0"
BUILD_DIR=""
KEEP_AIR=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)      OUTPUT="$2"; shift 2 ;;
        --metal-version) METAL_STD="$2"; shift 2 ;;
        --keep-air)    KEEP_AIR=1; shift ;;
        --help|-h)
            echo "Usage: $0 [--output PATH] [--metal-version VERSION] [--keep-air]"
            echo ""
            echo "Options:"
            echo "  --output PATH          Output .metallib path (default: bindings/metallib/zkmetal.metallib)"
            echo "  --metal-version VER    Metal Shading Language version (default: metal3.0)"
            echo "                         Use metal2.4 for older macOS, metal3.1 for latest features"
            echo "  --keep-air             Keep intermediate .air files after linking"
            echo ""
            echo "Supported Metal versions by macOS:"
            echo "  metal2.4  — macOS 12 Monterey+"
            echo "  metal3.0  — macOS 13 Ventura+  (default)"
            echo "  metal3.1  — macOS 14 Sonoma+"
            echo "  metal3.2  — macOS 15 Sequoia+"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Verify tools
# ---------------------------------------------------------------------------
if ! xcrun --find metal &>/dev/null; then
    echo "Error: Metal compiler not found. Install Xcode Command Line Tools:"
    echo "  xcode-select --install"
    exit 1
fi

METAL_CC="xcrun -sdk macosx metal"
METAL_LD="xcrun -sdk macosx metallib"

echo "zkMetal .metallib build"
echo "  Shader dir:    $SHADER_DIR"
echo "  Metal version: -std=$METAL_STD"
echo "  Output:        $OUTPUT"
echo ""

# ---------------------------------------------------------------------------
# Create temp build directory
# ---------------------------------------------------------------------------
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/zkmetal-metallib.XXXXXX")"
trap 'if [ "$KEEP_AIR" -eq 0 ] && [ -n "$BUILD_DIR" ]; then rm -rf "$BUILD_DIR"; fi' EXIT

# ---------------------------------------------------------------------------
# Discover all .metal files that contain kernel functions
# ---------------------------------------------------------------------------
KERNEL_FILES=()
while IFS= read -r f; do
    if grep -q '^kernel ' "$f"; then
        KERNEL_FILES+=("$f")
    fi
done < <(find "$SHADER_DIR" -name '*.metal' -type f | sort)

echo "Found ${#KERNEL_FILES[@]} shader files with kernel functions"

# ---------------------------------------------------------------------------
# Compile each kernel file to .air
# ---------------------------------------------------------------------------
AIR_FILES=()
FAIL=0

compile_shader() {
    local src="$1"
    local rel="${src#$SHADER_DIR/}"
    local air_name
    air_name="$(echo "$rel" | tr '/' '_' | sed 's/\.metal$/.air/')"
    local air_path="$BUILD_DIR/$air_name"

    # The Metal compiler resolves #include paths relative to the source file,
    # so "../fields/bn254_fp.metal" from msm/msm_kernels.metal works naturally.
    # We also add -I for the Shaders root as a fallback.
    if $METAL_CC \
        -std=$METAL_STD \
        -O2 \
        -Wall \
        -Wno-unused-variable \
        -I "$SHADER_DIR" \
        -c "$src" \
        -o "$air_path" 2>&1; then
        echo "  OK  $rel"
    else
        echo "  FAIL $rel"
        return 1
    fi
}

echo ""
echo "Compiling shaders to AIR..."
for src in "${KERNEL_FILES[@]}"; do
    if ! compile_shader "$src"; then
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "Error: Some shaders failed to compile. See errors above."
    exit 1
fi

# Collect all .air files
while IFS= read -r -d '' air; do
    AIR_FILES+=("$air")
done < <(find "$BUILD_DIR" -name '*.air' -print0)

echo ""
echo "Compiled ${#AIR_FILES[@]} AIR files"

# ---------------------------------------------------------------------------
# Link all .air files into a single .metallib
# ---------------------------------------------------------------------------
echo ""
echo "Linking into metallib..."
mkdir -p "$(dirname "$OUTPUT")"

$METAL_LD -o "$OUTPUT" "${AIR_FILES[@]}"

METALLIB_SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null)
echo ""
echo "Success: $OUTPUT ($(( METALLIB_SIZE / 1024 )) KB)"

# ---------------------------------------------------------------------------
# Print available kernels
# ---------------------------------------------------------------------------
echo ""
echo "Available kernel functions:"
# Use xcrun metal-readobj or nm to list symbols if available, otherwise grep
if xcrun --find metal-readobj &>/dev/null 2>&1; then
    xcrun metal-readobj --symbols "$OUTPUT" 2>/dev/null | grep -oE 'Name: [a-zA-Z_][a-zA-Z0-9_]*' | sed 's/Name: /  /' | sort -u || true
else
    # Fallback: list kernel names from source
    for src in "${KERNEL_FILES[@]}"; do
        grep '^kernel ' "$src" | sed 's/kernel [a-z]* /  /' | sed 's/(.*//'
    done | sort -u
fi

if [ "$KEEP_AIR" -eq 1 ]; then
    echo ""
    echo "AIR files preserved in: $BUILD_DIR"
fi
