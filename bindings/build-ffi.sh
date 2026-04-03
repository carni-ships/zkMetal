#!/bin/bash
# Build zkMetal FFI library for Rust consumption
# Produces: bindings/lib/libzkmetal_ffi.a and copies the C header

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$SCRIPT_DIR/lib"

echo "Building zkMetal-ffi..."
cd "$ROOT_DIR"

# Build in release mode
swift build -c release --product zkMetal-ffi 2>&1

# Find the build artifacts directory
BUILD_DIR="$ROOT_DIR/.build/arm64-apple-macosx/release"

# Create output directory
mkdir -p "$OUT_DIR"

# Create a static library from the object files
# Collect all .o files from the ffi target and its dependencies
echo "Packaging static library..."

OBJ_FILES=()
for target in zkMetal_ffi.build zkMetal.build NeonFieldOps.build; do
    TARGET_DIR="$BUILD_DIR/$target"
    if [ -d "$TARGET_DIR" ]; then
        while IFS= read -r -d '' f; do
            OBJ_FILES+=("$f")
        done < <(find "$TARGET_DIR" -name "*.o" -print0)
    fi
done

if [ ${#OBJ_FILES[@]} -eq 0 ]; then
    echo "Error: No object files found in $BUILD_DIR"
    exit 1
fi

ar rcs "$OUT_DIR/libzkmetal_ffi.a" "${OBJ_FILES[@]}"
echo "Created: $OUT_DIR/libzkmetal_ffi.a"

# Copy header
cp "$ROOT_DIR/Sources/zkMetal-ffi/include/zkmetal.h" "$OUT_DIR/zkmetal.h"
echo "Copied: $OUT_DIR/zkmetal.h"

echo ""
echo "To link from Rust, use:"
echo "  cargo:rustc-link-search=native=$OUT_DIR"
echo "  cargo:rustc-link-lib=static=zkmetal_ffi"
echo "  cargo:rustc-link-lib=framework=Metal"
echo "  cargo:rustc-link-lib=framework=Foundation"
echo "  cargo:rustc-link-lib=framework=CoreGraphics"
echo ""
echo "Done!"
