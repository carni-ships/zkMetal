#!/bin/bash
#
# build.sh -- Build libzkmetal.a and libzkmetal.dylib from Sources/NeonFieldOps
#
# Usage:
#   ./build.sh                    # build in ./build/
#   ./build.sh install            # build + install to PREFIX (default /usr/local)
#   PREFIX=/opt/zkmetal ./build.sh install
#
# Requirements: ARM64/Apple Silicon, Xcode command-line tools

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$REPO_ROOT/Sources/NeonFieldOps"
BUILD_DIR="$SCRIPT_DIR/build"
PREFIX="${PREFIX:-/usr/local}"

CC="${CC:-cc}"
AR="${AR:-ar}"

CFLAGS="-O3 -march=armv8.2-a+crypto -std=c11 -fPIC -Wall -Wextra -Wno-unused-parameter"
CFLAGS="$CFLAGS -I$SRC_DIR/include"

# Verify ARM64
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ] && [ "$ARCH" != "aarch64" ]; then
    echo "Error: zkMetal requires ARM64/Apple Silicon (detected: $ARCH)" >&2
    exit 1
fi

echo "=== zkMetal build ==="
echo "  Sources:  $SRC_DIR"
echo "  Build:    $BUILD_DIR"
echo "  CC:       $CC"
echo "  CFLAGS:   $CFLAGS"
echo ""

mkdir -p "$BUILD_DIR"

# Collect all .c source files
C_SOURCES=("$SRC_DIR"/*.c)

# Also include assembly
ASM_SOURCES=()
if ls "$SRC_DIR"/*.s 1>/dev/null 2>&1; then
    ASM_SOURCES=("$SRC_DIR"/*.s)
fi

OBJ_FILES=()

# Compile C sources
for src in "${C_SOURCES[@]}"; do
    obj="$BUILD_DIR/$(basename "$src" .c).o"
    echo "  CC  $(basename "$src")"
    $CC $CFLAGS -c "$src" -o "$obj"
    OBJ_FILES+=("$obj")
done

# Compile assembly sources
for src in "${ASM_SOURCES[@]}"; do
    obj="$BUILD_DIR/$(basename "$src" .s).o"
    echo "  AS  $(basename "$src")"
    $CC -c "$src" -o "$obj"
    OBJ_FILES+=("$obj")
done

# Static library
echo ""
echo "  AR  libzkmetal.a"
$AR rcs "$BUILD_DIR/libzkmetal.a" "${OBJ_FILES[@]}"

# Dynamic library (macOS dylib)
echo "  LD  libzkmetal.dylib"
$CC -dynamiclib -o "$BUILD_DIR/libzkmetal.dylib" \
    "${OBJ_FILES[@]}" \
    -install_name "@rpath/libzkmetal.dylib" \
    -current_version 0.1.0 \
    -compatibility_version 0.1.0

# Generate pkg-config file
echo "  GEN zkmetal.pc"
sed -e "s|@PREFIX@|$PREFIX|g" \
    -e "s|@VERSION@|0.1.0|g" \
    "$SCRIPT_DIR/zkmetal.pc.in" > "$BUILD_DIR/zkmetal.pc"

echo ""
echo "Build complete:"
echo "  $BUILD_DIR/libzkmetal.a"
echo "  $BUILD_DIR/libzkmetal.dylib"
echo "  $BUILD_DIR/zkmetal.pc"

# Install target
if [ "${1:-}" = "install" ]; then
    echo ""
    echo "Installing to $PREFIX ..."
    install -d "$PREFIX/lib"
    install -d "$PREFIX/lib/pkgconfig"
    install -d "$PREFIX/include/zkmetal"

    install -m 644 "$BUILD_DIR/libzkmetal.a"     "$PREFIX/lib/"
    install -m 755 "$BUILD_DIR/libzkmetal.dylib"  "$PREFIX/lib/"
    install -m 644 "$BUILD_DIR/zkmetal.pc"        "$PREFIX/lib/pkgconfig/"
    install -m 644 "$SCRIPT_DIR/include/zkmetal.h" "$PREFIX/include/zkmetal/"

    echo "Installed."
fi
