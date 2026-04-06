// GPU-accelerated matrix transpose for field elements
//
// Tiled transpose using threadgroup shared memory for coalesced reads/writes.
// Supports arbitrary field element sizes (BN254 Fr = 32 bytes, BabyBear = 4 bytes, etc.)
// by operating on uint4 chunks (16 bytes each). A 32-byte Fr = 2 chunks per element.
//
// Tile size: 16x16 elements. Each threadgroup handles one tile.
// Padding (+1) in shared memory eliminates bank conflicts on the transpose.
//
// Kernels:
//   matrix_transpose_fr      — out-of-place transpose for BN254 Fr (8x uint limbs)
//   matrix_transpose_u32     — out-of-place transpose for 32-bit fields (BabyBear, Mersenne31)
//   matrix_transpose_u64     — out-of-place transpose for 64-bit fields (Goldilocks)
//   matrix_transpose_fr_inplace — in-place square matrix transpose for BN254 Fr

#include <metal_stdlib>
using namespace metal;

// --- BN254 Fr transpose (32 bytes per element, 8x uint limbs) ---

#define TILE_DIM 16
#define BANK_PAD 1

// Out-of-place: dst[j * rows + i] = src[i * cols + j]
// Params buffer: [rows, cols] as uint2
kernel void matrix_transpose_fr(
    device const uint* src [[buffer(0)]],
    device uint* dst       [[buffer(1)]],
    constant uint2& dims   [[buffer(2)]],    // .x = rows, .y = cols
    uint2 gid              [[thread_position_in_grid]],
    uint2 tid              [[thread_position_in_threadgroup]],
    uint2 tgid             [[threadgroup_position_in_grid]]
) {
    const uint rows = dims.x;
    const uint cols = dims.y;
    const uint LIMBS = 8; // Fr = 8x uint32

    // Shared memory tile with bank-conflict padding
    threadgroup uint tile[TILE_DIM][TILE_DIM + BANK_PAD][8];

    // Source coordinates for this tile
    uint srcRow = tgid.y * TILE_DIM + tid.y;
    uint srcCol = tgid.x * TILE_DIM + tid.x;

    // Load: read src[srcRow][srcCol] into shared memory tile[tid.y][tid.x]
    if (srcRow < rows && srcCol < cols) {
        uint srcIdx = (srcRow * cols + srcCol) * LIMBS;
        for (uint l = 0; l < LIMBS; l++) {
            tile[tid.y][tid.x][l] = src[srcIdx + l];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Destination coordinates: transposed tile position
    uint dstRow = tgid.x * TILE_DIM + tid.y;
    uint dstCol = tgid.y * TILE_DIM + tid.x;

    // Store: write tile[tid.x][tid.y] (transposed read) to dst[dstRow][dstCol]
    if (dstRow < cols && dstCol < rows) {
        uint dstIdx = (dstRow * rows + dstCol) * LIMBS;
        for (uint l = 0; l < LIMBS; l++) {
            dst[dstIdx + l] = tile[tid.x][tid.y][l];
        }
    }
}

// In-place square matrix transpose for BN254 Fr
// Only swaps elements where (i < j) to avoid double-swapping.
// Params buffer: n (matrix dimension, must be square n x n)
kernel void matrix_transpose_fr_inplace(
    device uint* data      [[buffer(0)]],
    constant uint& n       [[buffer(1)]],
    uint2 gid              [[thread_position_in_grid]],
    uint2 tid              [[thread_position_in_threadgroup]],
    uint2 tgid             [[threadgroup_position_in_grid]]
) {
    const uint LIMBS = 8;

    threadgroup uint tileA[TILE_DIM][TILE_DIM + BANK_PAD][8];
    threadgroup uint tileB[TILE_DIM][TILE_DIM + BANK_PAD][8];

    uint blockRow = tgid.y * TILE_DIM;
    uint blockCol = tgid.x * TILE_DIM;

    // Only process upper-triangle blocks (blockRow <= blockCol)
    if (blockRow > blockCol) return;

    bool isDiagonal = (blockRow == blockCol);

    // Load tile A: data[blockRow + tid.y][blockCol + tid.x]
    uint rA = blockRow + tid.y;
    uint cA = blockCol + tid.x;
    if (rA < n && cA < n) {
        uint idxA = (rA * n + cA) * LIMBS;
        for (uint l = 0; l < LIMBS; l++) {
            tileA[tid.y][tid.x][l] = data[idxA + l];
        }
    }

    if (!isDiagonal) {
        // Load tile B: data[blockCol + tid.y][blockRow + tid.x] (the mirror tile)
        uint rB = blockCol + tid.y;
        uint cB = blockRow + tid.x;
        if (rB < n && cB < n) {
            uint idxB = (rB * n + cB) * LIMBS;
            for (uint l = 0; l < LIMBS; l++) {
                tileB[tid.y][tid.x][l] = data[idxB + l];
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (isDiagonal) {
        // Diagonal block: write transposed tile A back
        // tile A[tid.y][tid.x] -> data[blockRow + tid.x][blockCol + tid.y]
        uint rOut = blockRow + tid.x;
        uint cOut = blockCol + tid.y;
        if (rOut < n && cOut < n) {
            uint idxOut = (rOut * n + cOut) * LIMBS;
            for (uint l = 0; l < LIMBS; l++) {
                data[idxOut + l] = tileA[tid.y][tid.x][l];
            }
        }
    } else {
        // Off-diagonal: swap tile A and tile B (transposed)
        // Write transposed tile A -> position of tile B
        uint rB = blockCol + tid.x;
        uint cB = blockRow + tid.y;
        if (rB < n && cB < n) {
            uint idxB = (rB * n + cB) * LIMBS;
            for (uint l = 0; l < LIMBS; l++) {
                data[idxB + l] = tileA[tid.y][tid.x][l];
            }
        }

        // Write transposed tile B -> position of tile A
        uint rA2 = blockRow + tid.x;
        uint cA2 = blockCol + tid.y;
        if (rA2 < n && cA2 < n) {
            uint idxA = (rA2 * n + cA2) * LIMBS;
            for (uint l = 0; l < LIMBS; l++) {
                data[idxA + l] = tileB[tid.y][tid.x][l];
            }
        }
    }
}

// --- 32-bit field transpose (BabyBear, Mersenne31) ---

kernel void matrix_transpose_u32(
    device const uint* src [[buffer(0)]],
    device uint* dst       [[buffer(1)]],
    constant uint2& dims   [[buffer(2)]],
    uint2 gid              [[thread_position_in_grid]],
    uint2 tid              [[thread_position_in_threadgroup]],
    uint2 tgid             [[threadgroup_position_in_grid]]
) {
    const uint rows = dims.x;
    const uint cols = dims.y;

    threadgroup uint tile[TILE_DIM][TILE_DIM + BANK_PAD];

    uint srcRow = tgid.y * TILE_DIM + tid.y;
    uint srcCol = tgid.x * TILE_DIM + tid.x;

    if (srcRow < rows && srcCol < cols) {
        tile[tid.y][tid.x] = src[srcRow * cols + srcCol];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint dstRow = tgid.x * TILE_DIM + tid.y;
    uint dstCol = tgid.y * TILE_DIM + tid.x;

    if (dstRow < cols && dstCol < rows) {
        dst[dstRow * rows + dstCol] = tile[tid.x][tid.y];
    }
}

// --- 64-bit field transpose (Goldilocks) ---

kernel void matrix_transpose_u64(
    device const ulong* src [[buffer(0)]],
    device ulong* dst       [[buffer(1)]],
    constant uint2& dims    [[buffer(2)]],
    uint2 gid               [[thread_position_in_grid]],
    uint2 tid               [[thread_position_in_threadgroup]],
    uint2 tgid              [[threadgroup_position_in_grid]]
) {
    const uint rows = dims.x;
    const uint cols = dims.y;

    threadgroup ulong tile[TILE_DIM][TILE_DIM + BANK_PAD];

    uint srcRow = tgid.y * TILE_DIM + tid.y;
    uint srcCol = tgid.x * TILE_DIM + tid.x;

    if (srcRow < rows && srcCol < cols) {
        tile[tid.y][tid.x] = src[srcRow * cols + srcCol];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint dstRow = tgid.x * TILE_DIM + tid.y;
    uint dstCol = tgid.y * TILE_DIM + tid.x;

    if (dstRow < cols && dstCol < rows) {
        dst[dstRow * rows + dstCol] = tile[tid.x][tid.y];
    }
}
