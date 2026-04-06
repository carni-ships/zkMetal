// Grand product Metal shader — segmented parallel prefix product
//
// Computes multiplicative prefix products (exclusive scan) for BN254 Fr.
// Used by permutation arguments, lookup arguments, and GKR.
//
// Three kernels:
//   1. grand_product_local: per-block sequential prefix product + block total
//   2. grand_product_propagate: multiply each element by its block's prefix
//   3. grand_product_ratio: fused num[i]*inv_den[i] element-wise multiply

#include "../fields/bn254_fr.metal"

// ============================================================================
// Per-block exclusive prefix product (sequential within threadgroup)
//
// output[0] = 1
// output[i] = input[0] * input[1] * ... * input[i-1]
//
// Thread 0 does the sequential scan for the entire tile (field mul is
// not associative-friendly for parallel scan due to order dependence,
// so sequential within a block is the correct approach).
// ============================================================================

kernel void grand_product_local(
    device const Fr* input         [[buffer(0)]],
    device Fr* output              [[buffer(1)]],
    device Fr* block_products      [[buffer(2)]],
    constant uint& count           [[buffer(3)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint tile_size = tg_size;
    uint base = tgid * tile_size;

    if (tid == 0) {
        Fr running = fr_one();
        uint end = min(base + tile_size, count);
        for (uint i = base; i < end; i++) {
            output[i] = running;          // exclusive: write before multiply
            running = fr_mul(running, input[i]);
        }
        block_products[tgid] = running;   // total product of this block
    }
}

// ============================================================================
// Propagate block prefixes: multiply each element by the product of all
// preceding blocks. block_prefix[i] = product of blocks 0..i-1 (exclusive).
// ============================================================================

kernel void grand_product_propagate(
    device Fr* data                [[buffer(0)]],
    device const Fr* block_prefix  [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    if (tgid == 0) return;  // Block 0 needs no adjustment

    uint base = tgid * tg_size;
    uint idx = base + tid;
    if (idx < count) {
        Fr prefix = block_prefix[tgid];
        data[idx] = fr_mul(prefix, data[idx]);
    }
}

// ============================================================================
// Element-wise multiply: out[i] = a[i] * b[i]
// Used to compute ratios = numerators[i] * inv_denominators[i]
// ============================================================================

kernel void grand_product_elem_mul(
    device const Fr* a             [[buffer(0)]],
    device const Fr* b             [[buffer(1)]],
    device Fr* output              [[buffer(2)]],
    constant uint& count           [[buffer(3)]],
    uint gid                       [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = fr_mul(a[gid], b[gid]);
}

// ============================================================================
// Full product reduction: reduce N elements to their product.
// Each threadgroup computes a sequential product of its tile.
// For N > tile_size, the caller reduces the block products recursively.
// ============================================================================

kernel void grand_product_reduce(
    device const Fr* input         [[buffer(0)]],
    device Fr* block_products      [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint tid                       [[thread_index_in_threadgroup]],
    uint tgid                      [[threadgroup_position_in_grid]],
    uint tg_size                   [[threads_per_threadgroup]]
) {
    uint tile_size = tg_size;
    uint base = tgid * tile_size;

    if (tid == 0) {
        Fr running = fr_one();
        uint end = min(base + tile_size, count);
        for (uint i = base; i < end; i++) {
            running = fr_mul(running, input[i]);
        }
        block_products[tgid] = running;
    }
}
