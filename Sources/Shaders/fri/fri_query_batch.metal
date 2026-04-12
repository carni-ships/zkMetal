// Batch FRI query kernel — parallel extraction of query evaluations across all layers
//
// For each (query, layer) pair, extracts:
//   - value at the folded index in that layer
//   - sibling value at foldedIdx ^ 1
//
// Then performs the FRI fold recomputation: folded = (a + b) + c * (a - b) * d_inv
// This allows the verifier to check fold consistency across layers.
//
// Thread layout: Q queries × L layers → one thread per (query, layer) pair
// Dispatch: threads_per_threadgroup = min(Q*L, 256), num_threadgroups = ceil(Q*L / 256)

#include "../fields/bn254_fr.metal"

// Extracted query data for one (query, layer) pair
struct FRIQueryPair {
    Fr value;         // evaluation at folded index
    Fr sibling;        // sibling evaluation (foldedIdx ^ 1)
    Fr folded;         // fold recomputation: (a+b) + c*(a-b)*dinv
    uint foldedIdx;    // index within this layer
};

// Batch extract + fold: for each query at each layer
// Inputs:
//   friLayersFlat: flattened [L][maxLayerSize] evaluations
//   queryIndices: [Q] original query indices
//   challenges: [L] folding challenges per layer
//   domainInvsFlat: flattened [L][maxLayerSize] domain inverses per layer
//   layerSizes: [L] size of each layer
//   layerOffsets: [L] offset into friLayersFlat for each layer
//   invOffsets: [L] offset into domainInvsFlat for each layer
//
// Outputs (packed):
//   valuesOut: [Q*L] Fr — value at (query, layer)
//   siblingsOut: [Q*L] Fr — sibling value
//   foldedOut: [Q*L] Fr — fold recomputation
//   indicesOut: [Q*L] uint — folded index for each (query, layer)
kernel void fri_batch_query_bn254(
    device const Fr* friLayersFlat   [[buffer(0)]],
    device const uint* queryIndices [[buffer(1)]],
    device const Fr* challenges     [[buffer(2)]],
    device const Fr* domainInvsFlat  [[buffer(3)]],
    device const uint* layerSizes    [[buffer(4)]],
    device const uint* layerOffsets  [[buffer(5)]],
    device const uint* invOffsets   [[buffer(6)]],
    device Fr* valuesOut            [[buffer(7)]],
    device Fr* siblingsOut          [[buffer(8)]],
    device Fr* foldedOut            [[buffer(9)]],
    device uint* indicesOut         [[buffer(10)]],
    constant uint& numQueries       [[buffer(11)]],
    constant uint& numLayers       [[buffer(12)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= numQueries * numLayers) return;

    uint q = gid / numLayers;
    uint l = gid % numLayers;

    uint queryIdx = queryIndices[q];
    uint layerSize = layerSizes[l];
    uint layerOff = layerOffsets[l];
    uint invOff = invOffsets[l];

    // Compute fold index through layers:
    // Layer 0: foldedIdx = queryIdx % layerSize, nextIdx = foldedIdx >> 1
    // Layer 1: foldedIdx = nextIdx % layerSizes[1], nextIdx = foldedIdx >> 1
    // etc.
    uint currentIdx = queryIdx;
    for (uint prev = 0; prev < l; prev++) {
        uint prevSize = layerSizes[prev];
        uint fi = currentIdx % prevSize;
        currentIdx = fi >> 1;
    }
    uint foldedIdx = currentIdx % layerSize;
    uint siblingIdx = foldedIdx ^ 1;

    // Clamp sibling to layer size (for queries at odd positions near boundary)
    if (siblingIdx >= layerSize) siblingIdx = foldedIdx;

    // Load values
    uint valOffset = layerOff + foldedIdx;
    uint sibOffset = layerOff + siblingIdx;

    Fr a = friLayersFlat[valOffset];
    Fr b = friLayersFlat[sibOffset];

    // FRI fold recomputation: (a + b) + c * (a - b) * d_inv
    Fr c = challenges[l];
    Fr d_inv = domainInvsFlat[invOff + foldedIdx];

    Fr sum = fr_add(a, b);
    Fr diff = fr_sub(a, b);
    Fr term = fr_mul(c, fr_mul(diff, d_inv));
    Fr folded = fr_add(sum, term);

    // Store outputs
    uint outIdx = q * numLayers + l;
    valuesOut[outIdx] = a;
    siblingsOut[outIdx] = b;
    foldedOut[outIdx] = folded;
    indicesOut[outIdx] = foldedIdx;
}

// Merkle path generation kernel for FRI layers (GPU batch)
//
// For each (query, layer) pair, walks from leaf to root computing the full
// authentication path. Uses the same path-extraction logic as extractMerklePath
// but processes all queries in parallel.
//
// Inputs:
//   friMerkleLeaves: [L][layerSize] — leaf hashes per layer
//   friMerkleNodes: [L][2*layerSize] — internal nodes per layer (flat)
//   layerSizes: [L] — size of each layer's Merkle tree
//   layerLeafOffsets: [L] — offset into friMerkleLeaves for each layer
//   layerNodeOffsets: [L] — offset into friMerkleNodes for each layer
//   indicesOut: [Q*L] uint — folded index per (query, layer) from fri_batch_query_bn254
//
// Outputs:
//   pathsOut: [Q*L*maxDepth] Fr — packed Merkle sibling hashes
//   maxDepth = ceil(log2(maxLayerSize))
//   Each path[k] for query q, layer l: siblings at each level, bottom-up
kernel void fri_merkle_paths_bn254(
    device const Fr* friMerkleLeaves [[buffer(0)]],
    device const Fr* friMerkleNodes  [[buffer(1)]],
    device const uint* layerSizes    [[buffer(2)]],
    device const uint* layerLeafOff  [[buffer(3)]],
    device const uint* layerNodeOff  [[buffer(4)]],
    device const uint* indicesIn     [[buffer(5)]],  // [Q*L] folded index per (query, layer)
    device Fr* pathsOut              [[buffer(6)]],  // [Q*L*maxDepth]
    constant uint& numQueries        [[buffer(7)]],
    constant uint& numLayers        [[buffer(8)]],
    constant uint& maxDepth         [[buffer(9)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= numQueries * numLayers) return;

    uint q = gid / numLayers;
    uint l = gid % numLayers;

    uint leafIdx = indicesIn[gid];
    uint layerSize = layerSizes[l];
    uint leafOff = layerLeafOff[l];
    uint nodeOff = layerNodeOff[l];

    uint depth = 0;
    uint curIdx = leafIdx;

    // Compute depth from layer size
    uint sz = layerSize;
    while (sz > 1) {
        sz >>= 1;
        depth++;
    }
    if (depth > maxDepth) depth = maxDepth;

    uint pathStart = gid * maxDepth;

    for (uint d = 0; d < maxDepth; d++) {
        uint sibIdx = curIdx ^ 1;

        Fr sibling;
        if (d == 0) {
            // Level 0: sibling comes from leaves array
            sibling = friMerkleLeaves[leafOff + sibIdx];
        } else {
            // Level d>0: sibling from internal nodes
            // Node offset: at level d, we need to skip d levels of bottom-up internal nodes
            uint nodesPerLevel = layerSize;
            uint nodeBase = 0;
            for (uint prev = 0; prev < d; prev++) {
                nodeBase += nodesPerLevel;
                nodesPerLevel >>= 1;
            }
            sibling = friMerkleNodes[nodeOff + nodeBase + sibIdx];
        }

        pathsOut[pathStart + d] = sibling;
        curIdx >>= 1;
    }
}