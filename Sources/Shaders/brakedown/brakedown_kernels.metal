// Brakedown polynomial commitment — GPU kernels
// Expander-code-based sparse encoding + dot products for opening.
// No NTT required — all operations are simple field arithmetic.
//
// The key kernel is brakedown_sparse_encode: each thread computes one
// redundancy element using exactly D multiply-accumulates, where D is
// the expander graph degree (typically 10). This is D× faster than
// the dense O(k) per element approach.

#include <metal_stdlib>
using namespace metal;

// Fr type and arithmetic are prepended at compile time from bn254_fr.metal

// MARK: - Sparse Expander Encode

// Sparse encode: compute one redundancy element using the expander graph.
// Each redundancy element is a weighted sum of exactly `degree` message elements,
// where the indices and coefficients come from the expander graph structure.
//
// neighbors[red_col * degree + d] = index of d-th left neighbor for redundancy column red_col
// coefficients[red_col * degree + d] = coefficient for that edge
//
// This is O(degree) per element instead of O(cols) for dense encoding.
kernel void brakedown_sparse_encode(
    device const Fr* matrix          [[buffer(0)]],   // rows x cols input
    device Fr* output                [[buffer(1)]],   // rows x redundancy_cols output
    device const uint* neighbors     [[buffer(2)]],   // redundancy_cols * degree neighbor indices
    device const Fr* coefficients    [[buffer(3)]],   // redundancy_cols * degree coefficients
    constant uint& cols              [[buffer(4)]],   // message length per row
    constant uint& redundancy_cols   [[buffer(5)]],   // redundancy length per row
    constant uint& degree            [[buffer(6)]],   // expander degree
    uint2 tid                        [[thread_position_in_grid]]
    // tid.x = redundancy column index, tid.y = row index
) {
    uint row = tid.y;
    uint red_col = tid.x;
    if (red_col >= redundancy_cols) return;

    device const Fr* msg = matrix + row * cols;

    Fr acc;
    for (int i = 0; i < 8; i++) acc.v[i] = 0;

    uint base = red_col * degree;
    for (uint d = 0; d < degree; d++) {
        uint left_idx = neighbors[base + d];
        Fr coeff = coefficients[base + d];
        Fr val = msg[left_idx];
        Fr prod = fr_mul(coeff, val);
        acc = fr_add(acc, prod);
    }

    output[row * redundancy_cols + red_col] = acc;
}

// MARK: - Dense Encode (legacy, kept for comparison)

// Batch encode using dense random matrix: each thread computes one redundancy element
// by iterating over ALL message elements. O(cols) per element.
kernel void brakedown_batch_encode(
    device const Fr* matrix          [[buffer(0)]],   // rows x cols
    device Fr* output                [[buffer(1)]],   // rows x redundancy_cols
    constant uint& cols              [[buffer(2)]],   // message length per row
    constant uint& redundancy_cols   [[buffer(3)]],   // redundancy length per row
    constant uint& seed              [[buffer(4)]],
    uint2 tid                        [[thread_position_in_grid]]
    // tid.x = redundancy column index, tid.y = row index
) {
    uint row = tid.y;
    uint red_col = tid.x;
    if (red_col >= redundancy_cols) return;

    device const Fr* msg = matrix + row * cols;

    Fr acc;
    for (int i = 0; i < 8; i++) acc.v[i] = 0;

    for (uint j = 0; j < cols; j++) {
        uint s = seed ^ (red_col * 2654435761u) ^ (j * 2246822519u);
        s ^= s >> 16; s *= 0x45d9f3b; s ^= s >> 16; s *= 0x45d9f3b; s ^= s >> 16;

        Fr r_elem;
        r_elem.v[0] = s;
        r_elem.v[1] = s ^ 0xDEADBEEF;
        for (int i = 2; i < 8; i++) {
            s ^= s >> 13; s *= 0x5bd1e995; s ^= s >> 15;
            r_elem.v[i] = s & 0x0FFFFFFFu;
        }
        r_elem.v[7] &= 0x0FFFFFFFu;

        Fr prod = fr_mul(r_elem, msg[j]);
        acc = fr_add(acc, prod);
    }

    output[row * redundancy_cols + red_col] = acc;
}

// MARK: - Dot Product

// Dot product: compute inner product of a vector with each row of a matrix.
// matrix is rows x cols, tensor is a vector of length cols.
// results[i] = sum_j matrix[i][j] * tensor[j]
kernel void brakedown_dot_product(
    device const Fr* matrix          [[buffer(0)]],   // rows x cols
    device const Fr* tensor          [[buffer(1)]],   // cols elements
    device Fr* results               [[buffer(2)]],   // rows elements
    constant uint& num_cols          [[buffer(3)]],
    uint tid                         [[thread_position_in_grid]]
) {
    device const Fr* row = matrix + tid * num_cols;

    Fr acc;
    for (int i = 0; i < 8; i++) acc.v[i] = 0;

    for (uint j = 0; j < num_cols; j++) {
        Fr prod = fr_mul(row[j], tensor[j]);
        acc = fr_add(acc, prod);
    }

    results[tid] = acc;
}

// MARK: - Column Extraction

// Column extraction: extract column col_idx from a rows x cols matrix.
// Useful for opening proof (extracting specific columns for Merkle verification).
kernel void brakedown_extract_column(
    device const Fr* matrix          [[buffer(0)]],   // rows x total_cols
    device Fr* column                [[buffer(1)]],   // rows elements output
    constant uint& total_cols        [[buffer(2)]],
    constant uint& col_idx           [[buffer(3)]],
    uint tid                         [[thread_position_in_grid]]
) {
    column[tid] = matrix[tid * total_cols + col_idx];
}
