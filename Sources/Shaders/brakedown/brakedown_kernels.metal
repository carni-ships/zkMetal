// Brakedown polynomial commitment — GPU kernels
// Matrix-vector multiply for linear code encoding and dot products for opening.
// No NTT required — all operations are simple field arithmetic.

#include <metal_stdlib>
using namespace metal;

// Fr type and arithmetic are prepended at compile time from bn254_fr.metal

// Encode one redundancy element: computes one entry of R * message
// where R is the (n-k) x k redundancy portion of the generator matrix.
// Generator matrix G = [I_k | R^T], so codeword = [message | R^T * message].
// Each thread computes one output element: sum_j R[tid][j] * message[j]
// R is generated deterministically from a seed using a simple PRNG.
kernel void brakedown_encode_row(
    device const Fr* message         [[buffer(0)]],   // k elements
    device Fr* redundancy            [[buffer(1)]],   // (n-k) elements output
    constant uint& k                 [[buffer(2)]],   // message length
    constant uint& seed              [[buffer(3)]],   // deterministic seed for R
    uint tid                         [[thread_position_in_grid]]
) {
    // Generate pseudo-random coefficients for row tid of R
    // Using a simple xorshift-based PRNG seeded by (seed, tid, j)
    Fr acc;
    for (int i = 0; i < 8; i++) acc.v[i] = 0;

    for (uint j = 0; j < k; j++) {
        // Deterministic pseudo-random Fr element for R[tid][j]
        // Hash (seed, tid, j) to get a field element
        uint s = seed ^ (tid * 2654435761u) ^ (j * 2246822519u);
        s ^= s >> 16; s *= 0x45d9f3b; s ^= s >> 16; s *= 0x45d9f3b; s ^= s >> 16;

        // Create a small Fr element from the hash (not full 256-bit, but sufficient for random linear code)
        Fr r_elem;
        r_elem.v[0] = s;
        r_elem.v[1] = s ^ 0xDEADBEEF;
        for (int i = 2; i < 8; i++) {
            s ^= s >> 13; s *= 0x5bd1e995; s ^= s >> 15;
            r_elem.v[i] = s & 0x0FFFFFFFu; // Keep under modulus
        }
        r_elem.v[7] &= 0x0FFFFFFFu;

        // acc += r_elem * message[j]
        Fr prod = fr_mul(r_elem, message[j]);
        acc = fr_add(acc, prod);
    }

    redundancy[tid] = acc;
}

// Batch encode: encode multiple rows of the matrix simultaneously.
// matrix is rows x cols, each row is a message to encode.
// output is rows x redundancy_cols, one redundancy block per row.
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
