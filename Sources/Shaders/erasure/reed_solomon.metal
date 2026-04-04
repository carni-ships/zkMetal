// Reed-Solomon erasure coding Metal shaders
// GF(2^16) field operations via log/antilog table lookup.
// Supports batch multiply and systematic RS encoding (matrix-vector).

#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------
// GF(2^16) batch multiply via log/antilog tables
// -----------------------------------------------------------------------
kernel void gf16_batch_mul(
    device const ushort* a [[buffer(0)]],
    device const ushort* b [[buffer(1)]],
    device ushort* out [[buffer(2)]],
    constant ushort* log_table [[buffer(3)]],
    constant ushort* antilog_table [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    ushort va = a[tid], vb = b[tid];
    if (va == 0 || vb == 0) { out[tid] = 0; return; }
    uint sum = (uint)log_table[va] + (uint)log_table[vb];
    if (sum >= 65535u) sum -= 65535u;
    out[tid] = antilog_table[sum];
}

// -----------------------------------------------------------------------
// GF(2^16) batch add (XOR)
// -----------------------------------------------------------------------
kernel void gf16_batch_add(
    device const ushort* a [[buffer(0)]],
    device const ushort* b [[buffer(1)]],
    device ushort* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    out[tid] = a[tid] ^ b[tid];
}

// -----------------------------------------------------------------------
// RS encode systematic: compute parity elements via matrix-vector product
// generator is (n-k) x k matrix stored row-major
// One thread per parity element: parity[tid] = sum_j generator[tid*k + j] * data[j]
// -----------------------------------------------------------------------
kernel void rs_encode_systematic(
    device const ushort* data [[buffer(0)]],
    device ushort* parity [[buffer(1)]],
    constant ushort* generator [[buffer(2)]],
    constant ushort* log_table [[buffer(3)]],
    constant ushort* antilog_table [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    ushort acc = 0;
    uint row_offset = tid * k;
    for (uint j = 0; j < k; j++) {
        ushort g = generator[row_offset + j];
        ushort d = data[j];
        if (g != 0 && d != 0) {
            uint s = (uint)log_table[g] + (uint)log_table[d];
            if (s >= 65535u) s -= 65535u;
            acc ^= antilog_table[s];  // XOR = add in GF(2^k)
        }
    }
    parity[tid] = acc;
}

// -----------------------------------------------------------------------
// RS decode: matrix-vector product for reconstruction
// invMatrix is k x k matrix (inverse of sub-Vandermonde)
// One thread per output element: out[tid] = sum_j invMatrix[tid*k + j] * shards[j]
// -----------------------------------------------------------------------
kernel void rs_decode_matrix(
    device const ushort* shards [[buffer(0)]],
    device ushort* out [[buffer(1)]],
    constant ushort* inv_matrix [[buffer(2)]],
    constant ushort* log_table [[buffer(3)]],
    constant ushort* antilog_table [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    ushort acc = 0;
    uint row_offset = tid * k;
    for (uint j = 0; j < k; j++) {
        ushort m = inv_matrix[row_offset + j];
        ushort s = shards[j];
        if (m != 0 && s != 0) {
            uint sum = (uint)log_table[m] + (uint)log_table[s];
            if (sum >= 65535u) sum -= 65535u;
            acc ^= antilog_table[sum];
        }
    }
    out[tid] = acc;
}

// -----------------------------------------------------------------------
// GF(2^16) polynomial evaluate at multiple points (Horner's method)
// Each thread evaluates poly at one point: out[tid] = poly(points[tid])
// coeffs[0] + coeffs[1]*x + ... + coeffs[deg]*x^deg
// -----------------------------------------------------------------------
kernel void gf16_poly_eval(
    device const ushort* coeffs [[buffer(0)]],
    device const ushort* points [[buffer(1)]],
    device ushort* out [[buffer(2)]],
    constant ushort* log_table [[buffer(3)]],
    constant ushort* antilog_table [[buffer(4)]],
    constant uint& deg_plus_1 [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    ushort x = points[tid];
    if (x == 0) {
        // Just the constant term
        out[tid] = coeffs[0];
        return;
    }

    uint log_x = (uint)log_table[x];
    ushort acc = coeffs[deg_plus_1 - 1];

    for (int i = (int)deg_plus_1 - 2; i >= 0; i--) {
        // acc = acc * x + coeffs[i]
        if (acc != 0) {
            uint s = (uint)log_table[acc] + log_x;
            if (s >= 65535u) s -= 65535u;
            acc = antilog_table[s];
        }
        acc ^= coeffs[i];
    }
    out[tid] = acc;
}
