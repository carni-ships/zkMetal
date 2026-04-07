// Lattice polynomial operations — batch add, sub, pointwise multiply, matrix-vector
// These operate on polynomials already in NTT domain for Kyber/Dilithium module-lattice ops.

#include <metal_stdlib>
using namespace metal;

// Field constants (duplicated here to avoid include issues with Metal compilation)
constant ushort KYBER_Q_OPS = 3329;
constant uint KYBER_BARRETT_M_OPS = 5039;
constant uint KYBER_BARRETT_SHIFT_OPS = 24;

constant uint DIL_Q_OPS = 8380417;
constant ulong DIL_BARRETT_M_OPS = 33579385UL;
constant ulong DIL_BARRETT_SHIFT_OPS = 48;

inline ushort kyber_reduce_ops(uint a) {
    return ushort(a % uint(KYBER_Q_OPS));
}

inline ushort kyber_add_ops(ushort a, ushort b) {
    uint s = uint(a) + uint(b);
    return s >= uint(KYBER_Q_OPS) ? ushort(s - uint(KYBER_Q_OPS)) : ushort(s);
}

inline ushort kyber_sub_ops(ushort a, ushort b) {
    return a >= b ? (a - b) : (a + KYBER_Q_OPS - b);
}

inline ushort kyber_mul_ops(ushort a, ushort b) {
    return kyber_reduce_ops(uint(a) * uint(b));
}

inline uint dil_reduce_ops(ulong a) {
    return uint(a % ulong(DIL_Q_OPS));
}

inline uint dil_add_ops(uint a, uint b) {
    ulong s = ulong(a) + ulong(b);
    return s >= ulong(DIL_Q_OPS) ? uint(s - ulong(DIL_Q_OPS)) : uint(s);
}

inline uint dil_sub_ops(uint a, uint b) {
    return a >= b ? (a - b) : (a + DIL_Q_OPS - b);
}

inline uint dil_mul_ops(uint a, uint b) {
    return dil_reduce_ops(ulong(a) * ulong(b));
}

// ============================================================
// Kyber batch polynomial operations
// ============================================================

// Batch element-wise addition: out[i] = a[i] + b[i] mod q
kernel void kyber_poly_add(
    device const ushort* a [[buffer(0)]],
    device const ushort* b [[buffer(1)]],
    device ushort* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],  // total elements (num_polys * 256)
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    out[gid] = kyber_add_ops(a[gid], b[gid]);
}

// Batch element-wise subtraction
kernel void kyber_poly_sub(
    device const ushort* a [[buffer(0)]],
    device const ushort* b [[buffer(1)]],
    device ushort* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    out[gid] = kyber_sub_ops(a[gid], b[gid]);
}

// Batch basemul in NTT domain: pairs at (4i, 4i+1) use +gamma, (4i+2, 4i+3) use -gamma
// gid indexes pairs (count/2 total). twiddles buffer has 128 entries; gammas are twiddles[64..127].
kernel void kyber_poly_pointwise_mul(
    device const ushort* a [[buffer(0)]],
    device const ushort* b [[buffer(1)]],
    device ushort* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant ushort* twiddles [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total_pairs = count / 2;
    if (gid >= total_pairs) return;

    uint pair_in_poly = gid % 128;
    uint group = pair_in_poly / 2;
    bool is_second = (pair_in_poly & 1) != 0;
    uint base = gid * 2;

    ushort a0 = a[base], a1 = a[base + 1];
    ushort b0 = b[base], b1 = b[base + 1];
    ushort fwd_gamma = twiddles[64 + group];
    ushort gamma = is_second ? (KYBER_Q_OPS - fwd_gamma) : fwd_gamma;

    out[base]     = kyber_add_ops(kyber_mul_ops(a0, b0), kyber_mul_ops(kyber_mul_ops(a1, b1), gamma));
    out[base + 1] = kyber_add_ops(kyber_mul_ops(a0, b1), kyber_mul_ops(a1, b0));
}

// Matrix-vector multiply for Kyber (k x k matrix of NTT polys, k-vector of NTT polys)
// A is k*k*256 elements (row-major), s is k*256 elements, out is k*256 elements
// out[i] = sum_j A[i][j] * s[j]  (in NTT domain, so pointwise multiply + add)
kernel void kyber_matvec_ntt(
    device const ushort* A [[buffer(0)]],     // k*k*256 matrix
    device const ushort* s [[buffer(1)]],     // k*256 vector
    device ushort* out [[buffer(2)]],         // k*256 result
    constant uint& k [[buffer(3)]],           // dimension (3 for Kyber-768)
    uint gid [[thread_position_in_grid]]       // ranges over k*256
) {
    uint total = k * 256;
    if (gid >= total) return;

    uint row = gid / 256;  // which output polynomial
    uint coeff = gid % 256;  // which coefficient

    ushort acc = 0;
    for (uint col = 0; col < k; col++) {
        ushort a_val = A[(row * k + col) * 256 + coeff];
        ushort s_val = s[col * 256 + coeff];
        acc = kyber_add_ops(acc, kyber_mul_ops(a_val, s_val));
    }
    out[gid] = acc;
}

// ============================================================
// Dilithium batch polynomial operations
// ============================================================

kernel void dilithium_poly_add(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    out[gid] = dil_add_ops(a[gid], b[gid]);
}

kernel void dilithium_poly_sub(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    out[gid] = dil_sub_ops(a[gid], b[gid]);
}

kernel void dilithium_poly_pointwise_mul(
    device const uint* a [[buffer(0)]],
    device const uint* b [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint* twiddles [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total_pairs = count / 2;
    if (gid >= total_pairs) return;

    uint pair_in_poly = gid % 128;
    uint group = pair_in_poly / 2;
    bool is_second = (pair_in_poly & 1) != 0;
    uint base = gid * 2;

    uint a0 = a[base], a1 = a[base + 1];
    uint b0 = b[base], b1 = b[base + 1];
    uint fwd_gamma = twiddles[64 + group];
    uint gamma = is_second ? (DIL_Q_OPS - fwd_gamma) : fwd_gamma;

    out[base]     = dil_add_ops(dil_mul_ops(a0, b0), dil_mul_ops(dil_mul_ops(a1, b1), gamma));
    out[base + 1] = dil_add_ops(dil_mul_ops(a0, b1), dil_mul_ops(a1, b0));
}

// Matrix-vector multiply for Dilithium
kernel void dilithium_matvec_ntt(
    device const uint* A [[buffer(0)]],
    device const uint* s [[buffer(1)]],
    device uint* out [[buffer(2)]],
    constant uint& k [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = k * 256;
    if (gid >= total) return;

    uint row = gid / 256;
    uint coeff = gid % 256;

    uint acc = 0;
    for (uint col = 0; col < k; col++) {
        uint a_val = A[(row * k + col) * 256 + coeff];
        uint s_val = s[col * 256 + coeff];
        acc = dil_add_ops(acc, dil_mul_ops(a_val, s_val));
    }
    out[gid] = acc;
}

// Compress/decompress kernels for Kyber ciphertext encoding
kernel void kyber_compress(
    device const ushort* input [[buffer(0)]],
    device ushort* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& d [[buffer(3)]],  // compression bits
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    // Compress: round(2^d / q * x) mod 2^d
    uint x = uint(input[gid]);
    uint shifted = (x << d) + (KYBER_Q_OPS / 2);
    uint compressed = shifted / uint(KYBER_Q_OPS);
    output[gid] = ushort(compressed & ((1u << d) - 1u));
}

kernel void kyber_decompress(
    device const ushort* input [[buffer(0)]],
    device ushort* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& d [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    // Decompress: round(q / 2^d * x)
    uint x = uint(input[gid]);
    uint decompressed = (x * uint(KYBER_Q_OPS) + (1u << (d - 1))) >> d;
    output[gid] = ushort(decompressed);
}
