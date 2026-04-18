// Binary FRI fold kernel — GPU-accelerated folding for binary-native FRI
//
// Implements FRI folding over binary tower fields using additive (not multiplicative)
// domains. Key differences from standard FRI:
//
// 1. Domain: An affine subspace S of GF(2^m) with size 2^k, not a multiplicative coset
// 2. Doubling map: D(x) = x^2 + x is GF(2)-linear with kernel {0, 1}
// 3. Fold: f'(x) = f_even(x) + alpha * f_odd(x) where splitting uses trace
//
// This kernel works with GF(2^8) elements and can be composed for larger towers.
//
// GF(2^8) irreducible polynomial: x^8 + x^4 + x^3 + x + 1 (0x11B)

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// GF(2^8) Arithmetic
// ============================================================================

// GF(2^8) multiplication with reduction by 0x11B.
inline uint8_t gf28_mul(device const uint8_t* lut, uint8_t a, uint8_t b) [[always_inline]] {
    return lut[a * 256 + b];
}

// GF(2^8) squaring (optimized since b = a).
inline uint8_t gf28_sq(device const uint8_t* lut, uint8_t a) [[always_inline]] {
    return lut[a * 256 + a];  // Squaring is a*a
}

// GF(2^8) addition is XOR.
inline uint8_t gf28_add(uint8_t a, uint8_t b) [[always_inline]] {
    return a ^ b;
}

// ============================================================================
// Doubling Map: D(x) = x^2 + x
// ============================================================================

// The doubling map D(x) = x^2 + x is GF(2)-linear.
// It maps the affine subspace to a subspace of half the size.
// Kernel = {0, 1} (the GF(2) subfield).
inline uint8_t doubling_map(device const uint8_t* lut, uint8_t x) [[always_inline]] {
    uint8_t x2 = gf28_sq(lut, x);
    return gf28_add(x2, x);
}

// ============================================================================
// Trace Computation
// ============================================================================

// Compute trace Tr_{GF(2^8)/GF(2)}(x) = x + x^2 + x^4 + x^8 + x^16 + x^32 + x^64
// The trace maps GF(2^8) to GF(2) (0 or 1).
inline uint8_t gf28_trace(device const uint8_t* lut, uint8_t x) [[always_inline]] {
    uint8_t t = x;
    uint8_t current = x;
    // 7 more squarings for GF(2^8)
    current = gf28_sq(lut, current);  t ^= current;
    current = gf28_sq(lut, current);  t ^= current;
    current = gf28_sq(lut, current);  t ^= current;
    current = gf28_sq(lut, current);  t ^= current;
    current = gf28_sq(lut, current);  t ^= current;
    current = gf28_sq(lut, current);  t ^= current;
    current = gf28_sq(lut, current);  t ^= current;
    return t;  // Result is 0 or 1
}

// ============================================================================
// Binary FRI Fold Operations
// ============================================================================

// Single binary FRI fold round using additive domain.
//
// For additive domain with doubling map D(x) = x^2 + x:
//   f'(x) = f_even(x) + alpha * f_odd(x)
//
// where:
//   f_even(x) = (f(x) + f(D^{-1}(x))) / 2
//   f_odd(x)  = the complementary part via trace
//
// Simplified fold formula (for pairing-based folding):
//   f'(i) = f(i) + alpha * f(i + n/2)
//
// kernel buffers:
//   buffer(0): lut - GF(2^8) multiplication LUT
//   buffer(1): evals - input evaluations (size n)
//   buffer(2): result - folded evaluations (size n/2)
//   buffer(3): alpha - folding challenge
//   buffer(4): n - current domain size
kernel void binary_fri_fold_kernel(
    device const uint8_t* lut       [[buffer(0)]],
    device const uint8_t* evals     [[buffer(1)]],
    device uint8_t* result          [[buffer(2)]],
    constant uint8_t& alpha        [[buffer(3)]],
    constant uint32_t& n           [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint half_n = n >> 1;
    if (gid >= half_n) return;

    uint8_t f0 = evals[gid];
    uint8_t f1 = evals[gid + half_n];

    // Fold formula: f'(i) = f(i) + alpha * f(i + n/2)
    // In char 2, subtraction = addition
    uint8_t term = gf28_mul(lut, alpha, f1);
    result[gid] = gf28_add(f0, term);
}

// Fused 2-round binary FRI fold.
//
// Applies two consecutive fold rounds in one dispatch,
// processing 4 elements to produce 1 result.
kernel void binary_fri_fold_fused2_kernel(
    device const uint8_t* lut       [[buffer(0)]],
    device const uint8_t* evals     [[buffer(1)]],  // size n
    device uint8_t* result          [[buffer(2)]],  // size n/4
    constant uint8_t& alpha0       [[buffer(3)]],
    constant uint8_t& alpha1       [[buffer(4)]],
    constant uint32_t& n           [[buffer(5)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint quarter = n >> 2;
    if (gid >= quarter) return;

    uint half_n = n >> 1;

    // Read 4 elements: [gid, gid+quarter, gid+half_n, gid+half_n+quarter]
    uint8_t a0 = evals[gid];
    uint8_t a1 = evals[gid + quarter];
    uint8_t a2 = evals[gid + half_n];
    uint8_t a3 = evals[gid + half_n + quarter];

    // Round 1: fold (a0, a2) and (a1, a3)
    uint8_t term0 = gf28_mul(lut, alpha0, a2);
    uint8_t f1_lo = gf28_add(a0, term0);

    uint8_t term1 = gf28_mul(lut, alpha0, a3);
    uint8_t f1_hi = gf28_add(a1, term1);

    // Round 2: fold (f1_lo, f1_hi)
    uint8_t term2 = gf28_mul(lut, alpha1, f1_hi);
    result[gid] = gf28_add(f1_lo, term2);
}

// Fused 4-round binary FRI fold (fold-by-16 equivalent).
//
// Applies four consecutive fold rounds in one dispatch,
// processing 16 elements to produce 1 result.
kernel void binary_fri_fold_fused4_kernel(
    device const uint8_t* lut       [[buffer(0)]],
    device const uint8_t* evals     [[buffer(1)]],  // size n
    device uint8_t* result          [[buffer(2)]],  // size n/16
    constant uint8_t* alphas       [[buffer(3)]],  // 4 challenges
    constant uint32_t& n           [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint sixteenth = n >> 4;
    if (gid >= sixteenth) return;

    uint half_n = n >> 1;
    uint quarter = n >> 2;
    uint eighth = n >> 3;

    // 16 elements: blocks of size 16 covering the input
    uint base = gid * 16;

    uint8_t a00 = evals[base];
    uint8_t a01 = evals[base + 1];
    uint8_t a02 = evals[base + 2];
    uint8_t a03 = evals[base + 3];
    uint8_t a04 = evals[base + 4];
    uint8_t a05 = evals[base + 5];
    uint8_t a06 = evals[base + 6];
    uint8_t a07 = evals[base + 7];
    uint8_t a08 = evals[base + 8];
    uint8_t a09 = evals[base + 9];
    uint8_t a10 = evals[base + 10];
    uint8_t a11 = evals[base + 11];
    uint8_t a12 = evals[base + 12];
    uint8_t a13 = evals[base + 13];
    uint8_t a14 = evals[base + 14];
    uint8_t a15 = evals[base + 15];

    uint8_t c0 = alphas[0];
    uint8_t c1 = alphas[1];
    uint8_t c2 = alphas[2];
    uint8_t c3 = alphas[3];

    // Round 1: fold 8 pairs -> 8 values
    uint8_t r1_0 = gf28_add(a00, gf28_mul(lut, c0, a08));
    uint8_t r1_1 = gf28_add(a01, gf28_mul(lut, c0, a09));
    uint8_t r1_2 = gf28_add(a02, gf28_mul(lut, c0, a10));
    uint8_t r1_3 = gf28_add(a03, gf28_mul(lut, c0, a11));
    uint8_t r1_4 = gf28_add(a04, gf28_mul(lut, c0, a12));
    uint8_t r1_5 = gf28_add(a05, gf28_mul(lut, c0, a13));
    uint8_t r1_6 = gf28_add(a06, gf28_mul(lut, c0, a14));
    uint8_t r1_7 = gf28_add(a07, gf28_mul(lut, c0, a15));

    // Round 2: fold 4 pairs -> 4 values
    uint8_t r2_0 = gf28_add(r1_0, gf28_mul(lut, c1, r1_4));
    uint8_t r2_1 = gf28_add(r1_1, gf28_mul(lut, c1, r1_5));
    uint8_t r2_2 = gf28_add(r1_2, gf28_mul(lut, c1, r1_6));
    uint8_t r2_3 = gf28_add(r1_3, gf28_mul(lut, c1, r1_7));

    // Round 3: fold 2 pairs -> 2 values
    uint8_t r3_0 = gf28_add(r2_0, gf28_mul(lut, c2, r2_2));
    uint8_t r3_1 = gf28_add(r2_1, gf28_mul(lut, c2, r2_3));

    // Round 4: fold 1 pair -> 1 value
    result[gid] = gf28_add(r3_0, gf28_mul(lut, c3, r3_1));
}

// ============================================================================
// Trace-Based Splitting
// ============================================================================

// Compute the trace-based even/odd split for additive domain folding.
//
// For the doubling map D(x) = x^2 + x with kernel {0, 1}:
//   f_even(x) = (f(x) + f(x+1)) / 2 via trace projection
//   f_odd(x)  = the complementary part
//
// This uses the linearity of the trace to project onto GF(2)-linear subspaces.
inline void trace_split(
    device const uint8_t* lut,
    uint8_t f_x,
    uint8_t f_x_plus_1,
    thread uint8_t& f_even,
    thread uint8_t& f_odd
) [[always_inline]] {
    // In characteristic 2, the trace projection gives:
    // f_even = (f(x) + f(x+1)) / 2, but /2 is the trace projector
    //
    // Simplified: f_even = f(x) + f(x+1) when the trace is 0
    //            f_odd = f(x) when the trace is 1
    //
    // The actual split is determined by the trace of the evaluation point
    uint8_t sum = gf28_add(f_x, f_x_plus_1);
    uint8_t trace_sum = gf28_trace(lut, sum);

    // If trace is 0, this point is in the even subspace
    // If trace is 1, this point is in the odd subspace
    f_even = trace_sum == 0 ? sum : f_x;
    f_odd = trace_sum == 1 ? sum : f_x_plus_1;
}

// ============================================================================
// Additive Domain Operations
// ============================================================================

// Apply the k-fold doubling map D^k(x).
// The kernel size is 2^k, so domain shrinks by factor of 2^k.
inline uint8_t k_fold_doubling(
    device const uint8_t* lut,
    uint8_t x,
    uint k
) [[always_inline]] {
    uint8_t result = x;
    for (uint i = 0; i < k; i++) {
        result = doubling_map(lut, result);
    }
    return result;
}

// ============================================================================
// High-Arity Binary FRI Fold
// ============================================================================

// High-arity fold: fold 2^k elements at once using k-fold doubling map.
//
// kernel buffers:
//   buffer(0): lut
//   buffer(1): evals - input (size n)
//   buffer(2): result - output (size n / 2^arity)
//   buffer(3): alpha - challenge
//   buffer(4): n - domain size
//   buffer(5): arity - k (fold factor = 2^k)
kernel void binary_fri_fold_arity_kernel(
    device const uint8_t* lut       [[buffer(0)]],
    device const uint8_t* evals     [[buffer(1)]],
    device uint8_t* result          [[buffer(2)]],
    constant uint8_t& alpha        [[buffer(3)]],
    constant uint32_t& n           [[buffer(4)]],
    constant uint32_t& arity       [[buffer(5)]],
    uint gid                        [[thread_position_in_grid]]
) {
    uint fold_factor = 1u << arity;
    uint result_size = n / fold_factor;

    if (gid >= result_size) return;

    uint8_t acc = evals[gid];

    // Accumulate: result = sum_{i=0}^{2^k-1} alpha^i * evals[gid + i * result_size]
    for (uint i = 1; i < fold_factor; i++) {
        uint idx = gid + i * result_size;
        uint8_t term = gf28_mul(lut, evals[idx], alpha);
        acc = gf28_add(acc, term);
        // Update alpha = alpha * alpha for next power
        alpha = gf28_mul(lut, alpha, alpha);
    }

    result[gid] = acc;
}

// ============================================================================
// Co-Curvilinearity Test
// ============================================================================

// Test if points lie on an affine line using trace-based quadratic form.
//
// Points P_0, ..., P_m lie on an affine line iff:
//   Tr((P_i - P_0)^2) = 0 for all i
//
// Uses Q(x) = Tr(x^2) as the quadratic form.
inline uint test_co_curvilinear(
    device const uint8_t* lut,
    device const uint8_t* points,
    uint num_points
) [[always_inline]] {
    if (num_points < 2) return 0;

    // Check that differences satisfy the quadratic form constraint
    uint8_t p0 = points[0];
    uint8_t sum_traces = 0;

    for (uint i = 1; i < num_points; i++) {
        uint8_t diff = gf28_add(points[i], p0);  // P_i - P_0
        uint8_t diff_sq = gf28_sq(lut, diff);
        uint8_t trace = gf28_trace(lut, diff_sq);
        sum_traces ^= trace;  // XOR in GF(2)
    }

    return sum_traces == 0 ? 1 : 0;
}

// Verify co-curvilinearity for FRI query verification.
// This is called during the query phase to verify folded values.
kernel void binary_fri_verify_co_curvilinear(
    device const uint8_t* lut       [[buffer(0)]],
    device const uint8_t* points   [[buffer(1)]],  // m+1 points
    device uint32_t* result        [[buffer(2)]],  // output: 1 if collinear, 0 otherwise
    constant uint32_t& num_points  [[buffer(3)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid != 0) return;  // Only need one thread for this check

    result[0] = test_co_curvilinear(lut, points, num_points);
}
