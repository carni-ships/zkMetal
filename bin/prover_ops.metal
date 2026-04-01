// BN254 Scalar Field (Fr) Operations and Prover Kernels for Metal GPU
//
// Implements Fr arithmetic matching barretenberg's exact lazy reduction behavior.
// Uses 4x64-bit limb CIOS Montgomery multiplication for byte-identical output.
//
// BN254 scalar field order: r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// Fr elements are 256-bit values in Montgomery form, stored as 4x64-bit limbs (little-endian).
// All values are in "coarse form" [0, 2p) — matching barretenberg's lazy reduction.

#include <metal_stdlib>
using namespace metal;

// --- Fr (Scalar Field) Type ---

struct Fr {
    uint v[8]; // 256-bit value as 8x32-bit limbs (little-endian) for buffer I/O
};

// --- 64-bit Constants ---

// BN254 Fr modulus p in 64-bit limbs (little-endian)
constant ulong FR_P64[4] = {
    0x43e1f593f0000001UL,
    0x2833e84879b97091UL,
    0xb85045b68181585dUL,
    0x30644e72e131a029UL
};

// 2p in 64-bit limbs
constant ulong FR_2P64[4] = {
    0x87c3eb27e0000002UL,
    0x5067d090f372e122UL,
    0x70a08b6d0302b0baUL,
    0x60c89ce5c2634053UL
};

// -(2p) mod 2^256 = 2^256 - 2p, for add overflow detection
constant ulong FR_2P_NEG64[4] = {
    0x783c14d81ffffffeUL,
    0xAF982F6F0C8D1EDDUL,
    0x8f5f7492fcfd4f45UL,
    0x9f37631a3d9cbfacUL
};

// -(r^(-1)) mod 2^64  (Montgomery constant for 64-bit CIOS)
constant ulong FR_R_INV64 = 0xc2e1f593efffffffUL;

// --- 128-bit Arithmetic Helpers ---

// 64x64 → 128 multiply
struct u128 { ulong lo; ulong hi; };

u128 mul_wide(ulong a, ulong b) {
    ulong a_lo = a & 0xFFFFFFFF;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFF;
    ulong b_hi = b >> 32;

    ulong t0 = a_lo * b_lo;
    ulong t1 = a_lo * b_hi;
    ulong t2 = a_hi * b_lo;
    ulong t3 = a_hi * b_hi;

    ulong cross = t1 + t2;
    ulong cross_carry = (cross < t1) ? (1UL << 32) : 0UL;

    ulong lo = t0 + (cross << 32);
    ulong lo_carry = (lo < t0) ? 1UL : 0UL;
    ulong hi = t3 + (cross >> 32) + cross_carry + lo_carry;

    return { lo, hi };
}

// mac: a + b*c + carry_in → (out, carry_out) where out = low 64, carry_out = high 64
ulong mac_out(ulong a, ulong b, ulong c, ulong carry_in, thread ulong &carry_out) {
    u128 bc = mul_wide(b, c);
    // a + bc.lo
    ulong sum_lo = a + bc.lo;
    ulong c1 = (sum_lo < a) ? 1UL : 0UL;
    // + carry_in
    sum_lo += carry_in;
    ulong c2 = (sum_lo < carry_in) ? 1UL : 0UL;
    carry_out = bc.hi + c1 + c2;
    return sum_lo;
}

// mac that writes to out and carry_out (void version)
void mac_void(ulong a, ulong b, ulong c, ulong carry_in, thread ulong &out, thread ulong &carry_out) {
    out = mac_out(a, b, c, carry_in, carry_out);
}

// mac_mini: a + b*c → (result, carry)
ulong mac_mini_ret(ulong a, ulong b, ulong c, thread ulong &carry_out) {
    u128 bc = mul_wide(b, c);
    ulong sum = a + bc.lo;
    ulong c1 = (sum < a) ? 1UL : 0UL;
    carry_out = bc.hi + c1;
    return sum;
}

// mac_mini void: a + b*c → (out, carry_out)
void mac_mini_void(ulong a, ulong b, ulong c, thread ulong &out, thread ulong &carry_out) {
    u128 bc = mul_wide(b, c);
    out = a + bc.lo;
    ulong c1 = (out < a) ? 1UL : 0UL;
    carry_out = bc.hi + c1;
}

// mac_discard_lo: high 64 bits of (a + b*c)
ulong mac_discard_lo(ulong a, ulong b, ulong c) {
    u128 bc = mul_wide(b, c);
    ulong sum = a + bc.lo;
    ulong c1 = (sum < a) ? 1UL : 0UL;
    return bc.hi + c1;
}

// addc: a + b + carry_in → (result, carry_out)
ulong addc(ulong a, ulong b, ulong carry_in, thread ulong &carry_out) {
    ulong sum = a + b;
    ulong c1 = (sum < a) ? 1UL : 0UL;
    sum += carry_in;
    ulong c2 = (sum < carry_in) ? 1UL : 0UL;
    carry_out = c1 + c2;
    return sum;
}

// sbb: a - b - (borrow_in != 0) → (result, borrow_out)
// borrow_in and borrow_out are in {0, 0xFFFFFFFFFFFFFFFF}
ulong sbb(ulong a, ulong b, ulong borrow_in, thread ulong &borrow_out) {
    ulong borrow_val = (borrow_in != 0) ? 1UL : 0UL;
    ulong t = a - b;
    ulong b1 = (a < b) ? 0xFFFFFFFFFFFFFFFFUL : 0UL;
    t -= borrow_val;
    ulong b2 = ((a - b) < borrow_val) ? 0xFFFFFFFFFFFFFFFFUL : 0UL;
    // Actually barretenberg's sbb is more subtle. Let me match it exactly.
    // borrow_out = 0 (no borrow) or 0xFFFFFFFFFFFFFFFF (borrow)
    borrow_out = b1 | b2;
    return t;
}

// --- Fr Conversion (8x32 ↔ 4x64) ---

void fr_to_64(Fr f, thread ulong out[4]) {
    out[0] = ulong(f.v[0]) | (ulong(f.v[1]) << 32);
    out[1] = ulong(f.v[2]) | (ulong(f.v[3]) << 32);
    out[2] = ulong(f.v[4]) | (ulong(f.v[5]) << 32);
    out[3] = ulong(f.v[6]) | (ulong(f.v[7]) << 32);
}

Fr fr_from_64(thread ulong a[4]) {
    Fr r;
    r.v[0] = uint(a[0]); r.v[1] = uint(a[0] >> 32);
    r.v[2] = uint(a[1]); r.v[3] = uint(a[1] >> 32);
    r.v[4] = uint(a[2]); r.v[5] = uint(a[2] >> 32);
    r.v[6] = uint(a[3]); r.v[7] = uint(a[3] >> 32);
    return r;
}

// --- Montgomery Multiplication (4x64 CIOS, matching barretenberg exactly) ---
// NO final reduction — result is in [0, 2p) coarse form.

Fr fr_mul(Fr a_in, Fr b_in) {
    ulong a[4], b[4];
    fr_to_64(a_in, a);
    fr_to_64(b_in, b);

    // Iteration 0: process a[0]
    // Two carry chains: 'c' for product accumulation, 'r' for reduction accumulation
    // BB names: c = product carry, a = reduction carry
    u128 w = mul_wide(a[0], b[0]);
    ulong t0 = w.lo;
    ulong c = w.hi;  // product carry chain
    ulong k = t0 * FR_R_INV64;
    ulong r = mac_discard_lo(t0, k, FR_P64[0]);  // reduction carry chain

    ulong t1 = mac_mini_ret(r, a[0], b[1], r);       // uses reduction carry
    mac_void(t1, k, FR_P64[1], c, t0, c);            // uses product carry
    ulong t2 = mac_mini_ret(r, a[0], b[2], r);
    mac_void(t2, k, FR_P64[2], c, t1, c);
    ulong t3 = mac_mini_ret(r, a[0], b[3], r);
    mac_void(t3, k, FR_P64[3], c, t2, c);
    t3 = c + r;

    // Iteration 1: process a[1]
    mac_mini_void(t0, a[1], b[0], t0, r);
    k = t0 * FR_R_INV64;
    c = mac_discard_lo(t0, k, FR_P64[0]);
    mac_void(t1, a[1], b[1], r, t1, r);
    mac_void(t1, k, FR_P64[1], c, t0, c);
    mac_void(t2, a[1], b[2], r, t2, r);
    mac_void(t2, k, FR_P64[2], c, t1, c);
    mac_void(t3, a[1], b[3], r, t3, r);
    mac_void(t3, k, FR_P64[3], c, t2, c);
    t3 = c + r;

    // Iteration 2: process a[2]
    mac_mini_void(t0, a[2], b[0], t0, r);
    k = t0 * FR_R_INV64;
    c = mac_discard_lo(t0, k, FR_P64[0]);
    mac_void(t1, a[2], b[1], r, t1, r);
    mac_void(t1, k, FR_P64[1], c, t0, c);
    mac_void(t2, a[2], b[2], r, t2, r);
    mac_void(t2, k, FR_P64[2], c, t1, c);
    mac_void(t3, a[2], b[3], r, t3, r);
    mac_void(t3, k, FR_P64[3], c, t2, c);
    t3 = c + r;

    // Iteration 3: process a[3]
    mac_mini_void(t0, a[3], b[0], t0, r);
    k = t0 * FR_R_INV64;
    c = mac_discard_lo(t0, k, FR_P64[0]);
    mac_void(t1, a[3], b[1], r, t1, r);
    mac_void(t1, k, FR_P64[1], c, t0, c);
    mac_void(t2, a[3], b[2], r, t2, r);
    mac_void(t2, k, FR_P64[2], c, t1, c);
    mac_void(t3, a[3], b[3], r, t3, r);
    mac_void(t3, k, FR_P64[3], c, t2, c);
    t3 = c + r;

    // NO final reduction — return in coarse form [0, 2p)
    ulong result[4] = { t0, t1, t2, t3 };
    return fr_from_64(result);
}

// --- Modular Addition (matching barretenberg's small-modulus lazy add) ---
// Input: [0, 2p), Output: [0, 2p)
// Subtracts 2p if sum >= 2p

Fr fr_add(Fr a_in, Fr b_in) {
    ulong a[4], b[4];
    fr_to_64(a_in, a);
    fr_to_64(b_in, b);

    // Compute sum
    ulong r0 = a[0] + b[0];
    ulong c = (r0 < a[0]) ? 1UL : 0UL;
    ulong r1 = addc(a[1], b[1], c, c);
    ulong r2 = addc(a[2], b[2], c, c);
    // For small modulus (254-bit), top limb add can't overflow 64 bits
    // since both are < 2p and 2p < 2^255
    ulong r3 = a[3] + b[3] + c;

    // Check if sum >= 2p by adding (2^256 - 2p) and checking carry
    ulong t0 = r0 + FR_2P_NEG64[0];
    c = (t0 < FR_2P_NEG64[0]) ? 1UL : 0UL;
    ulong t1 = addc(r1, FR_2P_NEG64[1], c, c);
    ulong t2 = addc(r2, FR_2P_NEG64[2], c, c);
    ulong t3 = addc(r3, FR_2P_NEG64[3], c, c);

    // c == 1 means sum >= 2p, use reduced (t values = sum - 2p)
    // c == 0 means sum < 2p, use original (r values)
    ulong mask = 0UL - c;  // all 1s if c==1, all 0s if c==0
    ulong inv_mask = ~mask;

    ulong result[4] = {
        (r0 & inv_mask) | (t0 & mask),
        (r1 & inv_mask) | (t1 & mask),
        (r2 & inv_mask) | (t2 & mask),
        (r3 & inv_mask) | (t3 & mask)
    };
    return fr_from_64(result);
}

// --- Modular Subtraction (matching barretenberg's small-modulus lazy sub) ---
// Input: [0, 2p), Output: [0, 2p)
// Adds 2p if borrow

Fr fr_sub(Fr a_in, Fr b_in) {
    ulong a[4], b[4];
    fr_to_64(a_in, a);
    fr_to_64(b_in, b);

    ulong borrow = 0;
    ulong r0 = sbb(a[0], b[0], borrow, borrow);
    ulong r1 = sbb(a[1], b[1], borrow, borrow);
    ulong r2 = sbb(a[2], b[2], borrow, borrow);
    ulong r3 = sbb(a[3], b[3], borrow, borrow);

    // If borrow (borrow == 0xFFFF...), add 2p
    r0 += (FR_2P64[0] & borrow);
    ulong carry = (r0 < (FR_2P64[0] & borrow)) ? 1UL : 0UL;
    r1 = addc(r1, FR_2P64[1] & borrow, carry, carry);
    r2 = addc(r2, FR_2P64[2] & borrow, carry, carry);
    r3 += (FR_2P64[3] & borrow) + carry;

    ulong result[4] = { r0, r1, r2, r3 };
    return fr_from_64(result);
}

// --- Utility Functions ---

Fr fr_zero() {
    Fr r;
    for (int i = 0; i < 8; i++) r.v[i] = 0;
    return r;
}

bool fr_is_zero(Fr a) {
    for (int i = 0; i < 8; i++) {
        if (a.v[i] != 0) return false;
    }
    return true;
}

// ============================================================
// Kernel: Fr Arithmetic Test (debug only)
// ============================================================
kernel void fr_test(
    device const Fr* a_buf   [[buffer(0)]],
    device const Fr* b_buf   [[buffer(1)]],
    device Fr*       out     [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    Fr a = a_buf[0];
    Fr b = b_buf[0];
    out[0] = fr_mul(a, b);
    out[1] = fr_add(a, b);
    out[2] = fr_sub(a, b);
}

// ============================================================
// Kernel: Debug Buffer Read
// ============================================================
kernel void debug_read(
    device const Fr* in_buf  [[buffer(0)]],
    device Fr*       out_buf [[buffer(1)]],
    device const uint* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint n = params[0];
    if (tid >= n) return;
    out_buf[tid] = in_buf[tid];
}

// ============================================================
// Kernel: Gemini Polynomial Folding
// ============================================================
// Computes: A_fold[j] = A[2j] + u * (A[2j+1] - A[2j])
// Must match barretenberg's CPU formula exactly for byte-identical output.
// u_buf[0] = u, u_buf[1] = (1-u) (1-u not used here but kept for API compat)

kernel void gemini_fold(
    device const Fr* A_in      [[buffer(0)]],
    device Fr*       A_out     [[buffer(1)]],
    device const Fr* u_buf     [[buffer(2)]],
    device const uint* params  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint n_fold = params[0];
    if (tid >= n_fold) return;

    Fr u = u_buf[0];
    Fr even = A_in[tid * 2];
    Fr odd  = A_in[tid * 2 + 1];
    A_out[tid] = fr_add(even, fr_mul(u, fr_sub(odd, even)));
}

// ============================================================
// Kernel: Batch Polynomial Add
// ============================================================
kernel void poly_add(
    device const Fr* a     [[buffer(0)]],
    device const Fr* b     [[buffer(1)]],
    device Fr*       out   [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint n = params[0];
    if (tid >= n) return;
    out[tid] = fr_add(a[tid], b[tid]);
}

// ============================================================
// Kernel: Batch Polynomial Sub
// ============================================================
kernel void poly_sub(
    device const Fr* a     [[buffer(0)]],
    device const Fr* b     [[buffer(1)]],
    device Fr*       out   [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint n = params[0];
    if (tid >= n) return;
    out[tid] = fr_sub(a[tid], b[tid]);
}

// ============================================================
// Kernel: Scalar-Polynomial Multiply
// ============================================================
kernel void poly_scalar_mul(
    device const Fr* a       [[buffer(0)]],
    device Fr*       out     [[buffer(1)]],
    device const Fr* scalar  [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint n = params[0];
    if (tid >= n) return;
    Fr s = scalar[0];
    out[tid] = fr_mul(s, a[tid]);
}

// ============================================================
// Kernel: Partially Evaluate Multilinear Polynomial
// ============================================================
kernel void partial_evaluate(
    device Fr*       poly     [[buffer(0)]],
    device const Fr* u_buf    [[buffer(1)]],
    device const uint* params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint n_half = params[0];
    if (tid >= n_half) return;

    Fr u = u_buf[0];
    Fr lo = poly[tid];
    Fr hi = poly[tid + n_half];
    poly[tid] = fr_add(lo, fr_mul(u, fr_sub(hi, lo)));
}

// ============================================================
// Kernel: Batch Polynomial Fold (Sumcheck Partial Evaluate)
// ============================================================
kernel void batch_fold(
    device const Fr*   packed_src   [[buffer(0)]],
    device Fr*         packed_dst   [[buffer(1)]],
    device const Fr*   u_buf        [[buffer(2)]],
    device const uint* src_offsets  [[buffer(3)]],
    device const uint* dst_offsets  [[buffer(4)]],
    device const uint* halves       [[buffer(5)]],
    device const uint* params       [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint poly_idx = tid.y;
    uint elem_idx = tid.x;

    uint num_polys = params[0];
    if (poly_idx >= num_polys) return;

    uint n_half = halves[poly_idx];
    if (elem_idx >= n_half) return;

    uint s_off = src_offsets[poly_idx];
    uint d_off = dst_offsets[poly_idx];

    Fr u    = u_buf[0];
    Fr even = packed_src[s_off + elem_idx * 2];
    Fr odd  = packed_src[s_off + elem_idx * 2 + 1];
    packed_dst[d_off + elem_idx] = fr_add(even, fr_mul(u, fr_sub(odd, even)));
}

// ============================================================
// Kernel: Batch Polynomial Accumulate with Scalar
// ============================================================
kernel void poly_add_scaled(
    device Fr*       acc     [[buffer(0)]],
    device const Fr* poly    [[buffer(1)]],
    device const Fr* scalar  [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint n = params[0];
    if (tid >= n) return;
    Fr s = scalar[0];
    Fr term = fr_mul(s, poly[tid]);
    acc[tid] = fr_add(acc[tid], term);
}

// ============================================================
// Kernel: Fold with Start Index Offset (Zero-Copy Sumcheck)
// ============================================================
// Handles virtual zeros before src_start and beyond src_end:
// elements at logical indices < src_start or >= src_end are
// treated as zero without memory access.
// dst is always written from physical index 0 (dst_start = 0).
kernel void fold_offset(
    device const Fr*   src    [[buffer(0)]],
    device Fr*         dst    [[buffer(1)]],
    device const Fr*   u_buf  [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint n_fold    = params[0];
    uint src_start = params[1];
    uint src_end   = params[2];

    if (tid >= n_fold) return;

    Fr u = u_buf[0];
    uint logical_even = tid * 2;
    uint logical_odd  = logical_even + 1;

    Fr lo, hi;
    if (logical_even >= src_start && logical_even < src_end) {
        lo = src[logical_even - src_start];
    } else {
        lo = fr_zero();
    }
    if (logical_odd >= src_start && logical_odd < src_end) {
        hi = src[logical_odd - src_start];
    } else {
        hi = fr_zero();
    }

    dst[tid] = fr_add(lo, fr_mul(u, fr_sub(hi, lo)));
}

