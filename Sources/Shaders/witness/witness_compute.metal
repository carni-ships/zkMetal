// GPU witness computation kernels for circuit witness generation
//
// Computes intermediate wire values from inputs for Plonk-style circuits.
// Each thread processes one gate within a dependency level (layer).
// Gates in the same layer are independent and execute in parallel.
//
// Supported fields: BN254 Fr (256-bit Montgomery), BabyBear (32-bit)

#include <metal_stdlib>
using namespace metal;

// ============================================================
// BN254 Fr field (included inline at compile time from bn254_fr.metal)
// ============================================================

// ============================================================
// BabyBear field (included inline at compile time from babybear.metal)
// ============================================================

// ============================================================
// Gate operation opcodes
// ============================================================

constant uint WITNESS_OP_ADD            = 0;  // out = left + right
constant uint WITNESS_OP_MUL            = 1;  // out = left * right
constant uint WITNESS_OP_LINEAR_COMBO   = 2;  // out = qL*left + qR*right + qC
constant uint WITNESS_OP_POSEIDON2      = 3;  // Poseidon2 round (full or partial)
constant uint WITNESS_OP_CONSTANT       = 4;  // out = constant value
constant uint WITNESS_OP_COPY           = 5;  // out = left

// ============================================================
// BN254 Fr Witness Kernels
// ============================================================

// Gate descriptor for BN254 witness computation:
// Packed as: [op, leftIdx, rightIdx, outIdx, constant_offset]
// For linear combination: constants at constant_offset are [qL, qR, qC] (3 Fr elements)
// For constant op: constants at constant_offset is 1 Fr element
// For poseidon2 round: leftIdx = state base wire, rightIdx = round_constant_offset,
//   outIdx = output state base wire, constant_offset encodes (full_round:1, t:16bits)

/// witness_add_bn254: For each gate, compute output = left + right (BN254 Fr).
/// Each thread processes one gate.
kernel void witness_add_bn254(
    device Fr* wires               [[buffer(0)]],  // wire values array
    device const uint* left_idx    [[buffer(1)]],  // left operand wire index per gate
    device const uint* right_idx   [[buffer(2)]],  // right operand wire index per gate
    device const uint* out_idx     [[buffer(3)]],  // output wire index per gate
    constant uint& num_gates       [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]])
{
    if (gid >= num_gates) return;

    Fr a = wires[left_idx[gid]];
    Fr b = wires[right_idx[gid]];
    wires[out_idx[gid]] = fr_add(a, b);
}

/// witness_mul_bn254: For each gate, compute output = left * right (BN254 Fr).
kernel void witness_mul_bn254(
    device Fr* wires               [[buffer(0)]],
    device const uint* left_idx    [[buffer(1)]],
    device const uint* right_idx   [[buffer(2)]],
    device const uint* out_idx     [[buffer(3)]],
    constant uint& num_gates       [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]])
{
    if (gid >= num_gates) return;

    Fr a = wires[left_idx[gid]];
    Fr b = wires[right_idx[gid]];
    wires[out_idx[gid]] = fr_mul(a, b);
}

/// witness_linear_combination_bn254: output = qL*a + qR*b + qC
/// constants buffer: [qL_0, qR_0, qC_0, qL_1, qR_1, qC_1, ...]
kernel void witness_linear_combination_bn254(
    device Fr* wires               [[buffer(0)]],
    device const uint* left_idx    [[buffer(1)]],
    device const uint* right_idx   [[buffer(2)]],
    device const uint* out_idx     [[buffer(3)]],
    device const Fr* constants     [[buffer(4)]],  // qL, qR, qC triples
    constant uint& num_gates       [[buffer(5)]],
    uint gid                       [[thread_position_in_grid]])
{
    if (gid >= num_gates) return;

    Fr a = wires[left_idx[gid]];
    Fr b = wires[right_idx[gid]];
    Fr qL = constants[gid * 3];
    Fr qR = constants[gid * 3 + 1];
    Fr qC = constants[gid * 3 + 2];

    // out = qL*a + qR*b + qC
    Fr term1 = fr_mul(qL, a);
    Fr term2 = fr_mul(qR, b);
    Fr sum = fr_add(term1, term2);
    wires[out_idx[gid]] = fr_add(sum, qC);
}

/// witness_poseidon2_round_bn254: One Poseidon2 round across many parallel instances.
/// For hash-heavy circuits. Each thread handles one Poseidon2 instance.
/// State width t=3: state[0..2] are read from wires, result written back.
///
/// Layout: For instance i, state is at wires[state_base + i*3 + 0..2]
/// Round constants are in the constants buffer at rc_offset + 0..2
/// full_round: if 1, apply S-box to all state elements; if 0, only to state[0]
kernel void witness_poseidon2_round_bn254(
    device Fr* wires               [[buffer(0)]],
    device const Fr* round_consts  [[buffer(1)]],  // round constants [rc0, rc1, rc2]
    constant uint& state_base      [[buffer(2)]],  // base index in wires for state arrays
    constant uint& num_instances   [[buffer(3)]],  // number of parallel Poseidon2 instances
    constant uint& full_round      [[buffer(4)]],  // 1 = full round, 0 = partial round
    uint gid                       [[thread_position_in_grid]])
{
    if (gid >= num_instances) return;

    uint base = state_base + gid * 3;

    // Load state
    Fr s0 = wires[base + 0];
    Fr s1 = wires[base + 1];
    Fr s2 = wires[base + 2];

    // Add round constants
    s0 = fr_add(s0, round_consts[0]);
    s1 = fr_add(s1, round_consts[1]);
    s2 = fr_add(s2, round_consts[2]);

    // S-box: x^5
    if (full_round) {
        // Full round: S-box on all elements
        Fr t0 = fr_sqr(s0); t0 = fr_sqr(t0); s0 = fr_mul(s0, t0);
        Fr t1 = fr_sqr(s1); t1 = fr_sqr(t1); s1 = fr_mul(s1, t1);
        Fr t2 = fr_sqr(s2); t2 = fr_sqr(t2); s2 = fr_mul(s2, t2);
    } else {
        // Partial round: S-box only on s0
        Fr t0 = fr_sqr(s0); t0 = fr_sqr(t0); s0 = fr_mul(s0, t0);
    }

    // MDS mixing (Poseidon2 t=3 diffusion)
    // t = s0 + s1 + s2
    Fr t = fr_add(s0, s1);
    t = fr_add(t, s2);
    // s_i = 2*s_i + t
    s0 = fr_add(s0, t);
    s1 = fr_add(s1, t);
    s2 = fr_add(s2, t);

    // Write back
    wires[base + 0] = s0;
    wires[base + 1] = s1;
    wires[base + 2] = s2;
}

// ============================================================
// BabyBear Witness Kernels
// ============================================================

/// witness_add_babybear: For each gate, compute output = left + right (BabyBear).
kernel void witness_add_babybear(
    device Bb* wires               [[buffer(0)]],
    device const uint* left_idx    [[buffer(1)]],
    device const uint* right_idx   [[buffer(2)]],
    device const uint* out_idx     [[buffer(3)]],
    constant uint& num_gates       [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]])
{
    if (gid >= num_gates) return;

    Bb a = wires[left_idx[gid]];
    Bb b = wires[right_idx[gid]];
    wires[out_idx[gid]] = bb_add(a, b);
}

/// witness_mul_babybear: For each gate, compute output = left * right (BabyBear).
kernel void witness_mul_babybear(
    device Bb* wires               [[buffer(0)]],
    device const uint* left_idx    [[buffer(1)]],
    device const uint* right_idx   [[buffer(2)]],
    device const uint* out_idx     [[buffer(3)]],
    constant uint& num_gates       [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]])
{
    if (gid >= num_gates) return;

    Bb a = wires[left_idx[gid]];
    Bb b = wires[right_idx[gid]];
    wires[out_idx[gid]] = bb_mul(a, b);
}

/// witness_linear_combination_babybear: output = qL*a + qR*b + qC
kernel void witness_linear_combination_babybear(
    device Bb* wires               [[buffer(0)]],
    device const uint* left_idx    [[buffer(1)]],
    device const uint* right_idx   [[buffer(2)]],
    device const uint* out_idx     [[buffer(3)]],
    device const Bb* constants     [[buffer(4)]],
    constant uint& num_gates       [[buffer(5)]],
    uint gid                       [[thread_position_in_grid]])
{
    if (gid >= num_gates) return;

    Bb a = wires[left_idx[gid]];
    Bb b = wires[right_idx[gid]];
    Bb qL = constants[gid * 3];
    Bb qR = constants[gid * 3 + 1];
    Bb qC = constants[gid * 3 + 2];

    Bb term1 = bb_mul(qL, a);
    Bb term2 = bb_mul(qR, b);
    Bb sum = bb_add(term1, term2);
    wires[out_idx[gid]] = bb_add(sum, qC);
}

/// witness_poseidon2_round_babybear: One Poseidon2 round across parallel instances (BabyBear).
kernel void witness_poseidon2_round_babybear(
    device Bb* wires               [[buffer(0)]],
    device const Bb* round_consts  [[buffer(1)]],
    constant uint& state_base      [[buffer(2)]],
    constant uint& num_instances   [[buffer(3)]],
    constant uint& full_round      [[buffer(4)]],
    uint gid                       [[thread_position_in_grid]])
{
    if (gid >= num_instances) return;

    uint base = state_base + gid * 3;

    Bb s0 = wires[base + 0];
    Bb s1 = wires[base + 1];
    Bb s2 = wires[base + 2];

    // Add round constants
    s0 = bb_add(s0, round_consts[0]);
    s1 = bb_add(s1, round_consts[1]);
    s2 = bb_add(s2, round_consts[2]);

    // S-box: x^5
    if (full_round) {
        Bb t0 = bb_sqr(s0); t0 = bb_sqr(t0); s0 = bb_mul(s0, t0);
        Bb t1 = bb_sqr(s1); t1 = bb_sqr(t1); s1 = bb_mul(s1, t1);
        Bb t2 = bb_sqr(s2); t2 = bb_sqr(t2); s2 = bb_mul(s2, t2);
    } else {
        Bb t0 = bb_sqr(s0); t0 = bb_sqr(t0); s0 = bb_mul(s0, t0);
    }

    // MDS mixing
    Bb t = bb_add(s0, s1);
    t = bb_add(t, s2);
    s0 = bb_add(s0, t);
    s1 = bb_add(s1, t);
    s2 = bb_add(s2, t);

    // Write back
    wires[base + 0] = s0;
    wires[base + 1] = s1;
    wires[base + 2] = s2;
}
