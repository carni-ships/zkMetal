// Poseidon2 batch sponge GPU kernel for parallel Fiat-Shamir transcript processing
//
// Each thread processes one independent transcript's Poseidon2 sponge:
//   1. Initialize state with domain tag in capacity element
//   2. Absorb variable-length message (rate=2, capacity=1)
//   3. Squeeze one or more output elements
//
// This enables GPU-accelerated batch proof generation where N independent
// proofs each need their own Fiat-Shamir transcript.

#include "../fields/bn254_fr.metal"

// ============================================================================
// Poseidon2 sponge helpers (inlined from poseidon2_permutation.metal)
// ============================================================================

// S-box: x -> x^5
Fr batch_sponge_sbox(Fr x) {
    Fr x2 = fr_mul(x, x);
    Fr x4 = fr_mul(x2, x2);
    return fr_mul(x4, x);
}

// External linear layer: circulant [2,1,1] for t=3
void batch_sponge_external_layer(thread Fr &s0, thread Fr &s1, thread Fr &s2) {
    Fr sum = fr_reduce(fr_add_lazy(fr_add_lazy(s0, s1), s2));
    s0 = fr_add_lazy(s0, sum);
    s1 = fr_add_lazy(s1, sum);
    s2 = fr_add_lazy(s2, sum);
}

// Internal linear layer: M_I = [[2,1,1],[1,2,1],[1,1,3]]
void batch_sponge_internal_layer(thread Fr &s0, thread Fr &s1, thread Fr &s2) {
    Fr sum = fr_add(fr_add(s0, s1), s2);
    s0 = fr_add(s0, sum);
    s1 = fr_add(s1, sum);
    s2 = fr_add(fr_add(s2, sum), s2);
}

// Full Poseidon2 permutation (t=3) — inlined for sponge use
void batch_sponge_permute(thread Fr &s0, thread Fr &s1, thread Fr &s2,
                          constant Fr *rc) {
    // Initial external linear layer
    batch_sponge_external_layer(s0, s1, s2);

    // First half of full rounds (0..3)
    for (uint r = 0; r < 4; r++) {
        uint rc_base = r * 3;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s0 = batch_sponge_sbox(s0);
        s1 = batch_sponge_sbox(s1);
        s2 = batch_sponge_sbox(s2);
        batch_sponge_external_layer(s0, s1, s2);
    }

    // Reduce before partial rounds
    s0 = fr_reduce(s0); s1 = fr_reduce(s1); s2 = fr_reduce(s2);

    // Partial rounds (4..59)
    for (uint r = 4; r < 60; r++) {
        s0 = fr_add_lazy(s0, rc[r * 3]);
        s0 = batch_sponge_sbox(s0);
        batch_sponge_internal_layer(s0, s1, s2);
    }

    // Second half of full rounds (60..63)
    for (uint r = 60; r < 64; r++) {
        uint rc_base = r * 3;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s0 = batch_sponge_sbox(s0);
        s1 = batch_sponge_sbox(s1);
        s2 = batch_sponge_sbox(s2);
        batch_sponge_external_layer(s0, s1, s2);
    }

    s0 = fr_reduce(s0);
    s1 = fr_reduce(s1);
    s2 = fr_reduce(s2);
}

// ============================================================================
// Batch absorb kernel
// ============================================================================
//
// Each thread i absorbs message[i] (of length msgLen) into its own sponge,
// then squeezes squeezeCount output elements.
//
// Layout:
//   states_in:  N * 3 Fr elements (s0, s1, s2 per transcript)
//   messages:   N * msgLen Fr elements (uniform-length messages)
//   states_out: N * 3 Fr elements (updated sponge states)
//   squeezed:   N * squeezeCount Fr elements (output challenges)
//   absorbed_counts: N uint32 values (how many rate cells filled before this call)
//
// The kernel handles multi-block absorption: for each pair of message elements,
// it adds them to rate positions and permutes when rate is full.

kernel void poseidon2_batch_absorb_squeeze(
    device const Fr* states_in       [[buffer(0)]],
    device const Fr* messages        [[buffer(1)]],
    device Fr* states_out            [[buffer(2)]],
    device Fr* squeezed              [[buffer(3)]],
    constant Fr* rc                  [[buffer(4)]],
    constant uint& count             [[buffer(5)]],  // N transcripts
    constant uint& msg_len           [[buffer(6)]],  // elements per message
    constant uint& squeeze_count     [[buffer(7)]],  // outputs per transcript
    device const uint* absorbed_in   [[buffer(8)]],  // absorbed count per transcript
    device uint* absorbed_out        [[buffer(9)]],  // updated absorbed count
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    // Load sponge state
    uint sbase = gid * 3;
    Fr s0 = states_in[sbase];
    Fr s1 = states_in[sbase + 1];
    Fr s2 = states_in[sbase + 2];
    uint absorbed = absorbed_in[gid];

    // Absorb message elements
    uint mbase = gid * msg_len;
    for (uint i = 0; i < msg_len; i++) {
        Fr elem = messages[mbase + i];
        if (absorbed == 0) {
            s0 = fr_add(s0, elem);
        } else {
            s1 = fr_add(s1, elem);
        }
        absorbed++;
        if (absorbed == 2) {
            batch_sponge_permute(s0, s1, s2, rc);
            absorbed = 0;
        }
    }

    // Squeeze phase: permute if dirty, then output from rate
    if (squeeze_count > 0) {
        // Finalize: permute if anything was absorbed
        if (absorbed > 0 || msg_len > 0) {
            batch_sponge_permute(s0, s1, s2, rc);
            absorbed = 0;
        }

        uint obase = gid * squeeze_count;
        uint squeeze_pos = 0;
        for (uint i = 0; i < squeeze_count; i++) {
            if (squeeze_pos >= 2) {
                batch_sponge_permute(s0, s1, s2, rc);
                squeeze_pos = 0;
            }
            if (squeeze_pos == 0) {
                squeezed[obase + i] = s0;
            } else {
                squeezed[obase + i] = s1;
            }
            squeeze_pos++;
        }

        // After squeezing, state is dirty
        // Store updated state (post-squeeze)
        states_out[sbase] = s0;
        states_out[sbase + 1] = s1;
        states_out[sbase + 2] = s2;
        absorbed_out[gid] = 0;  // reset after squeeze
    } else {
        // No squeeze, just store updated absorb state
        states_out[sbase] = s0;
        states_out[sbase + 1] = s1;
        states_out[sbase + 2] = s2;
        absorbed_out[gid] = absorbed;
    }
}

// ============================================================================
// Batch absorb-only kernel (no squeeze)
// ============================================================================
//
// Absorbs variable-length messages. Each transcript's message starts at
// offsets[gid] and has length (offsets[gid+1] - offsets[gid]) in the flat
// messages array.

kernel void poseidon2_batch_absorb_varlen(
    device const Fr* states_in       [[buffer(0)]],
    device const Fr* messages        [[buffer(1)]],
    device Fr* states_out            [[buffer(2)]],
    constant Fr* rc                  [[buffer(3)]],
    constant uint& count             [[buffer(4)]],
    device const uint* offsets       [[buffer(5)]],  // N+1 offsets into messages
    device const uint* absorbed_in   [[buffer(6)]],
    device uint* absorbed_out        [[buffer(7)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint sbase = gid * 3;
    Fr s0 = states_in[sbase];
    Fr s1 = states_in[sbase + 1];
    Fr s2 = states_in[sbase + 2];
    uint absorbed = absorbed_in[gid];

    uint msg_start = offsets[gid];
    uint msg_end = offsets[gid + 1];

    for (uint i = msg_start; i < msg_end; i++) {
        Fr elem = messages[i];
        if (absorbed == 0) {
            s0 = fr_add(s0, elem);
        } else {
            s1 = fr_add(s1, elem);
        }
        absorbed++;
        if (absorbed == 2) {
            batch_sponge_permute(s0, s1, s2, rc);
            absorbed = 0;
        }
    }

    states_out[sbase] = s0;
    states_out[sbase + 1] = s1;
    states_out[sbase + 2] = s2;
    absorbed_out[gid] = absorbed;
}

// ============================================================================
// Batch squeeze-only kernel
// ============================================================================

kernel void poseidon2_batch_squeeze(
    device const Fr* states_in       [[buffer(0)]],
    device Fr* states_out            [[buffer(1)]],
    device Fr* squeezed              [[buffer(2)]],
    constant Fr* rc                  [[buffer(3)]],
    constant uint& count             [[buffer(4)]],
    constant uint& squeeze_count     [[buffer(5)]],
    device const uint* absorbed_in   [[buffer(6)]],
    device uint* absorbed_out        [[buffer(7)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint sbase = gid * 3;
    Fr s0 = states_in[sbase];
    Fr s1 = states_in[sbase + 1];
    Fr s2 = states_in[sbase + 2];
    uint absorbed = absorbed_in[gid];

    // Finalize: permute if any data was absorbed
    if (absorbed > 0) {
        batch_sponge_permute(s0, s1, s2, rc);
    }

    uint obase = gid * squeeze_count;
    uint squeeze_pos = 0;
    for (uint i = 0; i < squeeze_count; i++) {
        if (squeeze_pos >= 2) {
            batch_sponge_permute(s0, s1, s2, rc);
            squeeze_pos = 0;
        }
        if (squeeze_pos == 0) {
            squeezed[obase + i] = s0;
        } else {
            squeezed[obase + i] = s1;
        }
        squeeze_pos++;
    }

    states_out[sbase] = s0;
    states_out[sbase + 1] = s1;
    states_out[sbase + 2] = s2;
    absorbed_out[gid] = 0;
}
