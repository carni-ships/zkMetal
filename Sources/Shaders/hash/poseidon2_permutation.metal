// Poseidon2 permutation GPU kernels for BN254 Fr
// Width-3 (t=3): standard ZK hash, d=5, rounds_f=8, rounds_p=56
// Width-4 (t=4): STARK trace hashing, d=5, rounds_f=8, rounds_p=56
//
// Each thread processes one independent permutation (batch parallelism).
// Round constants passed via buffer (not embedded) for flexibility.

#include "../fields/bn254_fr.metal"

// ============================================================================
// S-box: x -> x^5 (shared between width-3 and width-4)
// ============================================================================

Fr p2perm_sbox(Fr x) {
    Fr x2 = fr_mul(x, x);
    Fr x4 = fr_mul(x2, x2);
    return fr_mul(x4, x);
}

// ============================================================================
// Width-3 (t=3) linear layers
// ============================================================================

// External linear layer: circulant [2,1,1] for t=3
// M_E * [a,b,c] = [a+(a+b+c), b+(a+b+c), c+(a+b+c)]
// All outputs fully reduced to [0, p) to prevent value explosion
// through the partial rounds' internal layer.
void p2perm_external_layer_w3(thread Fr &s0, thread Fr &s1, thread Fr &s2) {
    Fr sum = fr_add(fr_add(s0, s1), s2);
    s0 = fr_add(s0, sum);
    s1 = fr_add(s1, sum);
    s2 = fr_add(s2, sum);
}

// Internal linear layer: M_I = [[2,1,1],[1,2,1],[1,1,3]]
// Inputs must be in [0, p) for fr_add to produce correct [0, p) outputs.
void p2perm_internal_layer_w3(thread Fr &s0, thread Fr &s1, thread Fr &s2) {
    Fr sum = fr_add(fr_add(s0, s1), s2);
    s0 = fr_add(s0, sum);
    s1 = fr_add(s1, sum);
    s2 = fr_add(fr_add(s2, sum), s2);  // s2 + sum + s2 = a+b+3c
}

// ============================================================================
// Width-4 (t=4) linear layers
// ============================================================================

// External linear layer: circ(5, 7, 1, 3) for t=4
// This is the Horizen Labs / Plonky3 standard for BN254 t=4.
// M_E * [a,b,c,d]:
//   out[0] = 5a + 7b + c + 3d
//   out[1] = 3a + 5b + 7c + d
//   out[2] = a + 3b + 5c + 7d
//   out[3] = 7a + b + 3c + 5d
void p2perm_external_layer_w4(thread Fr &s0, thread Fr &s1, thread Fr &s2, thread Fr &s3) {
    // Reduce inputs to [0, p) first
    Fr a = fr_reduce(s0), b = fr_reduce(s1), c = fr_reduce(s2), d = fr_reduce(s3);

    // sum = a + b + c + d (fully reduced)
    Fr sum = fr_add(fr_add(a, b), fr_add(c, d));

    // Direct computation of circ(5,7,1,3):
    // out0 = 5a + 7b + c + 3d
    // out1 = 3a + 5b + 7c + d
    // out2 = a + 3b + 5c + 7d
    // out3 = 7a + b + 3c + 5d
    //
    // Rewrite as: out_i = sum + 4*s_i + 6*s_{(i+1)%4} + 2*s_{(i+3)%4}
    // = sum + 4*s_i + 6*s_{i+1} + 2*s_{i-1}

    // 2x, 4x, 6x for each element
    Fr a2 = fr_add(a, a); Fr a4 = fr_add(a2, a2); Fr a6 = fr_add(a4, a2);
    Fr b2 = fr_add(b, b); Fr b4 = fr_add(b2, b2); Fr b6 = fr_add(b4, b2);
    Fr c2 = fr_add(c, c); Fr c4 = fr_add(c2, c2); Fr c6 = fr_add(c4, c2);
    Fr d2 = fr_add(d, d); Fr d4 = fr_add(d2, d2); Fr d6 = fr_add(d4, d2);

    s0 = fr_add(fr_add(sum, a4), fr_add(b6, d2));
    s1 = fr_add(fr_add(sum, b4), fr_add(c6, a2));
    s2 = fr_add(fr_add(sum, c4), fr_add(d6, b2));
    s3 = fr_add(fr_add(sum, d4), fr_add(a6, c2));
}

// Internal linear layer for t=4:
// M_I has diagonal [d0, d1, d2, d3] where we add sum of all elements.
// Standard Poseidon2 t=4: M_I = 1 + diag(d0-1, d1-1, d2-1, d3-1)
// Using diag constants [5, 7, 1, 3] (same as external for t=4 in HorizenLabs)
// Actually for internal: y_i = (d_i - 1) * x_i + sum(x_j)
// With constants passed as buffer for maximum flexibility.
//
// For BN254 t=4, the internal diffusion uses:
// M_I = I + [[1],[1],[1],[1]] * [1,1,1,1] where diagonal has extra weights
// Simplified: y_i = x_i + mu_i * x_i + sum  where mu_i are the internal diagonal weights
// Here we use: out[i] = diag[i] * s[i] + sum(s[j])
void p2perm_internal_layer_w4(thread Fr &s0, thread Fr &s1, thread Fr &s2, thread Fr &s3,
                               constant Fr *diag) {
    Fr sum = fr_add(fr_add(s0, s1), fr_add(s2, s3));
    s0 = fr_add(fr_mul(s0, diag[0]), sum);
    s1 = fr_add(fr_mul(s1, diag[1]), sum);
    s2 = fr_add(fr_mul(s2, diag[2]), sum);
    s3 = fr_add(fr_mul(s3, diag[3]), sum);
}

// ============================================================================
// Width-3 Permutation Kernel
// ============================================================================

// Full Poseidon2 permutation for t=3
// input: array of 3*n Fr elements
// output: array of 3*n Fr elements
// rc: 64*3 = 192 Fr elements (round constants)
kernel void poseidon2_permutation_bn254(
    device const Fr* input        [[buffer(0)]],
    device Fr* output             [[buffer(1)]],
    constant Fr* rc               [[buffer(2)]],
    constant uint& count          [[buffer(3)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint base = gid * 3;
    Fr s0 = input[base];
    Fr s1 = input[base + 1];
    Fr s2 = input[base + 2];

    // Initial external linear layer
    p2perm_external_layer_w3(s0, s1, s2);

    // First half of full rounds (0..3)
    #pragma unroll
    for (uint r = 0; r < 4; r++) {
        uint rc_base = r * 3;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s0 = p2perm_sbox(s0);
        s1 = p2perm_sbox(s1);
        s2 = p2perm_sbox(s2);
        p2perm_external_layer_w3(s0, s1, s2);
    }

    // Reduce before partial rounds
    s0 = fr_reduce(s0); s1 = fr_reduce(s1); s2 = fr_reduce(s2);

    // Partial rounds (4..59) — only s0 gets RC and S-box
    for (uint r = 4; r < 60; r++) {
        s0 = fr_add_lazy(s0, rc[r * 3]);
        s0 = p2perm_sbox(s0);
        p2perm_internal_layer_w3(s0, s1, s2);
    }

    // Second half of full rounds (60..63)
    #pragma unroll
    for (uint r = 60; r < 64; r++) {
        uint rc_base = r * 3;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s0 = p2perm_sbox(s0);
        s1 = p2perm_sbox(s1);
        s2 = p2perm_sbox(s2);
        p2perm_external_layer_w3(s0, s1, s2);
    }

    output[base] = fr_reduce(s0);
    output[base + 1] = fr_reduce(s1);
    output[base + 2] = fr_reduce(s2);
}

// ============================================================================
// Width-3 Two-to-one Compression Kernel
// ============================================================================

// Absorb 2 field elements, squeeze 1: state = [a, b, 0], output = permute(state)[0]
kernel void poseidon2_compress_bn254(
    device const Fr* input        [[buffer(0)]],
    device Fr* output             [[buffer(1)]],
    constant Fr* rc               [[buffer(2)]],
    constant uint& count          [[buffer(3)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    Fr s0 = input[gid * 2];
    Fr s1 = input[gid * 2 + 1];
    Fr s2 = fr_zero();

    p2perm_external_layer_w3(s0, s1, s2);

    #pragma unroll
    for (uint r = 0; r < 4; r++) {
        uint rc_base = r * 3;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s0 = p2perm_sbox(s0); s1 = p2perm_sbox(s1); s2 = p2perm_sbox(s2);
        p2perm_external_layer_w3(s0, s1, s2);
    }

    s0 = fr_reduce(s0); s1 = fr_reduce(s1); s2 = fr_reduce(s2);

    for (uint r = 4; r < 60; r++) {
        s0 = fr_add_lazy(s0, rc[r * 3]);
        s0 = p2perm_sbox(s0);
        p2perm_internal_layer_w3(s0, s1, s2);
    }

    #pragma unroll
    for (uint r = 60; r < 64; r++) {
        uint rc_base = r * 3;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s0 = p2perm_sbox(s0); s1 = p2perm_sbox(s1); s2 = p2perm_sbox(s2);
        p2perm_external_layer_w3(s0, s1, s2);
    }

    output[gid] = fr_reduce(s0);
}

// ============================================================================
// Width-4 Permutation Kernel (STARK trace hashing)
// ============================================================================

// Full Poseidon2 permutation for t=4
// input: array of 4*n Fr elements
// output: array of 4*n Fr elements
// rc: 64*4 = 256 Fr elements (round constants, only [r*4] used in partial rounds)
// diag: 4 Fr elements (internal diagonal constants)
kernel void poseidon2_permutation_bn254_width4(
    device const Fr* input        [[buffer(0)]],
    device Fr* output             [[buffer(1)]],
    constant Fr* rc               [[buffer(2)]],
    constant uint& count          [[buffer(3)]],
    constant Fr* diag             [[buffer(4)]],
    uint gid                      [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint base = gid * 4;
    Fr s0 = input[base];
    Fr s1 = input[base + 1];
    Fr s2 = input[base + 2];
    Fr s3 = input[base + 3];

    // Initial external linear layer
    p2perm_external_layer_w4(s0, s1, s2, s3);

    // First half of full rounds (0..3)
    #pragma unroll
    for (uint r = 0; r < 4; r++) {
        uint rc_base = r * 4;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s3 = fr_add_lazy(s3, rc[rc_base + 3]);
        s0 = p2perm_sbox(s0);
        s1 = p2perm_sbox(s1);
        s2 = p2perm_sbox(s2);
        s3 = p2perm_sbox(s3);
        p2perm_external_layer_w4(s0, s1, s2, s3);
    }

    // Reduce before partial rounds
    s0 = fr_reduce(s0); s1 = fr_reduce(s1); s2 = fr_reduce(s2); s3 = fr_reduce(s3);

    // Partial rounds (4..59) — only s0 gets RC and S-box
    for (uint r = 4; r < 60; r++) {
        s0 = fr_add_lazy(s0, rc[r * 4]);
        s0 = p2perm_sbox(s0);
        p2perm_internal_layer_w4(s0, s1, s2, s3, diag);
    }

    // Second half of full rounds (60..63)
    #pragma unroll
    for (uint r = 60; r < 64; r++) {
        uint rc_base = r * 4;
        s0 = fr_add_lazy(s0, rc[rc_base]);
        s1 = fr_add_lazy(s1, rc[rc_base + 1]);
        s2 = fr_add_lazy(s2, rc[rc_base + 2]);
        s3 = fr_add_lazy(s3, rc[rc_base + 3]);
        s0 = p2perm_sbox(s0);
        s1 = p2perm_sbox(s1);
        s2 = p2perm_sbox(s2);
        s3 = p2perm_sbox(s3);
        p2perm_external_layer_w4(s0, s1, s2, s3);
    }

    output[base] = fr_reduce(s0);
    output[base + 1] = fr_reduce(s1);
    output[base + 2] = fr_reduce(s2);
    output[base + 3] = fr_reduce(s3);
}
