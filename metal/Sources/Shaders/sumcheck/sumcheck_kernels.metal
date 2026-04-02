// Sumcheck protocol kernels for Metal GPU
// Used in STARK and GKR proof systems
// Core operation: partial evaluation of multilinear polynomial

#include "../fields/bn254_fr.metal"

// Reduce multilinear polynomial by fixing one variable
// Given evals[0..n-1] representing f(x_1,...,x_k) over boolean hypercube,
// compute evals_out[0..n/2-1] = (1-r)*evals[i] + r*evals[i + n/2]
// for challenge r, fixing the last variable to r.
kernel void sumcheck_reduce(
    device const Fr* evals          [[buffer(0)]],
    device Fr* evals_out            [[buffer(1)]],
    constant Fr* challenge          [[buffer(2)]],  // single Fr element (constant broadcast)
    constant uint& half_n           [[buffer(3)]],   // n/2
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= half_n) return;

    Fr a = evals[gid];            // f(..., x_k=0)
    Fr b = evals[gid + half_n];   // f(..., x_k=1)
    Fr r = challenge[0];

    // (1-r)*a + r*b = a + r*(b-a)
    Fr diff = fr_sub(b, a);
    Fr r_diff = fr_mul(r, diff);
    evals_out[gid] = fr_add(a, r_diff);
}

// Compute round polynomial: for variable x_k, evaluate the sum
// S_k(X) = sum_{(x_{k+1},...,x_m) in {0,1}} f(r_1,...,r_{k-1}, X, x_{k+1},...,x_m)
// at X=0, X=1, and X=2 (three evaluations determine the degree-2 polynomial)
//
// For each i in [0, quarter_n):
//   S(0) += evals[2*i]              (X=0, next_var=0) + evals[2*i + quarter_n*2] (X=0, next_var=1)
//   Wait, this needs careful indexing based on the multilinear structure.
//
// Simplified: for the CURRENT round with n evaluations (n is even),
// The partial sums at X=0 and X=1:
//   S(0) = sum of evals[i] for i in first half
//   S(1) = sum of evals[i] for i in second half
//   S(2) = sum of (2*evals[i+n/2] - evals[i]) for i in first half
//          (by linear interpolation: f(2) = 2*f(1) - f(0))
//
// This kernel computes per-thread partial sums; a reduction follows.
kernel void sumcheck_round_partial(
    device const Fr* evals          [[buffer(0)]],
    device Fr* partial_sums         [[buffer(1)]],  // 3 * num_groups entries
    constant uint& half_n           [[buffer(2)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    threadgroup Fr shared_s0[256];
    threadgroup Fr shared_s1[256];
    threadgroup Fr shared_s2[256];

    // Each thread accumulates over a strided range
    Fr local_s0 = fr_zero();
    Fr local_s1 = fr_zero();
    Fr local_s2 = fr_zero();

    uint base = tgid * tg_size;
    uint stride = tg_size;

    for (uint idx = base + tid; idx < half_n; idx += stride * 256) {
        if (idx < half_n) {
            Fr a = evals[idx];            // f(..., 0)
            Fr b = evals[idx + half_n];   // f(..., 1)
            local_s0 = fr_add(local_s0, a);
            local_s1 = fr_add(local_s1, b);
            // f(..., 2) = 2*b - a (linear extension)
            Fr two_b = fr_double(b);
            Fr f2 = fr_sub(two_b, a);
            local_s2 = fr_add(local_s2, f2);
        }
    }

    shared_s0[tid] = local_s0;
    shared_s1[tid] = local_s1;
    shared_s2[tid] = local_s2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint s = tg_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            shared_s0[tid] = fr_add(shared_s0[tid], shared_s0[tid + s]);
            shared_s1[tid] = fr_add(shared_s1[tid], shared_s1[tid + s]);
            shared_s2[tid] = fr_add(shared_s2[tid], shared_s2[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partial_sums[tgid * 3] = shared_s0[0];
        partial_sums[tgid * 3 + 1] = shared_s1[0];
        partial_sums[tgid * 3 + 2] = shared_s2[0];
    }
}

// SIMD-level Fr reduction using component-wise shuffle
inline Fr simd_reduce_fr(Fr val) {
    for (uint offset = 16; offset > 0; offset >>= 1) {
        Fr other;
        for (int k = 0; k < 8; k++) {
            other.v[k] = simd_shuffle_down(val.v[k], offset);
        }
        val = fr_add(val, other);
    }
    return val;
}

// Fused round-poly + reduce: reads data once, computes both partial sums and reduced output.
// Uses SIMD shuffle for intra-warp reduction (no barriers needed for first 5 levels).
kernel void sumcheck_round_reduce_fused(
    device const Fr* evals          [[buffer(0)]],
    device Fr* evals_out            [[buffer(1)]],    // reduced output (half_n entries)
    device Fr* partial_sums         [[buffer(2)]],    // 3 * num_groups entries
    constant Fr* challenge          [[buffer(3)]],    // single Fr element (constant broadcast)
    constant uint& half_n           [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    // Only need 8 entries per component for inter-SIMD reduction (256/32 = 8 SIMD groups)
    threadgroup Fr shared_s0[8];
    threadgroup Fr shared_s1[8];
    threadgroup Fr shared_s2[8];

    Fr local_s0 = fr_zero();
    Fr local_s1 = fr_zero();
    Fr local_s2 = fr_zero();

    Fr r = challenge[0];
    uint base = tgid * tg_size + tid;

    // Process elements: compute partial sums AND reduce in one pass
    if (base < half_n) {
        Fr a = evals[base];
        Fr b = evals[base + half_n];

        local_s0 = a;
        local_s1 = b;
        Fr two_b = fr_double(b);
        local_s2 = fr_sub(two_b, a);

        Fr diff = fr_sub(b, a);
        Fr r_diff = fr_mul(r, diff);
        evals_out[base] = fr_add(a, r_diff);
    }

    // Phase 1: SIMD-level reduction (no barriers needed)
    local_s0 = simd_reduce_fr(local_s0);
    local_s1 = simd_reduce_fr(local_s1);
    local_s2 = simd_reduce_fr(local_s2);

    // Phase 2: SIMD lane 0 writes to shared memory
    if (simd_lane == 0) {
        shared_s0[simd_id] = local_s0;
        shared_s1[simd_id] = local_s1;
        shared_s2[simd_id] = local_s2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: First SIMD group reduces the 8 inter-SIMD partial sums
    uint num_simds = tg_size / 32;
    if (tid < num_simds) {
        local_s0 = shared_s0[tid];
        local_s1 = shared_s1[tid];
        local_s2 = shared_s2[tid];
    } else {
        local_s0 = fr_zero();
        local_s1 = fr_zero();
        local_s2 = fr_zero();
    }

    if (simd_id == 0) {
        local_s0 = simd_reduce_fr(local_s0);
        local_s1 = simd_reduce_fr(local_s1);
        local_s2 = simd_reduce_fr(local_s2);
    }

    if (tid == 0) {
        partial_sums[tgid * 3] = local_s0;
        partial_sums[tgid * 3 + 1] = local_s1;
        partial_sums[tgid * 3 + 2] = local_s2;
    }
}

// Fused 2-round strided: reads 4 elements at strides 0, quarter, half, half+quarter.
// Computes 2 consecutive MSB-first rounds in registers, writes 1 element.
// Halves dispatch count for large sizes.
kernel void sumcheck_fused2_strided(
    device const Fr* evals          [[buffer(0)]],
    device Fr* evals_out            [[buffer(1)]],    // output: quarter_n entries
    device Fr* partial_sums         [[buffer(2)]],    // 2 * num_groups * 3 entries
    constant Fr* challenges         [[buffer(3)]],    // 2 Fr challenges
    constant uint& quarter_n        [[buffer(4)]],    // n/4
    constant uint& num_groups       [[buffer(5)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    uint global_tid = tgid * tg_size + tid;
    uint half_n = quarter_n * 2;

    threadgroup Fr shared_s0[8];
    threadgroup Fr shared_s1[8];
    threadgroup Fr shared_s2[8];

    uint num_simds = tg_size / 32;

    Fr v0 = fr_zero(), v1 = fr_zero();

    // Round 1: pairs at stride half_n
    {
        Fr s0 = fr_zero(), s1 = fr_zero(), s2 = fr_zero();
        Fr r = challenges[0];
        if (global_tid < quarter_n) {
            // Read 4 elements: i, i+quarter, i+half, i+half+quarter
            Fr a0 = evals[global_tid];
            Fr a1 = evals[global_tid + quarter_n];
            Fr a2 = evals[global_tid + half_n];
            Fr a3 = evals[global_tid + quarter_n + half_n];

            // Round 1 partial sums: a in first half, b in second half
            // S(0) = sum of first half = a0 + a1 (for this thread's contribution)
            s0 = fr_add(a0, a1);
            // S(1) = sum of second half = a2 + a3
            s1 = fr_add(a2, a3);
            // S(2) = 2*S(1) - S(0)
            s2 = fr_sub(fr_double(s1), s0);

            // Reduce: v0 = a0 + r*(a2-a0), v1 = a1 + r*(a3-a1)
            v0 = fr_add(a0, fr_mul(r, fr_sub(a2, a0)));
            v1 = fr_add(a1, fr_mul(r, fr_sub(a3, a1)));
        }
        s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2);
        if (simd_lane == 0) { shared_s0[simd_id] = s0; shared_s1[simd_id] = s1; shared_s2[simd_id] = s2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s0 = (tid < num_simds) ? shared_s0[tid] : fr_zero();
        s1 = (tid < num_simds) ? shared_s1[tid] : fr_zero();
        s2 = (tid < num_simds) ? shared_s2[tid] : fr_zero();
        if (simd_id == 0) { s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2); }
        if (tid == 0) {
            uint idx = (0 * num_groups + tgid) * 3;
            partial_sums[idx] = s0; partial_sums[idx+1] = s1; partial_sums[idx+2] = s2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Round 2: pairs at stride quarter_n (v0, v1 are adjacent in the reduced array)
    {
        Fr s0 = fr_zero(), s1 = fr_zero(), s2 = fr_zero();
        Fr r = challenges[1];
        if (global_tid < quarter_n) {
            s0 = v0;
            s1 = v1;
            s2 = fr_sub(fr_double(v1), v0);
            evals_out[global_tid] = fr_add(v0, fr_mul(r, fr_sub(v1, v0)));
        }
        s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2);
        if (simd_lane == 0) { shared_s0[simd_id] = s0; shared_s1[simd_id] = s1; shared_s2[simd_id] = s2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s0 = (tid < num_simds) ? shared_s0[tid] : fr_zero();
        s1 = (tid < num_simds) ? shared_s1[tid] : fr_zero();
        s2 = (tid < num_simds) ? shared_s2[tid] : fr_zero();
        if (simd_id == 0) { s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2); }
        if (tid == 0) {
            uint idx = (1 * num_groups + tgid) * 3;
            partial_sums[idx] = s0; partial_sums[idx+1] = s1; partial_sums[idx+2] = s2;
        }
    }
}

// Coalesced fused round-poly + reduce: reads ADJACENT pairs instead of stride-n/2.
// Processes LSB variable (elements at [2i, 2i+1]) for coalesced memory access.
// Output is compacted: evals_out[base] for each pair.
// Used when input is in "packed" layout (LSB variable determines adjacency).
kernel void sumcheck_round_reduce_coalesced(
    device const Fr* evals          [[buffer(0)]],
    device Fr* evals_out            [[buffer(1)]],    // reduced output (half_n entries)
    device Fr* partial_sums         [[buffer(2)]],    // 3 * num_groups entries
    constant Fr* challenge          [[buffer(3)]],
    constant uint& half_n           [[buffer(4)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    threadgroup Fr shared_s0[8];
    threadgroup Fr shared_s1[8];
    threadgroup Fr shared_s2[8];

    Fr local_s0 = fr_zero();
    Fr local_s1 = fr_zero();
    Fr local_s2 = fr_zero();

    Fr r = challenge[0];
    uint base = tgid * tg_size + tid;

    if (base < half_n) {
        // Read adjacent pair — fully coalesced across warp
        Fr a = evals[2 * base];
        Fr b = evals[2 * base + 1];

        local_s0 = a;
        local_s1 = b;
        Fr two_b = fr_double(b);
        local_s2 = fr_sub(two_b, a);

        Fr diff = fr_sub(b, a);
        Fr r_diff = fr_mul(r, diff);
        evals_out[base] = fr_add(a, r_diff);
    }

    local_s0 = simd_reduce_fr(local_s0);
    local_s1 = simd_reduce_fr(local_s1);
    local_s2 = simd_reduce_fr(local_s2);

    if (simd_lane == 0) {
        shared_s0[simd_id] = local_s0;
        shared_s1[simd_id] = local_s1;
        shared_s2[simd_id] = local_s2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = tg_size / 32;
    if (tid < num_simds) {
        local_s0 = shared_s0[tid];
        local_s1 = shared_s1[tid];
        local_s2 = shared_s2[tid];
    } else {
        local_s0 = fr_zero();
        local_s1 = fr_zero();
        local_s2 = fr_zero();
    }

    if (simd_id == 0) {
        local_s0 = simd_reduce_fr(local_s0);
        local_s1 = simd_reduce_fr(local_s1);
        local_s2 = simd_reduce_fr(local_s2);
    }

    if (tid == 0) {
        partial_sums[tgid * 3] = local_s0;
        partial_sums[tgid * 3 + 1] = local_s1;
        partial_sums[tgid * 3 + 2] = local_s2;
    }
}

// Fused multi-round sumcheck: loads 2^K elements into threadgroup memory,
// performs K rounds of reduce + partial-sum computation entirely in shared memory.
// Eliminates K-1 global memory barriers. Designed for K <= 8 (256 elements, 8KB data).
// Layout: shared[0..255] = data, shared[256..383] = s0, shared[384..511] = s1, shared[512..639] = s2
// Total: 640 * 32 = 20KB < 32KB threadgroup limit.
// tg_size must be 128, chunk_size = 256.
kernel void sumcheck_fused_multiround(
    device const Fr* evals          [[buffer(0)]],    // input: current_n elements
    device Fr* evals_out            [[buffer(1)]],    // output: current_n >> num_rounds elements
    device Fr* all_partial_sums     [[buffer(2)]],    // output: num_rounds * num_tgroups * 3 entries
    constant Fr* challenges         [[buffer(3)]],    // num_rounds challenges (constant broadcast)
    constant uint& current_n        [[buffer(4)]],    // current domain size
    constant uint& num_rounds       [[buffer(5)]],    // rounds to process (<= 8)
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    uint chunk_size = 256;
    uint base = tgid * chunk_size;
    uint num_tgroups = current_n / chunk_size;

    // Shared layout: data[0..255], reduce_s0[256..383], reduce_s1[384..511], reduce_s2[512..639]
    threadgroup Fr shared_data[256];
    threadgroup Fr reduce_s0[128];
    threadgroup Fr reduce_s1[128];
    threadgroup Fr reduce_s2[128];

    // Load chunk into shared memory
    if (base + tid < current_n)
        shared_data[tid] = evals[base + tid];
    if (base + tid + tg_size < current_n)
        shared_data[tid + tg_size] = evals[base + tid + tg_size];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint active_n = chunk_size;

    for (uint round = 0; round < num_rounds; round++) {
        uint half_n = active_n >> 1;
        Fr r = challenges[round];

        Fr local_s0 = fr_zero();
        Fr local_s1 = fr_zero();
        Fr local_s2 = fr_zero();

        if (tid < half_n) {
            Fr a = shared_data[tid];
            Fr b = shared_data[tid + half_n];

            local_s0 = a;
            local_s1 = b;
            Fr two_b = fr_double(b);
            local_s2 = fr_sub(two_b, a);

            // Reduce: write back to lower half_n
            Fr diff = fr_sub(b, a);
            Fr r_diff = fr_mul(r, diff);
            shared_data[tid] = fr_add(a, r_diff);
        }

        // Tree reduction for partial sums
        reduce_s0[tid] = local_s0;
        reduce_s1[tid] = local_s1;
        reduce_s2[tid] = local_s2;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = half_n >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                reduce_s0[tid] = fr_add(reduce_s0[tid], reduce_s0[tid + s]);
                reduce_s1[tid] = fr_add(reduce_s1[tid], reduce_s1[tid + s]);
                reduce_s2[tid] = fr_add(reduce_s2[tid], reduce_s2[tid + s]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Thread 0 writes this round's partial sums
        if (tid == 0) {
            uint ps_idx = (round * num_tgroups + tgid) * 3;
            all_partial_sums[ps_idx]     = reduce_s0[0];
            all_partial_sums[ps_idx + 1] = reduce_s1[0];
            all_partial_sums[ps_idx + 2] = reduce_s2[0];
        }

        active_n = half_n;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write final reduced elements
    uint final_n = chunk_size >> num_rounds;
    uint out_base = tgid * final_n;
    if (tid < final_n) {
        evals_out[out_base + tid] = shared_data[tid];
    }
}

// Fused 4-round coalesced sumcheck: reads 16 ADJACENT elements per thread,
// performs 4 rounds of reduce + partial-sum in registers, writes 1 element.
// Adjacent-pair access (LSB variable first) gives perfect memory coalescing.
// Cuts dispatches by 4x and eliminates strided access patterns.
kernel void sumcheck_fused4_coalesced(
    device const Fr* evals          [[buffer(0)]],
    device Fr* evals_out            [[buffer(1)]],
    device Fr* all_partial_sums     [[buffer(2)]],    // 4 * num_groups * 3 entries
    constant Fr* challenges         [[buffer(3)]],    // 4 Fr challenges
    constant uint& total_n          [[buffer(4)]],    // current domain size
    constant uint& num_groups       [[buffer(5)]],    // number of threadgroups
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    uint output_n = total_n >> 4;
    uint global_tid = tgid * tg_size + tid;

    // Shared memory for inter-SIMD reduction (reused across rounds)
    threadgroup Fr shared_s0[8];
    threadgroup Fr shared_s1[8];
    threadgroup Fr shared_s2[8];

    Fr d[16];
    if (global_tid < output_n) {
        for (uint k = 0; k < 16; k++)
            d[k] = evals[global_tid * 16 + k];
    }

    uint num_simds = tg_size / 32;

    // Round 1: 16 → 8 elements
    {
        Fr s0 = fr_zero(), s1 = fr_zero(), s2 = fr_zero();
        Fr r = challenges[0];
        if (global_tid < output_n) {
            for (uint k = 0; k < 8; k++) {
                Fr a = d[2*k], b = d[2*k+1];
                s0 = fr_add(s0, a);
                s1 = fr_add(s1, b);
                s2 = fr_add(s2, fr_sub(fr_double(b), a));
                d[k] = fr_add(a, fr_mul(r, fr_sub(b, a)));
            }
        }
        s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2);
        if (simd_lane == 0) { shared_s0[simd_id] = s0; shared_s1[simd_id] = s1; shared_s2[simd_id] = s2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s0 = (tid < num_simds) ? shared_s0[tid] : fr_zero();
        s1 = (tid < num_simds) ? shared_s1[tid] : fr_zero();
        s2 = (tid < num_simds) ? shared_s2[tid] : fr_zero();
        if (simd_id == 0) { s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2); }
        if (tid == 0) {
            uint idx = (0 * num_groups + tgid) * 3;
            all_partial_sums[idx] = s0; all_partial_sums[idx+1] = s1; all_partial_sums[idx+2] = s2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Round 2: 8 → 4 elements
    {
        Fr s0 = fr_zero(), s1 = fr_zero(), s2 = fr_zero();
        Fr r = challenges[1];
        if (global_tid < output_n) {
            for (uint k = 0; k < 4; k++) {
                Fr a = d[2*k], b = d[2*k+1];
                s0 = fr_add(s0, a);
                s1 = fr_add(s1, b);
                s2 = fr_add(s2, fr_sub(fr_double(b), a));
                d[k] = fr_add(a, fr_mul(r, fr_sub(b, a)));
            }
        }
        s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2);
        if (simd_lane == 0) { shared_s0[simd_id] = s0; shared_s1[simd_id] = s1; shared_s2[simd_id] = s2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s0 = (tid < num_simds) ? shared_s0[tid] : fr_zero();
        s1 = (tid < num_simds) ? shared_s1[tid] : fr_zero();
        s2 = (tid < num_simds) ? shared_s2[tid] : fr_zero();
        if (simd_id == 0) { s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2); }
        if (tid == 0) {
            uint idx = (1 * num_groups + tgid) * 3;
            all_partial_sums[idx] = s0; all_partial_sums[idx+1] = s1; all_partial_sums[idx+2] = s2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Round 3: 4 → 2 elements
    {
        Fr s0 = fr_zero(), s1 = fr_zero(), s2 = fr_zero();
        Fr r = challenges[2];
        if (global_tid < output_n) {
            for (uint k = 0; k < 2; k++) {
                Fr a = d[2*k], b = d[2*k+1];
                s0 = fr_add(s0, a);
                s1 = fr_add(s1, b);
                s2 = fr_add(s2, fr_sub(fr_double(b), a));
                d[k] = fr_add(a, fr_mul(r, fr_sub(b, a)));
            }
        }
        s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2);
        if (simd_lane == 0) { shared_s0[simd_id] = s0; shared_s1[simd_id] = s1; shared_s2[simd_id] = s2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s0 = (tid < num_simds) ? shared_s0[tid] : fr_zero();
        s1 = (tid < num_simds) ? shared_s1[tid] : fr_zero();
        s2 = (tid < num_simds) ? shared_s2[tid] : fr_zero();
        if (simd_id == 0) { s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2); }
        if (tid == 0) {
            uint idx = (2 * num_groups + tgid) * 3;
            all_partial_sums[idx] = s0; all_partial_sums[idx+1] = s1; all_partial_sums[idx+2] = s2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Round 4: 2 → 1 element
    {
        Fr s0 = fr_zero(), s1 = fr_zero(), s2 = fr_zero();
        Fr r = challenges[3];
        if (global_tid < output_n) {
            Fr a = d[0], b = d[1];
            s0 = a; s1 = b;
            s2 = fr_sub(fr_double(b), a);
            evals_out[global_tid] = fr_add(a, fr_mul(r, fr_sub(b, a)));
        }
        s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2);
        if (simd_lane == 0) { shared_s0[simd_id] = s0; shared_s1[simd_id] = s1; shared_s2[simd_id] = s2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s0 = (tid < num_simds) ? shared_s0[tid] : fr_zero();
        s1 = (tid < num_simds) ? shared_s1[tid] : fr_zero();
        s2 = (tid < num_simds) ? shared_s2[tid] : fr_zero();
        if (simd_id == 0) { s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2); }
        if (tid == 0) {
            uint idx = (3 * num_groups + tgid) * 3;
            all_partial_sums[idx] = s0; all_partial_sums[idx+1] = s1; all_partial_sums[idx+2] = s2;
        }
    }
}

// Fused 2-round coalesced sumcheck: reads 4 adjacent elements, 2 rounds, writes 1.
kernel void sumcheck_fused2_coalesced(
    device const Fr* evals          [[buffer(0)]],
    device Fr* evals_out            [[buffer(1)]],
    device Fr* all_partial_sums     [[buffer(2)]],    // 2 * num_groups * 3 entries
    constant Fr* challenges         [[buffer(3)]],    // 2 Fr challenges
    constant uint& total_n          [[buffer(4)]],
    constant uint& num_groups       [[buffer(5)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tgid                       [[threadgroup_position_in_grid]],
    uint tg_size                    [[threads_per_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_id                    [[simdgroup_index_in_threadgroup]]
) {
    uint output_n = total_n >> 2;
    uint global_tid = tgid * tg_size + tid;

    threadgroup Fr shared_s0[8];
    threadgroup Fr shared_s1[8];
    threadgroup Fr shared_s2[8];

    uint num_simds = tg_size / 32;

    // Round 1: pair (a0,a1) and (a2,a3)
    Fr v0 = fr_zero(), v1 = fr_zero();
    {
        Fr s0 = fr_zero(), s1 = fr_zero(), s2 = fr_zero();
        Fr r = challenges[0];
        if (global_tid < output_n) {
            Fr a0 = evals[global_tid * 4];
            Fr a1 = evals[global_tid * 4 + 1];
            Fr a2 = evals[global_tid * 4 + 2];
            Fr a3 = evals[global_tid * 4 + 3];
            s0 = fr_add(a0, a2);
            s1 = fr_add(a1, a3);
            s2 = fr_add(fr_sub(fr_double(a1), a0), fr_sub(fr_double(a3), a2));
            v0 = fr_add(a0, fr_mul(r, fr_sub(a1, a0)));
            v1 = fr_add(a2, fr_mul(r, fr_sub(a3, a2)));
        }
        s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2);
        if (simd_lane == 0) { shared_s0[simd_id] = s0; shared_s1[simd_id] = s1; shared_s2[simd_id] = s2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s0 = (tid < num_simds) ? shared_s0[tid] : fr_zero();
        s1 = (tid < num_simds) ? shared_s1[tid] : fr_zero();
        s2 = (tid < num_simds) ? shared_s2[tid] : fr_zero();
        if (simd_id == 0) { s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2); }
        if (tid == 0) {
            uint idx = (0 * num_groups + tgid) * 3;
            all_partial_sums[idx] = s0; all_partial_sums[idx+1] = s1; all_partial_sums[idx+2] = s2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Round 2: pair (v0, v1)
    {
        Fr s0 = fr_zero(), s1 = fr_zero(), s2 = fr_zero();
        Fr r = challenges[1];
        if (global_tid < output_n) {
            s0 = v0; s1 = v1;
            s2 = fr_sub(fr_double(v1), v0);
            evals_out[global_tid] = fr_add(v0, fr_mul(r, fr_sub(v1, v0)));
        }
        s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2);
        if (simd_lane == 0) { shared_s0[simd_id] = s0; shared_s1[simd_id] = s1; shared_s2[simd_id] = s2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s0 = (tid < num_simds) ? shared_s0[tid] : fr_zero();
        s1 = (tid < num_simds) ? shared_s1[tid] : fr_zero();
        s2 = (tid < num_simds) ? shared_s2[tid] : fr_zero();
        if (simd_id == 0) { s0 = simd_reduce_fr(s0); s1 = simd_reduce_fr(s1); s2 = simd_reduce_fr(s2); }
        if (tid == 0) {
            uint idx = (1 * num_groups + tgid) * 3;
            all_partial_sums[idx] = s0; all_partial_sums[idx+1] = s1; all_partial_sums[idx+2] = s2;
        }
    }
}

// GPU-side final reduction of partial sums: reduce numGroups partial sums to 3 final values.
kernel void sumcheck_partial_final_reduce(
    device const Fr* partial_sums   [[buffer(0)]],    // 3 * num_groups entries
    device Fr* output               [[buffer(1)]],    // 3 entries: S(0), S(1), S(2)
    constant uint& num_groups       [[buffer(2)]],
    uint tid                        [[thread_index_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]]
) {
    threadgroup Fr shared_s0[256];
    threadgroup Fr shared_s1[256];
    threadgroup Fr shared_s2[256];

    Fr local_s0 = fr_zero();
    Fr local_s1 = fr_zero();
    Fr local_s2 = fr_zero();

    // Each thread accumulates over strided range
    for (uint g = tid; g < num_groups; g += tg_size) {
        local_s0 = fr_add(local_s0, partial_sums[g * 3]);
        local_s1 = fr_add(local_s1, partial_sums[g * 3 + 1]);
        local_s2 = fr_add(local_s2, partial_sums[g * 3 + 2]);
    }

    shared_s0[tid] = local_s0;
    shared_s1[tid] = local_s1;
    shared_s2[tid] = local_s2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tg_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            shared_s0[tid] = fr_add(shared_s0[tid], shared_s0[tid + s]);
            shared_s1[tid] = fr_add(shared_s1[tid], shared_s1[tid + s]);
            shared_s2[tid] = fr_add(shared_s2[tid], shared_s2[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[0] = shared_s0[0];
        output[1] = shared_s1[0];
        output[2] = shared_s2[0];
    }
}
