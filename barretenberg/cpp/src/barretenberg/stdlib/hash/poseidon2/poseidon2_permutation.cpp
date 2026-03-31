// === AUDIT STATUS ===
// internal:    { status: Complete, auditors: [Sergei], commit: 777717f6af324188ecd6bb68c3c86ee7befef94d}
// external_1:  { status: Complete, auditors: [@ed25519 (Spearbit)], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#include "poseidon2_permutation.hpp"

#include "barretenberg/honk/execution_trace/gate_data.hpp"

namespace bb::stdlib {

template <typename Builder>
typename Poseidon2Permutation<Builder>::State Poseidon2Permutation<Builder>::permutation(
    Builder* builder, const typename Poseidon2Permutation<Builder>::State& input)
{
    State current_state(input);
    NativeState current_native_state;
    for (size_t i = 0; i < t; ++i) {
        current_native_state[i] = current_state[i].get_value();
    }

    // Apply 1st linear layer both natively and in-circuit.
    NativePermutation::matrix_multiplication_external(current_native_state);
    matrix_multiplication_external(current_state);

    // First set of external rounds
    constexpr size_t rounds_f_beginning = rounds_f / 2;
    for (size_t i = 0; i < rounds_f_beginning; ++i) {
        poseidon2_external_gate_<FF> in{ current_state[0].get_witness_index(),
                                         current_state[1].get_witness_index(),
                                         current_state[2].get_witness_index(),
                                         current_state[3].get_witness_index(),
                                         i };
        builder->create_poseidon2_external_gate(in);
        // calculate the new witnesses
        NativePermutation::add_round_constants(current_native_state, round_constants[i]);
        NativePermutation::apply_sbox(current_native_state);
        NativePermutation::matrix_multiplication_external(current_native_state);
        for (size_t j = 0; j < t; ++j) {
            current_state[j] = witness_t<Builder>(builder, current_native_state[j]);
        }
    }

    propagate_current_state_to_next_row(builder, current_state, builder->blocks.poseidon2_external);

    // Internal rounds — batch-optimized: pre-compute all native states, then bulk-create witnesses and gates.
    constexpr size_t p_end = rounds_f_beginning + rounds_p;
    {
        // Phase 1: Pre-compute all native round outputs in a tight loop (pure field arithmetic, no allocations).
        std::array<NativeState, rounds_p> native_outputs;
        {
            NativeState s = current_native_state;
            for (size_t i = 0; i < rounds_p; ++i) {
                s[0] += round_constants[rounds_f_beginning + i][0];
                NativePermutation::apply_single_sbox(s[0]);
                NativePermutation::matrix_multiplication_internal(s);
                native_outputs[i] = s;
            }
            current_native_state = s;
        }

        // Phase 2: Batch-create all witness variables at once (replaces rounds_p * 4 individual add_variable calls).
        std::array<FF, rounds_p * t> batch_values;
        for (size_t i = 0; i < rounds_p; ++i) {
            for (size_t j = 0; j < t; ++j) {
                batch_values[i * t + j] = native_outputs[i][j];
            }
        }
        const uint32_t base_idx = builder->add_variables_batch(batch_values.data(), rounds_p * t);

        // Phase 3: Build wire index array and round index array for batch gate creation.
        // Layout: [count round gates][1 propagate gate], each gate has 4 wire indices.
        std::array<uint32_t, (rounds_p + 1) * t> wire_indices;
        std::array<size_t, rounds_p> round_indices_arr;

        // Gate 0: wires from external round output
        wire_indices[0] = current_state[0].get_witness_index();
        wire_indices[1] = current_state[1].get_witness_index();
        wire_indices[2] = current_state[2].get_witness_index();
        wire_indices[3] = current_state[3].get_witness_index();
        round_indices_arr[0] = rounds_f_beginning;

        // Gates 1..rounds_p-1: wires from batch-created variables
        for (size_t i = 1; i < rounds_p; ++i) {
            const uint32_t offset = base_idx + static_cast<uint32_t>((i - 1) * t);
            wire_indices[i * t + 0] = offset;
            wire_indices[i * t + 1] = offset + 1;
            wire_indices[i * t + 2] = offset + 2;
            wire_indices[i * t + 3] = offset + 3;
            round_indices_arr[i] = rounds_f_beginning + i;
        }

        // Propagate gate: wires from last batch output
        const uint32_t last_offset = base_idx + static_cast<uint32_t>((rounds_p - 1) * t);
        wire_indices[rounds_p * t + 0] = last_offset;
        wire_indices[rounds_p * t + 1] = last_offset + 1;
        wire_indices[rounds_p * t + 2] = last_offset + 2;
        wire_indices[rounds_p * t + 3] = last_offset + 3;

        // Batch-create all internal gates + propagate gate
        builder->create_poseidon2_internal_gates_batch(
            wire_indices.data(), round_indices_arr.data(), rounds_p);

        // Update current_state to reference the last batch outputs
        for (size_t j = 0; j < t; ++j) {
            current_state[j] = field_t<Builder>::from_witness_index(builder, last_offset + static_cast<uint32_t>(j));
        }
    }

    // Remaining external rounds
    for (size_t i = p_end; i < NUM_ROUNDS; ++i) {
        poseidon2_external_gate_<FF> in{ current_state[0].get_witness_index(),
                                         current_state[1].get_witness_index(),
                                         current_state[2].get_witness_index(),
                                         current_state[3].get_witness_index(),
                                         i };
        builder->create_poseidon2_external_gate(in);
        // calculate the new witnesses
        NativePermutation::add_round_constants(current_native_state, round_constants[i]);
        NativePermutation::apply_sbox(current_native_state);
        NativePermutation::matrix_multiplication_external(current_native_state);
        for (size_t j = 0; j < t; ++j) {
            current_state[j] = witness_t<Builder>(builder, current_native_state[j]);
        }
    }

    propagate_current_state_to_next_row(builder, current_state, builder->blocks.poseidon2_external);

    return current_state;
}

/**
 * @brief Separate function to do just the first linear layer (equivalent to external matrix mul).
 * @details Update the state with \f$ M_E \cdot (\text{state}[0], \text{state}[1], \text{state}[2],
 * \text{state}[3])^{\top}\f$. Where \f$ M_E \f$ is the external round matrix. See `Poseidon2ExternalRelationImpl`.
 */
template <typename Builder>
void Poseidon2Permutation<Builder>::matrix_multiplication_external(typename Poseidon2Permutation<Builder>::State& state)
{
    const bb::fr two(2);
    const bb::fr four(4);
    // create the 6 gates for the initial matrix multiplication
    // gate 1: Compute tmp1 = state[0] + state[1] + 2 * state[3]
    field_t<Builder> tmp1 = state[0].add_two(state[1], state[3] * two);

    // gate 2: Compute tmp2 = 2 * state[1] + state[2] + state[3]
    field_t<Builder> tmp2 = state[2].add_two(state[1] * two, state[3]);

    // gate 3: Compute v2 = 4 * state[0] + 4 * state[1] + tmp2
    state[1] = tmp2.add_two(state[0] * four, state[1] * four);

    // gate 4: Compute v1 = v2 + tmp1
    state[0] = state[1] + tmp1;

    // gate 5: Compute v4 = tmp1 + 4 * state[2] + 4 * state[3]
    state[3] = tmp1.add_two(state[2] * four, state[3] * four);

    // gate 6: Compute v3 = v4 + tmp2
    state[2] = state[3] + tmp2;
}

template class Poseidon2Permutation<MegaCircuitBuilder>;
template class Poseidon2Permutation<UltraCircuitBuilder>;

} // namespace bb::stdlib
