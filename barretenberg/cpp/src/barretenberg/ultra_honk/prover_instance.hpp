// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Sergei], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once
#include "barretenberg/common/assert.hpp"
#include "barretenberg/ext/starknet/flavor/ultra_starknet_flavor.hpp"
#include "barretenberg/ext/starknet/flavor/ultra_starknet_zk_flavor.hpp"
#include "barretenberg/flavor/flavor.hpp"
#include "barretenberg/flavor/mega_zk_flavor.hpp"
#include "barretenberg/flavor/ultra_keccak_flavor.hpp"
#include "barretenberg/flavor/ultra_keccak_zk_flavor.hpp"
#include "barretenberg/flavor/ultra_zk_flavor.hpp"
#include "barretenberg/polynomials/polynomial_stats.hpp"
#include "barretenberg/relations/relation_parameters.hpp"
#include "barretenberg/sumcheck/masking_tail_data.hpp"

namespace bb {

/**
 * @brief Cached precomputed polynomial data for a specific circuit, enabling fast re-proving with different witnesses.
 * @details Stores shared (zero-copy) references to the 28 precomputed polynomials (selectors, sigmas, IDs, tables,
 * lagrange) plus circuit metadata. Created via ProverInstance_::create_cache(), then passed to the ProverInstance
 * cached constructor for subsequent proofs with the same circuit bytecode.
 */
template <typename Flavor> struct PrecomputedCache {
    using Polynomial = typename Flavor::Polynomial;

    std::vector<Polynomial> precomputed_polys; // shared refs to the 28 precomputed polynomials
    size_t dyadic_size = 0;
    size_t final_active_wire_idx = 0;
};

/**
 * @brief Contains all the information required by a Honk prover to create a proof, constructed from a finalized
 * circuit.
 */

template <typename Flavor_> class ProverInstance_ {
  public:
    using Flavor = Flavor_;
    using FF = typename Flavor::FF;

  private:
    using Circuit = typename Flavor::CircuitBuilder;
    using ProverPolynomials = typename Flavor::ProverPolynomials;
    using WitnessCommitments = typename Flavor::WitnessCommitments;
    using Polynomial = typename Flavor::Polynomial;
    MetaData metadata; // circuit size and public inputs metadata
    // index of the last constrained wire in the execution trace; initialize to size_t::max to indicate uninitialized
    size_t final_active_wire_idx{ std::numeric_limits<size_t>::max() };

  public:
    std::vector<FF> public_inputs;
    ProverPolynomials polynomials; // the multilinear polynomials used by the prover
    WitnessCommitments commitments;
    FF alpha; // challenge whose powers batch subrelation contributions during Sumcheck
    RelationParameters<FF> relation_parameters;
    std::vector<FF> gate_challenges;

    MaskingTailData<Flavor> masking_tail_data; // ZK: stores masking values for short witness polys

    HonkProof ipa_proof; // utilized for rollup proofs (IO::HasIPA)

    std::vector<uint32_t> memory_read_records;
    std::vector<uint32_t> memory_write_records;

    size_t dyadic_size() const { return metadata.dyadic_size; }
    size_t log_dyadic_size() const { return numeric::get_msb(dyadic_size()); }
    size_t pub_inputs_offset() const { return metadata.pub_inputs_offset; }
    size_t num_public_inputs() const
    {
        BB_ASSERT_EQ(metadata.num_public_inputs, public_inputs.size());
        return metadata.num_public_inputs;
    }
    size_t get_final_active_wire_idx() const
    {
        BB_ASSERT(final_active_wire_idx != std::numeric_limits<size_t>::max(),
                  "final_active_wire_idx has not been initialized");
        return final_active_wire_idx;
    }

    Flavor::PrecomputedData get_precomputed()
    {
        return typename Flavor::PrecomputedData{ polynomials.get_precomputed(), metadata };
    }

    ProverInstance_(Circuit& circuit);

    /**
     * @brief Construct a ProverInstance using cached precomputed polynomials.
     * @details Skips selector population, copy cycle computation, permutation polynomial computation, and table
     * construction. Only populates witness-dependent data (wires, lookup read counts, memory records, public inputs).
     */
    ProverInstance_(Circuit& circuit, const PrecomputedCache<Flavor>& cache);

    /**
     * @brief Create a PrecomputedCache from this instance for reuse with different witnesses.
     */
    PrecomputedCache<Flavor> create_cache();

    ProverInstance_() = default;
    ProverInstance_(const ProverInstance_&) = delete;
    ProverInstance_(ProverInstance_&&) = delete;
    ProverInstance_& operator=(const ProverInstance_&) = delete;
    ProverInstance_& operator=(ProverInstance_&&) = delete;
    ~ProverInstance_() = default;

  private:
    /** @brief Get the size of the active trace range (0 to the final active wire index) */
    size_t trace_active_range_size() const { return get_final_active_wire_idx() + 1; }

    size_t compute_dyadic_size(Circuit&);

    void allocate_wires();

    void allocate_permutation_argument_polynomials();

    void allocate_lagrange_polynomials();

    void allocate_selectors(const Circuit&);

    void allocate_table_lookup_polynomials(const Circuit&);

    void allocate_ecc_op_polynomials(const Circuit&)
        requires IsMegaFlavor<Flavor>;

    void allocate_databus_polynomials(const Circuit&)
        requires HasDataBus<Flavor>;

    void construct_databus_polynomials(Circuit&)
        requires HasDataBus<Flavor>;

    void construct_lookup_polynomials(Circuit& circuit);

    void populate_memory_records(const Circuit& circuit);
};

} // namespace bb
