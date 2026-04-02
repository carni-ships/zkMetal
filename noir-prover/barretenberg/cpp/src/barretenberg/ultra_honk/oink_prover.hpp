// === AUDIT STATUS ===
// internal:    { status: not started, auditors: [], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once
#include <utility>

#include "barretenberg/commitment_schemes/commitment_key.hpp"
#include "barretenberg/ultra_honk/prover_instance.hpp"

namespace bb {

/**
 * @brief Class for all the oink rounds, which are shared between the folding prover and ultra prover.
 * @details This class contains send_vk_hash_and_public_inputs(), commit_to_wires(),
 * commit_to_lookup_counts_and_w4(), commit_to_logderiv_inverses(), commit_to_z_perm(),
 * commit_to_logderiv_inverses_and_z_perm(), and commit_to_masking_poly().
 *
 * @tparam Flavor
 */
template <typename Flavor> class OinkProver {
  public:
    using CommitmentKey = bb::CommitmentKey<typename Flavor::Curve>;
    using HonkVK = typename Flavor::VerificationKey;
    using ProverInstance = ProverInstance_<Flavor>;
    using Transcript = typename Flavor::Transcript;
    using FF = typename Flavor::FF;
    using Proof = typename Transcript::Proof;

    std::shared_ptr<ProverInstance> prover_instance;
    std::shared_ptr<HonkVK> honk_vk;
    std::shared_ptr<Transcript> transcript;

    typename Flavor::CommitmentLabels commitment_labels;

    CommitmentKey commitment_key;

    OinkProver(std::shared_ptr<ProverInstance> prover_instance,
               std::shared_ptr<HonkVK> honk_vk,
               const std::shared_ptr<typename Flavor::Transcript>& transcript = std::make_shared<Transcript>())
        : prover_instance(prover_instance)
        , honk_vk(honk_vk)
        , transcript(transcript)
    {}

    void prove(bool emit_alpha = true);
    Proof export_proof();

    void send_vk_hash_and_public_inputs();
    void commit_to_masking_poly();
    void commit_to_wires();
    void commit_to_lookup_counts_and_w4();
    void commit_to_logderiv_inverses();
    void commit_to_z_perm();
    void commit_to_logderiv_inverses_and_z_perm();

    static void add_ram_rom_memory_records_to_wire_4(ProverInstance& instance);
    static void compute_logderivative_inverses(ProverInstance& instance);
    static void compute_grand_product_polynomial(ProverInstance& instance);
};

using MegaOinkProver = OinkProver<MegaFlavor>;

} // namespace bb
