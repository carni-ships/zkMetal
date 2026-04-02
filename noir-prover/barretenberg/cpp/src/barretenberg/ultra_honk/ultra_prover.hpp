// === AUDIT STATUS ===
// internal:    { status: not started, auditors: [], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once
#include "barretenberg/commitment_schemes/small_subgroup_ipa/small_subgroup_ipa.hpp"
#include "barretenberg/flavor/mega_flavor.hpp"
#include "barretenberg/flavor/ultra_flavor.hpp"
#include "barretenberg/honk/proof_system/types/proof.hpp"
#include "barretenberg/relations/relation_parameters.hpp"
#include "barretenberg/sumcheck/sumcheck_output.hpp"
#include "barretenberg/transcript/transcript.hpp"
#include "barretenberg/ultra_honk/prover_instance.hpp"

namespace bb {

template <typename Flavor_> class UltraProver_ {
  public:
    using Flavor = Flavor_;
    using FF = typename Flavor::FF;
    using Builder = typename Flavor::CircuitBuilder;
    using Commitment = typename Flavor::Commitment;
    using CommitmentKey = bb::CommitmentKey<typename Flavor::Curve>;
    using Curve = typename Flavor::Curve;
    using Polynomial = typename Flavor::Polynomial;
    using ProverPolynomials = typename Flavor::ProverPolynomials;
    using CommitmentLabels = typename Flavor::CommitmentLabels;
    using PCS = typename Flavor::PCS;
    using ProverInstance = ProverInstance_<Flavor>;
    using SmallSubgroupIPA = SmallSubgroupIPAProver<Flavor>;
    using HonkVK = typename Flavor::VerificationKey;
    using Transcript = typename Flavor::Transcript;
    using Proof = typename Transcript::Proof;
    using ZKData = ZKSumcheckData<Flavor>;

    std::shared_ptr<ProverInstance> prover_instance;
    std::shared_ptr<HonkVK> honk_vk;

    std::shared_ptr<Transcript> transcript;

    bb::RelationParameters<FF> relation_parameters;

    Polynomial quotient_W;

    SumcheckOutput<Flavor> sumcheck_output;

    size_t virtual_log_n = 0;

    // Streaming sumcheck state: when non-empty, polynomials were serialized to disk
    // during sumcheck and must be streamed during PCS.
    std::string streaming_temp_dir;
    std::vector<std::string> streaming_unshifted_paths;
    std::vector<std::string> streaming_shifted_paths;
    std::vector<std::string> streaming_all_poly_paths;

    CommitmentKey commitment_key;

    UltraProver_(std::shared_ptr<ProverInstance> prover_instance,
                 const std::shared_ptr<HonkVK>& honk_vk,
                 const std::shared_ptr<Transcript>& transcript = std::make_shared<Transcript>());

    BB_PROFILE void execute_sumcheck_iop();
    BB_PROFILE void execute_pcs();
    void construct_proof_low_memory();

    BB_PROFILE void generate_gate_challenges();

    Proof export_proof();
    Proof construct_proof();
    Proof prove() { return construct_proof(); };

    ZKData zk_sumcheck_data;
};

using UltraProver = UltraProver_<UltraFlavor>;
using UltraZKProver = UltraProver_<UltraZKFlavor>;
using UltraKeccakProver = UltraProver_<UltraKeccakFlavor>;
#ifdef STARKNET_GARAGA_FLAVORS
using UltraStarknetProver = UltraProver_<UltraStarknetFlavor>;
using UltraStarknetZKProver = UltraProver_<UltraStarknetZKFlavor>;
#endif
using UltraKeccakZKProver = UltraProver_<UltraKeccakZKFlavor>;
using MegaProver = UltraProver_<MegaFlavor>;
using MegaZKProver = UltraProver_<MegaZKFlavor>;

} // namespace bb
