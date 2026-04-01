// === AUDIT STATUS ===
// internal:    { status: Completed, auditors: [Sergei], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#include "ultra_prover.hpp"
#include "barretenberg/commitment_schemes/gemini/gemini.hpp"
#include "barretenberg/commitment_schemes/shplonk/shplemini.hpp"
#include "barretenberg/flavor/mega_avm_flavor.hpp"
#include "barretenberg/polynomials/backing_memory.hpp"
#include "barretenberg/sumcheck/sumcheck.hpp"
#include "barretenberg/ultra_honk/oink_prover.hpp"
#include <filesystem>
#include <set>
namespace bb {

template <typename Flavor>
UltraProver_<Flavor>::UltraProver_(std::shared_ptr<ProverInstance> prover_instance,
                                   const std::shared_ptr<HonkVK>& honk_vk,
                                   const std::shared_ptr<Transcript>& transcript)
    : prover_instance(std::move(prover_instance))
    , transcript(transcript)
    , honk_vk(honk_vk)
{}

/**
 * @brief Export the complete proof, including IPA proof for rollup circuits
 * @details Two-level proof structure for rollup circuits:
 *
 * **Prover Level (this function):**
 *   [public_inputs | honk_proof | ipa_proof]
 *   - Appends IPA proof if prover_instance->ipa_proof is non-empty
 *   - SYMMETRIC with UltraVerifier_::split_rollup_proof() which extracts the IPA portion
 *
 * **API Level (bbapi):**
 *   - _prove() further splits into: public_inputs (ACIR only) vs proof (rest including IPA)
 *   - concatenate_proof() reassembles for verification
 *
 * @note IPA_PROOF_LENGTH is defined in ipa.hpp as 4*CONST_ECCVM_LOG_N + 4 = 64 elements
 */
template <typename Flavor> typename UltraProver_<Flavor>::Proof UltraProver_<Flavor>::export_proof()
{
    auto proof = transcript->export_proof();

    // Append IPA proof if present
    if (!prover_instance->ipa_proof.empty()) {
        BB_ASSERT_EQ(prover_instance->ipa_proof.size(), static_cast<size_t>(IPA_PROOF_LENGTH));
        proof.insert(proof.end(), prover_instance->ipa_proof.begin(), prover_instance->ipa_proof.end());
    }

    return proof;
}

template <typename Flavor> void UltraProver_<Flavor>::generate_gate_challenges()
{
    virtual_log_n =
        Flavor::USE_PADDING ? Flavor::VIRTUAL_LOG_N : static_cast<size_t>(prover_instance->log_dyadic_size());

    prover_instance->gate_challenges =
        transcript->template get_dyadic_powers_of_challenge<FF>("Sumcheck:gate_challenge", virtual_log_n);
}

template <typename Flavor> typename UltraProver_<Flavor>::Proof UltraProver_<Flavor>::construct_proof()
{
    if (slow_low_memory) {
        // Option D: Progressive memory management during Oink.
        // Serialize precomputed polys to disk early, free them between Oink rounds
        // to reduce peak memory.
        construct_proof_low_memory();
    } else {
        // The CRS must accommodate the full dyadic_size: Shplonk quotient Q is sized at
        // max(claim.polynomial.size()) which includes the batched polynomial A_0 at dyadic_size.
        // Using only max_end_index would fail when max_end_index < dyadic_size (circuit doesn't
        // fill the dyadic domain) and no prior SRS load warmed the cache to the full size.
        size_t key_size = prover_instance->dyadic_size();
        if constexpr (Flavor::HasZK) {
            // SmallSubgroupIPA commits fixed-size polynomials (up to SUBGROUP_SIZE + 3).
            constexpr size_t log_subgroup_size = static_cast<size_t>(numeric::get_msb(Curve::SUBGROUP_SIZE));
            key_size = std::max(key_size, size_t{ 1 } << (log_subgroup_size + 1));
        }
        commitment_key = CommitmentKey(key_size);

        OinkProver<Flavor> oink_prover(prover_instance, honk_vk, transcript);
        oink_prover.prove();
        vinfo("created oink proof");
    }

    generate_gate_challenges();

    // Run sumcheck
    execute_sumcheck_iop();
    vinfo("finished relation check rounds");
    // Execute Shplemini PCS
    execute_pcs();
    vinfo("finished PCS rounds");

    return export_proof();
}

/**
 * @brief Low-memory Oink path: serialize precomputed polys to disk early, call Oink rounds
 * individually, and free precomputed polys between rounds as they become unneeded.
 *
 * @details Memory timeline at 2^24 (each poly = 512 MiB):
 *   1. Start: 41 polys = 20.5 GiB
 *   2. Serialize 28 precomputed to disk, free 11 unused by Oink: -5.5 GiB -> 15 GiB
 *   3. Init CommitmentKey (SRS shared via cache): +1 GiB -> 16 GiB
 *   4. Wire + sorted list rounds: no change
 *   5. After log-derivative: free q_lookup, table_1..4, q_m, q_c, q_r, q_o (-4.5 GiB) -> 11.5 GiB
 *   6. After grand product: free sigma_1..4, id_1..4 (-4 GiB) -> 7.5 GiB
 *   7. Free commitment key: -1 GiB -> 6.5 GiB (witness only)
 */
template <typename Flavor> void UltraProver_<Flavor>::construct_proof_low_memory()
{
    // Create streaming temp dir early -- will be reused by execute_sumcheck_iop() and execute_pcs()
    auto temp_dir = std::filesystem::temp_directory_path() / ("bb-streaming-" + std::to_string(getpid()));
    std::filesystem::create_directories(temp_dir);
    streaming_temp_dir = temp_dir.string();
    vinfo("low-memory mode: temp dir ", streaming_temp_dir);

    auto& polys = prover_instance->polynomials;
    auto free_poly = [](auto& p) { p = Polynomial(); };

    // Phase 1: Serialize PRECOMPUTED polys to disk before Oink (they're immutable).
    // Witness polys are serialized AFTER Oink because Oink modifies w_4, lookup_inverses, z_perm.
    // Must serialize in BOTH orderings: get_all() for sumcheck, get_unshifted() for PCS.
    auto all_polys = polys.get_all();
    streaming_all_poly_paths.resize(all_polys.size());

    // Build set of precomputed poly data pointers to identify precomputed indices
    auto precomputed = polys.get_precomputed();
    std::set<const void*> precomputed_ptrs;
    for (auto& p : precomputed) {
        if (!p.is_empty()) {
            precomputed_ptrs.insert(static_cast<const void*>(p.data()));
        }
    }
    auto is_precomputed = [&](const auto& p) {
        return !p.is_empty() && precomputed_ptrs.count(static_cast<const void*>(p.data()));
    };

    // Serialize precomputed in get_all() ordering (for streaming sumcheck)
    size_t num_precomputed_serialized = 0;
    for (size_t i = 0; i < all_polys.size(); i++) {
        if (is_precomputed(all_polys[i])) {
            std::string path = streaming_temp_dir + "/all_" + std::to_string(i) + ".bin";
            all_polys[i].serialize_to_file(path);
            streaming_all_poly_paths[i] = path;
            num_precomputed_serialized++;
        }
    }

    // Serialize precomputed in get_unshifted() ordering (for PCS)
    auto unshifted = polys.get_unshifted();
    streaming_unshifted_paths.resize(unshifted.size());
    for (size_t i = 0; i < unshifted.size(); i++) {
        if (is_precomputed(unshifted[i])) {
            std::string path = streaming_temp_dir + "/unshifted_" + std::to_string(i) + ".bin";
            unshifted[i].serialize_to_file(path);
            streaming_unshifted_paths[i] = path;
        }
    }
    // get_to_be_shifted() is witness-only, no precomputed -- serialized after Oink
    auto to_be_shifted = polys.get_to_be_shifted();
    streaming_shifted_paths.resize(to_be_shifted.size());

    vinfo("serialized ", num_precomputed_serialized, " precomputed polys to disk before Oink");

    // Free precomputed polys not needed by any Oink round.
    // Oink needs: q_m, q_c, q_r, q_o, q_lookup (for log-derivative)
    //             table_1..4 (for log-derivative)
    //             sigma_1..4, id_1..4 (for grand product)
    // Safe to free: q_l, q_4, q_arith, q_delta_range, q_elliptic, q_memory, q_nnf,
    //               q_poseidon2_external, q_poseidon2_internal, lagrange_first, lagrange_last
    free_poly(polys.q_l);
    free_poly(polys.q_4);
    free_poly(polys.q_arith);
    free_poly(polys.q_delta_range);
    free_poly(polys.q_elliptic);
    free_poly(polys.q_memory);
    free_poly(polys.q_nnf);
    free_poly(polys.q_poseidon2_external);
    free_poly(polys.q_poseidon2_internal);
    free_poly(polys.lagrange_first);
    free_poly(polys.lagrange_last);
    vinfo("freed 11 precomputed polys not needed by Oink");

    // Initialize commitment key for Oink rounds
    commitment_key = CommitmentKey(prover_instance->dyadic_size());

    // Run Oink rounds individually for progressive memory management
    OinkProver<Flavor> oink_prover(prover_instance, honk_vk, transcript);
    oink_prover.commitment_key = commitment_key;

    oink_prover.send_vk_hash_and_public_inputs();
    oink_prover.commit_to_masking_poly();
    oink_prover.commit_to_wires();
    oink_prover.commit_to_lookup_counts_and_w4();
    oink_prover.commit_to_logderiv_inverses();

    // Free precomputed polys used only by log-derivative (lookup relation)
    free_poly(polys.q_lookup);
    free_poly(polys.table_1);
    free_poly(polys.table_2);
    free_poly(polys.table_3);
    free_poly(polys.table_4);
    free_poly(polys.q_m);
    free_poly(polys.q_c);
    free_poly(polys.q_r);
    free_poly(polys.q_o);
    vinfo("freed 9 log-derivative precomputed polys");

    oink_prover.commit_to_z_perm();

    // Free precomputed polys used only by grand product (permutation relation)
    free_poly(polys.sigma_1);
    free_poly(polys.sigma_2);
    free_poly(polys.sigma_3);
    free_poly(polys.sigma_4);
    free_poly(polys.id_1);
    free_poly(polys.id_2);
    free_poly(polys.id_3);
    free_poly(polys.id_4);
    vinfo("freed 8 grand product precomputed polys -- all precomputed now on disk only");

    // Generate alpha challenge (submodule: directly from transcript, not via OinkProver method)
    prover_instance->alpha = oink_prover.transcript->template get_challenge<FF>("alpha");

    // Free the commitment key used by Oink (will be recreated for PCS)
    commitment_key = CommitmentKey();

    // Phase 2: Serialize WITNESS polys (post-Oink, with correct w_4/lookup_inverses/z_perm).
    // Also serialize shifted views which reference witness poly memory.
    all_polys = polys.get_all();
    size_t num_witness_serialized = 0;
    for (size_t i = 0; i < all_polys.size(); i++) {
        if (streaming_all_poly_paths[i].empty() && !all_polys[i].is_empty()) {
            std::string path = streaming_temp_dir + "/all_" + std::to_string(i) + ".bin";
            all_polys[i].serialize_to_file(path);
            streaming_all_poly_paths[i] = path;
            num_witness_serialized++;
        }
    }

    // Serialize witness polys in PCS ordering (precomputed already serialized in Phase 1)
    {
        auto unshifted_post = polys.get_unshifted();
        for (size_t i = 0; i < unshifted_post.size(); i++) {
            if (streaming_unshifted_paths[i].empty() && !unshifted_post[i].is_empty()) {
                std::string path = streaming_temp_dir + "/unshifted_" + std::to_string(i) + ".bin";
                unshifted_post[i].serialize_to_file(path);
                streaming_unshifted_paths[i] = path;
            }
        }
        auto shifted_post = polys.get_to_be_shifted();
        for (size_t i = 0; i < shifted_post.size(); i++) {
            if (streaming_shifted_paths[i].empty() && !shifted_post[i].is_empty()) {
                std::string path = streaming_temp_dir + "/shifted_" + std::to_string(i) + ".bin";
                shifted_post[i].serialize_to_file(path);
                streaming_shifted_paths[i] = path;
            }
        }
    }
    vinfo("serialized ", num_witness_serialized, " witness polys post-Oink + PCS ordering");
    vinfo("created oink proof (low-memory path)");
}

/**
 * @brief Run Sumcheck to establish that ∑_i pow(\vec{β*})f_i(ω) = 0, producing sumcheck round challenges
 * u = (u_1,...,u_d) and claimed evaluations at u.
 */
template <typename Flavor> void UltraProver_<Flavor>::execute_sumcheck_iop()
{
    using Sumcheck = SumcheckProver<Flavor>;
    size_t polynomial_size = prover_instance->dyadic_size();
    Sumcheck sumcheck(polynomial_size,
                      prover_instance->polynomials,
                      transcript,
                      prover_instance->alpha,
                      prover_instance->gate_challenges,
                      prover_instance->relation_parameters,
                      virtual_log_n);

    // Enable streaming sumcheck when in low-memory mode to reduce peak memory.
    // In low-memory mode, construct_proof_low_memory() already serialized polys and freed precomputed.
    if (slow_low_memory) {
        if (streaming_temp_dir.empty()) {
            // Fallback: create temp dir if not already created by construct_proof_low_memory
            auto temp_dir = std::filesystem::temp_directory_path() / ("bb-streaming-" + std::to_string(getpid()));
            std::filesystem::create_directories(temp_dir);
            streaming_temp_dir = temp_dir.string();
        }
        sumcheck.streaming_temp_dir_ = streaming_temp_dir;
        vinfo("streaming sumcheck enabled, temp dir: ", streaming_temp_dir);

        // If construct_proof_low_memory already serialized polys, only serialize remaining witness polys.
        // Otherwise, serialize everything fresh.
        if (streaming_all_poly_paths.empty()) {
            // Fresh serialization (no low-memory Oink path used)
            auto unshifted = prover_instance->polynomials.get_unshifted();
            streaming_unshifted_paths.resize(unshifted.size());
            for (size_t i = 0; i < unshifted.size(); i++) {
                if (!unshifted[i].is_empty()) {
                    std::string path = streaming_temp_dir + "/unshifted_" + std::to_string(i) + ".bin";
                    unshifted[i].serialize_to_file(path);
                    streaming_unshifted_paths[i] = path;
                }
            }
            auto to_be_shifted = prover_instance->polynomials.get_to_be_shifted();
            streaming_shifted_paths.resize(to_be_shifted.size());
            for (size_t i = 0; i < to_be_shifted.size(); i++) {
                if (!to_be_shifted[i].is_empty()) {
                    std::string path = streaming_temp_dir + "/shifted_" + std::to_string(i) + ".bin";
                    to_be_shifted[i].serialize_to_file(path);
                    streaming_shifted_paths[i] = path;
                }
            }

            auto all_polys = prover_instance->polynomials.get_all();
            streaming_all_poly_paths.resize(all_polys.size());
            for (size_t i = 0; i < all_polys.size(); i++) {
                if (!all_polys[i].is_empty()) {
                    std::string path = streaming_temp_dir + "/all_" + std::to_string(i) + ".bin";
                    all_polys[i].serialize_to_file(path);
                    streaming_all_poly_paths[i] = path;
                }
            }
            vinfo("serialized all polys fresh: ", all_polys.size(), " get_all + ",
                  streaming_unshifted_paths.size(), " unshifted + ",
                  streaming_shifted_paths.size(), " shifted");
        } else {
            vinfo("using pre-serialized polys from low-memory Oink path");
        }

        // Compute effective round size before freeing remaining polys.
        size_t effective_round_size = polynomial_size;
        if constexpr (!Flavor::HasZK) {
            effective_round_size = 0;
            for (auto& witness_poly : prover_instance->polynomials.get_witness()) {
                if (!witness_poly.is_empty()) {
                    effective_round_size = std::max(effective_round_size, witness_poly.end_index());
                }
            }
            effective_round_size += effective_round_size % 2; // round up to even
            effective_round_size = std::min(effective_round_size, polynomial_size);
        }

        // Free ALL remaining source polynomials -- sumcheck will load from disk.
        auto all_polys = prover_instance->polynomials.get_all();
        for (auto& poly : all_polys) {
            poly = Polynomial();
        }
        vinfo("freed all source polys, effective_round_size=", effective_round_size);

        // Pass chunked streaming state to sumcheck
        sumcheck.streaming_all_poly_paths_ = std::move(streaming_all_poly_paths);
        sumcheck.streaming_effective_round_size_ = effective_round_size;
    }

    {
        BB_BENCH_NAME("sumcheck.prove");

        if constexpr (Flavor::HasZK) {
            zk_sumcheck_data = ZKData(numeric::get_msb(polynomial_size), transcript, commitment_key);
            sumcheck_output = sumcheck.prove(zk_sumcheck_data, prover_instance->masking_tail_data);
        } else {
            sumcheck_output = sumcheck.prove();
        }
    }
}

/**
 * @brief Reduce the sumcheck multivariate evaluations to a single univariate opening claim via Shplemini,
 * then produce an opening proof with the PCS (KZG or IPA).
 */
template <typename Flavor> void UltraProver_<Flavor>::execute_pcs()
{
    using OpeningClaim = ProverOpeningClaim<Curve>;
    using PolynomialBatcher = GeminiProver_<Curve>::PolynomialBatcher;

    auto& ck = commitment_key;

    PolynomialBatcher polynomial_batcher(prover_instance->dyadic_size(), prover_instance->polynomials.max_end_index());

    // In streaming mode, source polynomials were freed during sumcheck.
    // Set disk-backed paths so compute_batched loads from disk one at a time.
    if (!streaming_temp_dir.empty()) {
        polynomial_batcher.set_disk_backed(std::move(streaming_unshifted_paths), std::move(streaming_shifted_paths));
        vinfo("PCS using disk-backed polynomial streaming");
    } else {
        polynomial_batcher.set_unshifted(prover_instance->polynomials.get_unshifted());
        polynomial_batcher.set_to_be_shifted_by_one(prover_instance->polynomials.get_to_be_shifted());
    }

    // For ZK: register masking tail polynomials with the batcher so PCS includes them
    if constexpr (Flavor::HasZK) {
        if (prover_instance->masking_tail_data.is_active()) {
            prover_instance->masking_tail_data.add_tails_to_batcher(prover_instance->polynomials, polynomial_batcher);
        }
    }

    OpeningClaim prover_opening_claim;
    if constexpr (!Flavor::HasZK) {
        prover_opening_claim = ShpleminiProver_<Curve>::prove(
            prover_instance->dyadic_size(), polynomial_batcher, sumcheck_output.challenge, ck, transcript);
    } else {

        SmallSubgroupIPA small_subgroup_ipa_prover(
            zk_sumcheck_data, sumcheck_output.challenge, sumcheck_output.claimed_libra_evaluation, transcript, ck);
        small_subgroup_ipa_prover.prove();

        prover_opening_claim = ShpleminiProver_<Curve>::prove(prover_instance->dyadic_size(),
                                                              polynomial_batcher,
                                                              sumcheck_output.challenge,
                                                              ck,
                                                              transcript,
                                                              small_subgroup_ipa_prover.get_witness_polynomials());
    }
    vinfo("executed multivariate-to-univariate reduction");
    PCS::compute_opening_proof(ck, std::move(prover_opening_claim), transcript);
    vinfo("computed opening proof");

    // Clean up streaming temp files
    if (!streaming_temp_dir.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(streaming_temp_dir, ec);
        vinfo("cleaned up streaming temp dir");
    }
}

template class UltraProver_<UltraFlavor>;
template class UltraProver_<UltraZKFlavor>;
template class UltraProver_<UltraKeccakFlavor>;
#ifdef STARKNET_GARAGA_FLAVORS
template class UltraProver_<UltraStarknetFlavor>;
template class UltraProver_<UltraStarknetZKFlavor>;
#endif
template class UltraProver_<UltraKeccakZKFlavor>;
template class UltraProver_<MegaFlavor>;
template class UltraProver_<MegaZKFlavor>;
template class UltraProver_<MegaAvmFlavor>;

} // namespace bb
