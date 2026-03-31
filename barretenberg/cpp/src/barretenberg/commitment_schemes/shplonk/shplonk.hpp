// === AUDIT STATUS ===
// internal:    { status: Complete, auditors: [Khashayar], commit: }
// external_1:  { status: not started, auditors: [], commit: }
// external_2:  { status: not started, auditors: [], commit: }
// =====================

#pragma once
#include "barretenberg/commitment_schemes/claim.hpp"
#include "barretenberg/commitment_schemes/commitment_key.hpp"
#include "barretenberg/commitment_schemes/verification_key.hpp"
#include "barretenberg/common/assert.hpp"
#include "barretenberg/common/thread.hpp"
#include "barretenberg/stdlib/primitives/curves/bn254.hpp"
#include "barretenberg/transcript/transcript.hpp"

#include <future>


/**
 * @brief Reduces multiple claims about commitments, each opened at a single point
 *  into a single claim for a single polynomial opened at a single point.
 *
 * We use the following terminology:
 * - Bₖ(X) is a random linear combination of all polynomials opened at Ωₖ
 *   we refer to it a 'merged_polynomial'.
 * - Tₖ(X) is the polynomial that interpolates Bₖ(X) over Ωₖ,
 * - zₖ(X) is the product of all (X-x), for x ∈ Ωₖ
 * - ẑₖ(X) = 1/zₖ(X)
 *
 * The challenges are ρ (batching) and r (random evaluation).
 *
 */
namespace bb {

/**
 * @brief Shplonk Prover
 *
 * @tparam Curve EC parameters
 */
template <typename Curve> class ShplonkProver_ {
    using Fr = typename Curve::ScalarField;
    using Polynomial = bb::Polynomial<Fr>;

  public:
    /**
     * @brief Compute batched quotient polynomial Q(X) = ∑ⱼ νʲ ⋅ ( fⱼ(X) − vⱼ) / ( X − xⱼ )
     *
     * @param opening_claims list of prover opening claims {fⱼ(X), (xⱼ, vⱼ)} for a witness polynomial fⱼ(X), s.t. fⱼ(xⱼ)
     * = vⱼ.
     * @param nu batching challenge
     * @return Polynomial Q(X)
     */
    static Polynomial compute_batched_quotient(const size_t virtual_log_n,
                                               std::span<const ProverOpeningClaim<Curve>> opening_claims,
                                               const Fr& nu,
                                               std::span<Fr> gemini_fold_pos_evaluations,
                                               std::span<const ProverOpeningClaim<Curve>> libra_opening_claims,
                                               std::span<const ProverOpeningClaim<Curve>> sumcheck_round_claims)
    {
        BB_BENCH_NAME("Shplonk::compute_batched_quotient");
        // Find the maximum polynomial size among all claims to determine the dyadic size of the batched polynomial.
        size_t max_poly_size{ 0 };

        for (const auto& claim_set : { opening_claims, libra_opening_claims, sumcheck_round_claims }) {
            for (const auto& claim : claim_set) {
                max_poly_size = std::max(max_poly_size, claim.polynomial.size());
            }
        }
        // Q(X) = ∑ⱼ νʲ ⋅ ( fⱼ(X) − vⱼ) / ( X − xⱼ )
        Polynomial Q(max_poly_size);
        // Scratch buffers for out-of-place factor_roots (avoids copying claim polynomials)
        Polynomial tmp(max_poly_size, max_poly_size, 0, Polynomial::DontZeroMemory::FLAG);
        Polynomial tmp2(max_poly_size, max_poly_size, 0, Polynomial::DontZeroMemory::FLAG);

        Fr current_nu = Fr::one();

        // Parallelize factor_roots for the two large initial claims (A_0_pos, A_0_neg).
        // These are the first two non-gemini_fold claims, each at ~1M coefficients.
        // factor_roots is sequential (~20ms each), so running them in parallel saves ~20ms.
        // Out-of-place factor_roots reads directly from claim polynomial, no copy needed.
        size_t claims_start = 0;
        if (opening_claims.size() >= 2 && !opening_claims[0].gemini_fold && !opening_claims[1].gemini_fold) {
            Fr nu_0 = current_nu;
            current_nu *= nu;
            Fr nu_1 = current_nu;
            current_nu *= nu;

            auto future = std::async(std::launch::async, [&]() {
                opening_claims[0].polynomial.factor_roots_to(
                    tmp, opening_claims[0].opening_pair.challenge, opening_claims[0].opening_pair.evaluation);
            });

            opening_claims[1].polynomial.factor_roots_to(
                tmp2, opening_claims[1].opening_pair.challenge, opening_claims[1].opening_pair.evaluation);
            future.get();

            Q.add_scaled(tmp, nu_0);
            Q.add_scaled(tmp2, nu_1);
            claims_start = 2;
        }

        // Minimum polynomial size to justify parallel factor_roots (~16K coefficients).
        // Below this, thread spawn overhead exceeds the factor_roots savings.
        constexpr size_t PARALLEL_FACTOR_ROOTS_THRESHOLD = 1 << 14;

        size_t fold_idx = 0;
        for (size_t claim_idx = claims_start; claim_idx < opening_claims.size(); ++claim_idx) {
            const auto& claim = opening_claims[claim_idx];

            if (claim.gemini_fold) {
                // Gemini Fold Polynomials are opened at both -r^{2^j} and r^{2^j}.
                // For large folds, run both factor_roots in parallel since they're independent
                // serial recurrences on separate copies of the same polynomial.
                Fr nu_pos = current_nu;
                current_nu *= nu;
                Fr nu_neg = current_nu;
                current_nu *= nu;

                if (claim.polynomial.size() >= PARALLEL_FACTOR_ROOTS_THRESHOLD) {
                    auto future = std::async(std::launch::async, [&]() {
                        claim.polynomial.factor_roots_to(
                            tmp, -claim.opening_pair.challenge, gemini_fold_pos_evaluations[fold_idx]);
                    });

                    claim.polynomial.factor_roots_to(
                        tmp2, claim.opening_pair.challenge, claim.opening_pair.evaluation);
                    future.get();

                    Q.add_scaled(tmp, nu_pos);
                    Q.add_scaled(tmp2, nu_neg);
                } else {
                    // Small folds: sequential is cheaper than thread overhead
                    claim.polynomial.factor_roots_to(
                        tmp, -claim.opening_pair.challenge, gemini_fold_pos_evaluations[fold_idx]);
                    Q.add_scaled(tmp, nu_pos);

                    claim.polynomial.factor_roots_to(
                        tmp2, claim.opening_pair.challenge, claim.opening_pair.evaluation);
                    Q.add_scaled(tmp2, nu_neg);
                }
                fold_idx++;
                continue;
            }

            // Non-gemini_fold claim: single quotient
            claim.polynomial.factor_roots_to(
                tmp, claim.opening_pair.challenge, claim.opening_pair.evaluation);
            Q.add_scaled(tmp, current_nu);
            current_nu *= nu;
        }
        // We use the same batching challenge for Gemini and Libra opening claims. The number of the claims
        // batched before adding Libra commitments and evaluations is bounded by 2 * `virtual_log_n`,
        // which is the number of fold claims including the dummy ones.
        if (!libra_opening_claims.empty()) {
            current_nu = nu.pow(2 * virtual_log_n);
        }

        for (const auto& claim : libra_opening_claims) {
            // Compute individual claim quotient tmp = ( fⱼ(X) − vⱼ) / ( X − xⱼ )
            tmp = claim.polynomial;
            tmp.at(0) = tmp[0] - claim.opening_pair.evaluation;
            tmp.factor_roots(claim.opening_pair.challenge);

            // Add the claim quotient to the batched quotient polynomial
            Q.add_scaled(tmp, current_nu);
            current_nu *= nu;
        }

        for (const auto& claim : sumcheck_round_claims) {

            // Compute individual claim quotient tmp = ( fⱼ(X) − vⱼ) / ( X − xⱼ )
            tmp = claim.polynomial;
            tmp.at(0) = tmp[0] - claim.opening_pair.evaluation;
            tmp.factor_roots(claim.opening_pair.challenge);

            // Add the claim quotient to the batched quotient polynomial
            Q.add_scaled(tmp, current_nu);
            current_nu *= nu;
        }
        // Return batched quotient polynomial Q(X)
        return Q;
    };

    /**
     * @brief Compute partially evaluated batched quotient polynomial difference Q(X) - Q_z(X)
     *
     * @param opening_pairs list of opening pairs (xⱼ, vⱼ) for a witness polynomial fⱼ(X), s.t. fⱼ(xⱼ) = vⱼ.
     * @param witness_polynomials list of polynomials fⱼ(X).
     * @param batched_quotient_Q Q(X) = ∑ⱼ νʲ ⋅ ( fⱼ(X) − vⱼ) / ( X − xⱼ )
     * @param nu_challenge
     * @param z_challenge
     * @return Output{OpeningPair, Polynomial}
     */
    static ProverOpeningClaim<Curve> compute_partially_evaluated_batched_quotient(
        const size_t virtual_log_n,
        std::span<ProverOpeningClaim<Curve>> opening_claims,
        Polynomial& batched_quotient_Q,
        const Fr& nu_challenge,
        const Fr& z_challenge,
        std::span<Fr> gemini_fold_pos_evaluations,
        std::span<ProverOpeningClaim<Curve>> libra_opening_claims = {},
        std::span<ProverOpeningClaim<Curve>> sumcheck_opening_claims = {})
    {
        BB_BENCH_NAME("Shplonk::partial_eval_quotient");
        // Our main use case is the opening of Gemini fold polynomials and each Gemini fold is opened at 2 points.
        const size_t num_gemini_opening_claims = 2 * opening_claims.size();
        const size_t num_opening_claims =
            num_gemini_opening_claims + libra_opening_claims.size() + sumcheck_opening_claims.size();

        // {ẑⱼ(z)}ⱼ , where ẑⱼ(r) = 1/zⱼ(z) = 1/(z - xⱼ)
        std::vector<Fr> inverse_vanishing_evals;
        inverse_vanishing_evals.reserve(num_opening_claims);
        for (const auto& claim : opening_claims) {
            if (claim.gemini_fold) {
                inverse_vanishing_evals.emplace_back(z_challenge + claim.opening_pair.challenge);
            }
            inverse_vanishing_evals.emplace_back(z_challenge - claim.opening_pair.challenge);
        }

        // Add the terms (z - uₖ) for k = 0, …, d−1 where d is the number of rounds in Sumcheck
        for (const auto& claim : libra_opening_claims) {
            inverse_vanishing_evals.emplace_back(z_challenge - claim.opening_pair.challenge);
        }

        for (const auto& claim : sumcheck_opening_claims) {
            inverse_vanishing_evals.emplace_back(z_challenge - claim.opening_pair.challenge);
        }

        Fr::batch_invert(inverse_vanishing_evals);

        // G(X) = Q(X) - Q_z(X) = Q(X) - ∑ⱼ νʲ ⋅ ( fⱼ(X) − vⱼ) / ( z − xⱼ ),
        // s.t. G(r) = 0
        Polynomial G(std::move(batched_quotient_Q)); // G(X) = Q(X)

        // G₀ = ∑ⱼ νʲ ⋅ vⱼ / ( z − xⱼ )
        Fr current_nu = Fr::one();
        size_t idx = 0;

        size_t fold_idx = 0;
        for (auto& claim : opening_claims) {

            if (claim.gemini_fold) {
                // Save original a0, modify in-place for pos evaluation, then restore.
                // This avoids a full polynomial copy (tmp = claim.polynomial was ~16MB+ memcpy).
                Fr original_a0 = claim.polynomial[0];
                claim.polynomial.at(0) = original_a0 - gemini_fold_pos_evaluations[fold_idx++];
                Fr scaling_factor = current_nu * inverse_vanishing_evals[idx++]; // = νʲ / (z − xⱼ )
                // G -= νʲ ⋅ ( fⱼ(X) − vⱼ) / ( z − xⱼ )
                G.add_scaled(claim.polynomial, -scaling_factor);
                // Restore before neg evaluation use below
                claim.polynomial.at(0) = original_a0;

                current_nu *= nu_challenge;
            }
            claim.polynomial.at(0) = claim.polynomial[0] - claim.opening_pair.evaluation;
            Fr scaling_factor = current_nu * inverse_vanishing_evals[idx++]; // = νʲ / (z − xⱼ )

            // G -= νʲ ⋅ ( fⱼ(X) − vⱼ) / ( z − xⱼ )
            G.add_scaled(claim.polynomial, -scaling_factor);

            current_nu *= nu_challenge;
        }

        // Take into account the constant proof size in Gemini
        if (!libra_opening_claims.empty()) {
            current_nu = nu_challenge.pow(2 * virtual_log_n);
        }

        for (auto& claim : libra_opening_claims) {
            claim.polynomial.at(0) = claim.polynomial[0] - claim.opening_pair.evaluation;
            Fr scaling_factor = current_nu * inverse_vanishing_evals[idx++];
            G.add_scaled(claim.polynomial, -scaling_factor);
            current_nu *= nu_challenge;
        }

        for (auto& claim : sumcheck_opening_claims) {
            claim.polynomial.at(0) = claim.polynomial[0] - claim.opening_pair.evaluation;
            Fr scaling_factor = current_nu * inverse_vanishing_evals[idx++];
            G.add_scaled(claim.polynomial, -scaling_factor);
            current_nu *= nu_challenge;
        }
        // Return opening pair (z, 0) and polynomial G(X) = Q(X) - Q_z(X)
        return { .polynomial = G, .opening_pair = { .challenge = z_challenge, .evaluation = Fr::zero() } };
    };
    /**
     * @brief Compute evaluations of fold polynomials Fold_i at r^{2^i} for i>0.
     * TODO(https://github.com/AztecProtocol/barretenberg/issues/1223): Reconsider minor performance/memory
     * optimizations in Gemini.
     * @param opening_claims
     * @return std::vector<Fr>
     */
    static std::vector<Fr> compute_gemini_fold_pos_evaluations(
        std::span<const ProverOpeningClaim<Curve>> opening_claims)
    {
        // Collect fold claim indices for parallel evaluation
        std::vector<size_t> fold_indices;
        fold_indices.reserve(opening_claims.size());
        for (size_t i = 0; i < opening_claims.size(); ++i) {
            if (opening_claims[i].gemini_fold) {
                fold_indices.push_back(i);
            }
        }

        // Evaluate all fold polynomials at their positive challenge points in parallel.
        // Large folds (fold_1 at ~524K coefficients) dominate; parallelism lets them
        // run on separate cores instead of sequentially.
        std::vector<Fr> gemini_fold_pos_evaluations(fold_indices.size());
        parallel_for(fold_indices.size(), [&](size_t idx) {
            const auto& claim = opening_claims[fold_indices[idx]];
            // -r^{2^i} is stored in the claim; negate to get positive evaluation point
            const Fr evaluation_point = -claim.opening_pair.challenge;
            gemini_fold_pos_evaluations[idx] = claim.polynomial.evaluate(evaluation_point);
        });

        return gemini_fold_pos_evaluations;
    }

    /**
     * @brief Returns a batched opening claim equivalent to a set of opening claims consisting of polynomials, each
     * opened at a single point.
     *
     * @param commitment_key
     * @param opening_claims
     * @param transcript
     * @return ProverOpeningClaim<Curve>
     */
    template <typename Transcript>
    static ProverOpeningClaim<Curve> prove(const CommitmentKey<Curve>& commitment_key,
                                           std::span<ProverOpeningClaim<Curve>> opening_claims,
                                           const std::shared_ptr<Transcript>& transcript,
                                           std::span<ProverOpeningClaim<Curve>> libra_opening_claims = {},
                                           std::span<ProverOpeningClaim<Curve>> sumcheck_round_claims = {},
                                           const size_t virtual_log_n = 0)
    {
        const Fr nu = transcript->template get_challenge<Fr>("Shplonk:nu");

        std::vector<Fr> gemini_fold_pos_evaluations;
        {
            BB_BENCH_NAME("Shplonk::fold_pos_evals");
            // Compute the evaluations Fold_i(r^{2^i}) for i>0.
            gemini_fold_pos_evaluations = compute_gemini_fold_pos_evaluations(opening_claims);
        }

        auto batched_quotient = compute_batched_quotient(virtual_log_n,
                                                         opening_claims,
                                                         nu,
                                                         gemini_fold_pos_evaluations,
                                                         libra_opening_claims,
                                                         sumcheck_round_claims);
        auto batched_quotient_commitment = commitment_key.commit(batched_quotient);
        transcript->send_to_verifier("Shplonk:Q", batched_quotient_commitment);
        const Fr z = transcript->template get_challenge<Fr>("Shplonk:z");

        return compute_partially_evaluated_batched_quotient(virtual_log_n,
                                                            opening_claims,
                                                            batched_quotient,
                                                            nu,
                                                            z,
                                                            gemini_fold_pos_evaluations,
                                                            libra_opening_claims,
                                                            sumcheck_round_claims);
    }
};

/**
 * @brief Shplonk Verifier
 *
 *
 * @details Given commitments to polynomials \f$[p_1], \dots, [p_m]\f$ and couples of challenge/evaluation
 * \f$(x_i, v_i)\f$, the Shplonk verifier computes the following commitment:
 * \f[
 *      [G] := [Q] - \sum_{i=1}^m \frac{\nu^{i-1} [p_i]}{(z - x_i)} + \sum_{i=1}^m \frac{\nu^{i-1} v_i}{(z - x_i)} [1]
 * \f]
 * where \f$\nu\f$ is a random batching challenge, \f$[Q]\f$ is the commiment to the quotient polymomial
 * \f[
 *      \sum_{i=1}^m \nu^{i-1} \frac{(p_i - v_i)}{(x - x_i)}
 * \f]
 * and \f$z\f$ is the evaluation challenge.
 *
 * When the polynomials \f$p_1, \dots, p_m\f$ are linearly dependent, and the verifier which calls the Shplonk
 * verifier needs to compute the commitments \f$[p_1], \dots, [p_m]\f$ starting from the linearly independent factors,
 * computing the commitments and then executing the Shplonk verifier is not the most efficient way to execute the
 * Shplonk verifier algorithm.
 *
 * Consider the case \f$m = 2\f$, and take \f$p_2 = a p_1\f$ for some constant \f$a \in \mathbb{F}\f$. Then, the
 * most efficient way to execute the Shplonk verifier algorithm is to compute the following MSM
 * \f[
 *      [Q] - \left( \frac{1}{(z - x_1)} \
 *                  + \frac{a \nu}{(z - x_2)} \right) [p_1]  \
 *                      + \left( \frac{v_1}{(z - x_1)} + \frac{v_2 \nu}{(z - x_2)} \right) [1]
 * \f]
 *
 * The Shplonk verifier api is designed to allow the execution of the Shplonk verifier algorithm in its most efficient
 * form. To achieve this, the Shplonk verifier maintains an internal state depending of the following variables:
 *  - \f$[f_1], \dots, [f_n]\f$ (`commitments` in code) the commitments to the linearly independent polynomials such
 * that for each polynomial \f$p_i\f$ we wish to open it holds \f$p_i = \sum_{i=1}^n p_{i,j} f_j\f$ for some \f$p_j
 * \in \mathbb{F}\f$.
 *  - \f$\nu\f$ (`nu` in code) the challenge used to batch the polynomial commitments.
 *  - \f$\nu^{i}\f$ (`current_nu` in code), which is the power of the batching challenge used to batch the
 *      \f$i\f$-th polynomial \f$ p_i \f$ in the Shplonk verifier algorithm.
 *  - \f$[Q]\f$ (`quotient` in code).
 *  - \f$z\f$ (`z_challenge` in code), the partial evaluation challenge.
 *  - \f$(s_1, \dots, s_n)\f$ (`scalars` in code), the coefficient of \f$[f_i]\f$ in the Shplonk verifier MSM.
 *  - \f$\theta\f$ (`identity_scalar_coefficient` in code), the coefficient of \f$[1]\f$ in the Shplonk verifier MSM.
 *  - `evaluation`, the claimed evaluation at \f$z\f$ of the commitment produced by the Shplonk verifier, always equal
 *      to \f$0\f$.
 */
template <typename Curve> class ShplonkVerifier_ {
    using Fr = typename Curve::ScalarField;
    using GroupElement = typename Curve::Element;
    using Commitment = typename Curve::AffineElement;
    using VK = VerifierCommitmentKey<Curve>;

    // Random challenges
    std::vector<Fr> pows_of_nu;
    // Commitment to quotient polynomial
    Commitment quotient;
    // Partial evaluation challenge
    Fr z_challenge;
    // Commitments \f$[f_1], \dots, [f_n]\f$
    std::vector<Commitment> commitments;
    // Scalar coefficients of \f$[f_1], \dots, [f_n]\f$ in the MSM needed to compute the commitment to the partially
    // evaluated quotient
    std::vector<Fr> scalars;
    // Coefficient of the identity in partially evaluated quotient
    Fr identity_scalar_coefficient = Fr(0);
    // Target evaluation
    Fr evaluation = Fr(0);

  public:
    template <typename Transcript>
    ShplonkVerifier_(std::vector<Commitment>& polynomial_commitments,
                     std::shared_ptr<Transcript>& transcript,
                     const size_t num_claims)
        : pows_of_nu({ Fr(1), transcript->template get_challenge<Fr>("Shplonk:nu") })
        , quotient(transcript->template receive_from_prover<Commitment>("Shplonk:Q"))
        , z_challenge(transcript->template get_challenge<Fr>("Shplonk:z"))
        , commitments({ quotient })
        , scalars{ Fr{ 1 } }
    {
        BB_ASSERT_GT(num_claims, 1U, "Using Shplonk with just one claim. Should use batch reduction.");
        const size_t num_commitments = commitments.size();
        commitments.reserve(num_commitments);
        scalars.reserve(num_commitments);
        pows_of_nu.reserve(num_claims);

        commitments.insert(commitments.end(), polynomial_commitments.begin(), polynomial_commitments.end());
        scalars.insert(scalars.end(), commitments.size() - 1, Fr(0)); // Initialized as circuit constants
        // The first two powers of nu have already been initialized, we need another `num_claims - 2` powers to batch
        // all the claims
        for (size_t idx = 0; idx < num_claims - 2; idx++) {
            pows_of_nu.emplace_back(pows_of_nu.back() * pows_of_nu[1]);
        }

        if constexpr (Curve::is_stdlib_type) {
            evaluation.convert_constant_to_fixed_witness(pows_of_nu[1].get_context());
        }
    }

    /**
     * @brief Finalize the Shplonk verification and return the KZG opening claim
     *
     * @details Compute the commitment:
     *      \f[ [Q] - \sum_i s_i * [f_i] + \theta * [1] \f]
     * @param g1_identity
     * @return OpeningClaim<Curve>
     */
    OpeningClaim<Curve> finalize(const Commitment& g1_identity)
    {
        commitments.emplace_back(g1_identity);
        scalars.emplace_back(identity_scalar_coefficient);
        GroupElement result = GroupElement::batch_mul(commitments, scalars);

        return { { z_challenge, evaluation }, result };
    }

    /**
     * @brief Export a BatchOpeningClaim instead of performing final batch_mul
     *
     * @details Append g1_identity to `commitments`, `identity_scalar_factor` to scalars, and export the resulting
     * vectors. This method is useful when we perform KZG verification of the Shplonk claim right after Shplonk (because
     * we can add the last commitment \f$[W]\f$ and scalar factor (0 in this case) to the list and then execute a single
     * batch mul.
     *
     * @note This function modifies the `commitments` and `scalars` attribute of the class instance on which it is
     * called.
     *
     * @param g1_identity
     * @return BatchOpeningClaim<Curve>
     */
    // TODO(https://github.com/AztecProtocol/barretenberg/issues/1475): Compute g1_identity inside the function body
    BatchOpeningClaim<Curve> export_batch_opening_claim(const Commitment& g1_identity)
    {
        commitments.emplace_back(g1_identity);
        scalars.emplace_back(identity_scalar_coefficient);

        return { commitments, scalars, z_challenge };
    }

    /**
     * @brief Instantiate a Shplonk verifier and update its state with the provided claims.
     *
     * @param claims List of opening claims \f$(C_j, x_j, v_j)\f$ for a witness polynomial \f$f_j(X)\f$, s.t.
     * \f$f_j(x_j) = v_j\f$.
     * @param transcript
     */
    template <typename Transcript>
    static ShplonkVerifier_<Curve> reduce_verification_no_finalize(std::span<const OpeningClaim<Curve>> claims,
                                                                   std::shared_ptr<Transcript>& transcript)
    {
        // Initialize Shplonk verifier
        const size_t num_claims = claims.size();
        std::vector<Commitment> polynomial_commiments;
        polynomial_commiments.reserve(num_claims);
        for (const auto& claim : claims) {
            polynomial_commiments.emplace_back(claim.commitment);
        }
        ShplonkVerifier_<Curve> verifier(polynomial_commiments, transcript, num_claims);

        // Compute { 1 / (z - x_i) }
        std::vector<Fr> inverse_vanishing_evals;
        inverse_vanishing_evals.reserve(num_claims);
        if constexpr (Curve::is_stdlib_type) {
            for (const auto& claim : claims) {
                inverse_vanishing_evals.emplace_back((verifier.z_challenge - claim.opening_pair.challenge).invert());
            }
        } else {
            for (const auto& claim : claims) {
                inverse_vanishing_evals.emplace_back(verifier.z_challenge - claim.opening_pair.challenge);
            }
            Fr::batch_invert(inverse_vanishing_evals);
        }

        // Update the Shplonk verifier state with each claim
        // For each claim: s_i -= ν^i / (z - x_i) and θ += ν^i * v_i / (z - x_i)
        for (size_t idx = 0; idx < claims.size(); idx++) {
            // Compute ν^i / (z - x_i)
            auto scalar_factor = verifier.pows_of_nu[idx] * inverse_vanishing_evals[idx];
            // s_i -= ν^i / (z - x_i)
            verifier.scalars[idx + 1] -= scalar_factor;
            // θ += ν^i * v_i / (z - x_i)
            verifier.identity_scalar_coefficient += scalar_factor * claims[idx].opening_pair.evaluation;
        }

        return verifier;
    };

    /**
     * @brief Recomputes the new claim commitment [G] given the proof and
     * the challenge r. No verification happens so this function always succeeds.
     *
     * @param g1_identity the identity element for the Curve
     * @param claims list of opening claims \f$(C_j, x_j, v_j)\f$ for a witness polynomial \f$f_j(X)\f$, s.t.
     * \f$f_j(x_j) = v_j\f$.
     * @param transcript
     * @return OpeningClaim
     */
    template <typename Transcript>
    static OpeningClaim<Curve> reduce_verification(Commitment g1_identity,
                                                   std::span<const OpeningClaim<Curve>> claims,
                                                   std::shared_ptr<Transcript>& transcript)
    {
        auto verifier = ShplonkVerifier_::reduce_verification_no_finalize(claims, transcript);
        return verifier.finalize(g1_identity);
    };

    /**
     * @brief Computes \f$ \frac{1}{z - r}, \frac{1}{z + r}, \ldots, \frac{1}{z - r^{2^{d-1}}}, \frac{1}{z +
     * r^{2^{d-1}}} \f$.
     *
     * @param shplonk_eval_challenge \f$ z \f$
     * @param gemini_eval_challenge_powers \f$ (r , r^2, \ldots, r^{2^{d-1}}) \f$
     * @return \f[ \left( \frac{1}{z - r}, \frac{1}{z + r},  \ldots, \frac{1}{z - r^{2^{d-1}}}, \frac{1}{z +
     * r^{2^{d-1}}} \right) \f]
     */
    static std::vector<Fr> compute_inverted_gemini_denominators(const Fr& shplonk_eval_challenge,
                                                                const std::vector<Fr>& gemini_eval_challenge_powers)
    {
        std::vector<Fr> denominators;
        const size_t virtual_log_n = gemini_eval_challenge_powers.size();
        const size_t num_gemini_claims = 2 * virtual_log_n;
        denominators.reserve(num_gemini_claims);

        for (const auto& gemini_eval_challenge_power : gemini_eval_challenge_powers) {
            // Place 1/(z - r ^ {2^j})
            denominators.emplace_back(shplonk_eval_challenge - gemini_eval_challenge_power);
            // Place 1/(z + r ^ {2^j})
            denominators.emplace_back(shplonk_eval_challenge + gemini_eval_challenge_power);
        }

        if constexpr (!Curve::is_stdlib_type) {
            Fr::batch_invert(denominators);
        } else {
            for (auto& denominator : denominators) {
                denominator = denominator.invert();
            }
        }
        return denominators;
    }
};

/**
 * @brief A helper used by Shplemini Verifier. Precomputes a vector of the powers of \f$ \nu \f$ needed to batch all
 * univariate claims.
 *
 */
template <typename Fr>
static std::vector<Fr> compute_shplonk_batching_challenge_powers(const Fr& shplonk_batching_challenge,
                                                                 const size_t virtual_log_n,
                                                                 bool has_zk = false,
                                                                 bool committed_sumcheck = false)
{
    // Minimum number of powers: 2 * virtual_log_n for the Gemini fold claims
    size_t num_powers = 2 * virtual_log_n;
    // Each round univariate is opened at 0, 1, and a round challenge.
    static constexpr size_t NUM_COMMITTED_SUMCHECK_CLAIMS_PER_ROUND = 3;

    // Shplonk evaluation and batching challenges are re-used in SmallSubgroupIPA.
    if (has_zk) {
        num_powers += NUM_SMALL_IPA_EVALUATIONS;
    }

    // Commited sumcheck adds 3 claims per round.
    if (committed_sumcheck) {
        num_powers += NUM_COMMITTED_SUMCHECK_CLAIMS_PER_ROUND * virtual_log_n;
    }

    std::vector<Fr> result;
    result.reserve(num_powers);
    result.emplace_back(Fr{ 1 });
    for (size_t idx = 1; idx < num_powers; idx++) {
        result.emplace_back(result[idx - 1] * shplonk_batching_challenge);
    }
    return result;
}
} // namespace bb
