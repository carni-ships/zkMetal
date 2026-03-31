#include "barretenberg/bbapi/bbapi_ultra_honk.hpp"
#include "barretenberg/bbapi/bbapi_shared.hpp"
#include "barretenberg/common/serialize.hpp"
#include "barretenberg/dsl/acir_format/acir_to_constraint_buf.hpp"
#include "barretenberg/dsl/acir_format/serde/witness_stack.hpp"
#include "barretenberg/dsl/acir_proofs/honk_contract.hpp"
#include "barretenberg/dsl/acir_proofs/honk_optimized_contract.hpp"
#include "barretenberg/dsl/acir_proofs/honk_zk_contract.hpp"
#include "barretenberg/numeric/uint256/uint256.hpp"
#include "barretenberg/ultra_honk/ultra_prover.hpp"
#include "barretenberg/ultra_honk/ultra_verifier.hpp"
#include "barretenberg/ecc/scalar_multiplication/metal/metal_msm.hpp"

#include "barretenberg/api/file_io.hpp"

#include <mutex>
#include <string_view>
#include <unordered_map>

namespace bb::bbapi {

template <typename IO> acir_format::ProgramMetadata _create_program_metadata()
{
    return acir_format::ProgramMetadata{ .has_ipa_claim = IO::HasIPA };
}

template <typename Flavor, typename IO, typename Circuit = typename Flavor::CircuitBuilder>
Circuit _compute_circuit(std::vector<uint8_t>&& bytecode, std::vector<uint8_t>&& witness)
{
    const acir_format::ProgramMetadata metadata = _create_program_metadata<IO>();
    acir_format::AcirProgram program{ acir_format::circuit_buf_to_acir_format(std::move(bytecode)), {} };

    if (!witness.empty()) {
        program.witness = acir_format::witness_buf_to_witness_vector(std::move(witness));
    }
    return acir_format::create_circuit<Circuit>(program, metadata);
}

// Static cache for precomputed polynomial data, keyed by bytecode hash.
// Enables fast re-proving when the same circuit is proved multiple times with different witnesses.
template <typename Flavor> struct ProverInstanceCache {
    static std::mutex mutex;
    static std::unordered_map<size_t, PrecomputedCache<Flavor>> cache;
};
template <typename Flavor> std::mutex ProverInstanceCache<Flavor>::mutex;
template <typename Flavor> std::unordered_map<size_t, PrecomputedCache<Flavor>> ProverInstanceCache<Flavor>::cache;

template <typename Flavor, typename IO>
std::shared_ptr<ProverInstance_<Flavor>> _compute_prover_instance(const std::vector<uint8_t>& bytecode,
                                                                  std::vector<uint8_t>&& witness)
{
    // Check for cached precomputed data
    const size_t bytecode_hash = std::hash<std::string_view>{}(
        std::string_view(reinterpret_cast<const char*>(bytecode.data()), bytecode.size()));

    {
        std::lock_guard<std::mutex> lock(ProverInstanceCache<Flavor>::mutex);
        auto it = ProverInstanceCache<Flavor>::cache.find(bytecode_hash);
        if (it != ProverInstanceCache<Flavor>::cache.end()) {
            BB_BENCH_NAME("ProverInstance (cached)");
            vinfo("Using cached precomputed polynomials");

            // Build circuit (still needed for witness values)
            std::vector<uint8_t> bytecode_copy(bytecode);
            typename Flavor::CircuitBuilder builder =
                _compute_circuit<Flavor, IO>(std::move(bytecode_copy), std::move(witness));

            // Construct ProverInstance using cached precomputed data
            auto prover_instance = std::make_shared<ProverInstance_<Flavor>>(builder, it->second);

            if constexpr (IO::HasIPA) {
                BB_ASSERT(!prover_instance->ipa_proof.empty(),
                          "RollupIO circuit expected IPA proof but none was provided.");
            } else {
                BB_ASSERT(prover_instance->ipa_proof.empty(), "Non-rollup circuit should not have IPA proof.");
            }
            return prover_instance;
        }
    }

    // Cache miss: build normally
    std::vector<uint8_t> bytecode_copy(bytecode);
    typename Flavor::CircuitBuilder builder =
        _compute_circuit<Flavor, IO>(std::move(bytecode_copy), std::move(witness));
    auto prover_instance = std::make_shared<ProverInstance_<Flavor>>(builder);

    // Validate consistency between IO type and IPA proof presence
    if constexpr (IO::HasIPA) {
        BB_ASSERT(!prover_instance->ipa_proof.empty(),
                  "RollupIO circuit expected IPA proof but none was provided. "
                  "Ensure the circuit includes IPA accumulation data.");
    } else {
        BB_ASSERT(prover_instance->ipa_proof.empty(),
                  "Non-rollup circuit should not have IPA proof. "
                  "Use ipa_accumulation=true in settings for rollup circuits.");
    }

    // Cache the precomputed data for future proofs with the same circuit
    {
        std::lock_guard<std::mutex> lock(ProverInstanceCache<Flavor>::mutex);
        ProverInstanceCache<Flavor>::cache.emplace(bytecode_hash, prover_instance->create_cache());
    }

    return prover_instance;
}
template <typename Flavor, typename IO>
CircuitProve::Response _prove(const std::vector<uint8_t>& bytecode,
                              std::vector<uint8_t>&& witness,
                              std::vector<uint8_t>&& vk_bytes)
{
    using Proof = typename Flavor::Transcript::Proof;
    using VerificationKey = typename Flavor::VerificationKey;

    // Start Metal GPU init on background thread — overlaps with proving key computation (~38ms).
    // By the time prewarm() is called in OinkProver, init is already done (saves ~33ms).
#if BB_METAL_MSM_AVAILABLE
    if constexpr (std::is_same_v<typename Flavor::Curve, curve::BN254>) {
        scalar_multiplication::metal::metal_init_async();
    }
#endif

    auto prover_instance = _compute_prover_instance<Flavor, IO>(bytecode, std::move(witness));

    // Create or deserialize VK
    std::shared_ptr<VerificationKey> vk;
    if (vk_bytes.empty()) {
        info("WARNING: computing verification key while proving. Pass in a precomputed vk for better performance.");
        vk = std::make_shared<VerificationKey>(prover_instance->get_precomputed());
    } else {
        vk = std::make_shared<VerificationKey>(from_buffer<VerificationKey>(vk_bytes));
    }

    // Construct proof
    UltraProver_<Flavor> prover{ prover_instance, vk };
    Proof full_proof = prover.construct_proof();

    // BB_REPEAT: re-prove using cached precomputed data to measure cache speedup
    if (std::getenv("BB_REPEAT")) {
        // The cache was populated by the first _compute_prover_instance call above.
        // Load witness again from the original file (re-read needed since witness was moved).
        info("BB_REPEAT: re-proving with cached precomputed data...");
        // We need fresh witness data. Read from the witness file path via environment.
        const char* witness_path = std::getenv("BB_REPEAT_WITNESS");
        if (witness_path) {
            auto witness_data = read_file(witness_path);
            auto cached_instance = _compute_prover_instance<Flavor, IO>(bytecode, std::move(witness_data));
            auto cached_vk = vk; // reuse VK
            UltraProver_<Flavor> cached_prover{ cached_instance, cached_vk };
            auto cached_proof = cached_prover.construct_proof();
            info("BB_REPEAT: cached proof generated (", cached_proof.size(), " elements)");
        } else {
            info("BB_REPEAT: set BB_REPEAT_WITNESS=<path> to provide witness for cached re-prove");
        }
    }

    // Compute where to split (inner public inputs vs everything else)
    size_t num_public_inputs = prover.num_public_inputs();
    BB_ASSERT_GTE(num_public_inputs, IO::PUBLIC_INPUTS_SIZE, "Public inputs should contain the expected IO structure.");
    size_t num_inner_public_inputs = num_public_inputs - IO::PUBLIC_INPUTS_SIZE;

    // Optimization: if vk not provided, include it in response
    CircuitComputeVk::Response vk_response;
    if (vk_bytes.empty()) {
        vk_response = { .bytes = to_buffer(*vk), .fields = vk_to_uint256_fields(*vk), .hash = to_buffer(vk->hash()) };
    }

    // Split proof: inner public inputs at front, rest is the "proof"
    return { .public_inputs =
                 std::vector<uint256_t>{ full_proof.begin(),
                                         full_proof.begin() + static_cast<std::ptrdiff_t>(num_inner_public_inputs) },
             .proof = std::vector<uint256_t>{ full_proof.begin() + static_cast<std::ptrdiff_t>(num_inner_public_inputs),
                                              full_proof.end() },
             .vk = std::move(vk_response) };
}

template <typename Flavor, typename IO>
bool _verify(const std::vector<uint8_t>& vk_bytes,
             const std::vector<uint256_t>& public_inputs,
             const std::vector<uint256_t>& proof)
{
    using VerificationKey = typename Flavor::VerificationKey;
    using VKAndHash = typename Flavor::VKAndHash;
    using Verifier = UltraVerifier_<Flavor, IO>;

    // Validate VK size upfront before deserialization
    const size_t expected_vk_size = VerificationKey::calc_num_data_types() * sizeof(bb::fr);
    if (vk_bytes.size() != expected_vk_size) {
        info(
            "Proof verification failed: invalid VK size. Expected ", expected_vk_size, " bytes, got ", vk_bytes.size());
        return false;
    }

    std::shared_ptr<VerificationKey> vk = std::make_shared<VerificationKey>(from_buffer<VerificationKey>(vk_bytes));
    auto vk_and_hash = std::make_shared<VKAndHash>(vk);
    Verifier verifier{ vk_and_hash };

    // Validate proof size
    const size_t log_n = verifier.compute_log_n();
    const size_t expected_size = ProofLength::Honk<Flavor>::template expected_proof_size<IO>(log_n);
    if (proof.size() != expected_size) {
        info("Proof verification failed: invalid proof size. Expected ", expected_size, ", got ", proof.size());
        return false;
    }

    auto complete_proof = concatenate_proof<Flavor>(public_inputs, proof);
    bool verified = verifier.verify_proof(complete_proof).result;

    if (verified) {
        info("Proof verified successfully");
    } else {
        info("Proof verification failed");
    }

    return verified;
}

CircuitProve::Response CircuitProve::execute(BB_UNUSED const BBApiRequest& request) &&
{
    BB_BENCH_NAME(MSGPACK_SCHEMA_NAME);
    return dispatch_by_settings(settings, [&]<typename Flavor, typename IO>() {
        return _prove<Flavor, IO>(std::move(circuit.bytecode), std::move(witness), std::move(circuit.verification_key));
    });
}

CircuitComputeVk::Response CircuitComputeVk::execute(BB_UNUSED const BBApiRequest& request) &&
{
    BB_BENCH_NAME(MSGPACK_SCHEMA_NAME);
    return dispatch_by_settings(settings, [&]<typename Flavor, typename IO>() {
        auto prover_instance = _compute_prover_instance<Flavor, IO>(circuit.bytecode, {});
        auto vk = std::make_shared<typename Flavor::VerificationKey>(prover_instance->get_precomputed());
        return CircuitComputeVk::Response{ .bytes = to_buffer(*vk),
                                           .fields = vk_to_uint256_fields(*vk),
                                           .hash = to_buffer(vk->hash()) };
    });
}

template <typename Flavor, typename IO>
CircuitStats::Response _stats(std::vector<uint8_t>&& bytecode, bool include_gates_per_opcode)
{
    using Circuit = typename Flavor::CircuitBuilder;
    // Parse the circuit to get gate count information
    auto constraint_system = acir_format::circuit_buf_to_acir_format(std::move(bytecode));

    acir_format::ProgramMetadata metadata = _create_program_metadata<IO>();
    metadata.collect_gates_per_opcode = include_gates_per_opcode;
    CircuitStats::Response response;
    response.num_acir_opcodes = static_cast<uint32_t>(constraint_system.num_acir_opcodes);

    acir_format::AcirProgram program{ std::move(constraint_system), {} };
    auto builder = acir_format::create_circuit<Circuit>(program, metadata);
    builder.finalize_circuit(/*ensure_nonzero=*/true);

    response.num_gates = static_cast<uint32_t>(builder.get_finalized_total_circuit_size());
    response.num_gates_dyadic = static_cast<uint32_t>(builder.get_circuit_subgroup_size(response.num_gates));
    // note: will be empty if collect_gates_per_opcode is false
    response.gates_per_opcode =
        std::vector<uint32_t>(program.constraints.gates_per_opcode.begin(), program.constraints.gates_per_opcode.end());

    return response;
}

CircuitStats::Response CircuitStats::execute(BB_UNUSED const BBApiRequest& request) &&
{
    BB_BENCH_NAME(MSGPACK_SCHEMA_NAME);
    return dispatch_by_settings(settings, [&]<typename Flavor, typename IO>() {
        return _stats<Flavor, IO>(std::move(circuit.bytecode), include_gates_per_opcode);
    });
}

CircuitVerify::Response CircuitVerify::execute(BB_UNUSED const BBApiRequest& request) &&
{
    BB_BENCH_NAME(MSGPACK_SCHEMA_NAME);
    bool verified = dispatch_by_settings(settings, [&]<typename Flavor, typename IO>() {
        return _verify<Flavor, IO>(verification_key, public_inputs, proof);
    });
    return { verified };
}

VkAsFields::Response VkAsFields::execute(BB_UNUSED const BBApiRequest& request) &&
{
    BB_BENCH_NAME(MSGPACK_SCHEMA_NAME);

    using VK = UltraFlavor::VerificationKey;
    validate_vk_size<VK>(verification_key);

    // Standard UltraHonk flavors
    auto vk = from_buffer<VK>(verification_key);
    std::vector<bb::fr> fields;
    fields = vk.to_field_elements();

    return { std::move(fields) };
}

MegaVkAsFields::Response MegaVkAsFields::execute(BB_UNUSED const BBApiRequest& request) &&
{
    BB_BENCH_NAME(MSGPACK_SCHEMA_NAME);

    using VK = MegaFlavor::VerificationKey;
    validate_vk_size<VK>(verification_key);

    // MegaFlavor for private function verification keys
    auto vk = from_buffer<VK>(verification_key);
    std::vector<bb::fr> fields;
    fields = vk.to_field_elements();

    return { std::move(fields) };
}

CircuitWriteSolidityVerifier::Response CircuitWriteSolidityVerifier::execute(BB_UNUSED const BBApiRequest& request) &&
{
    BB_BENCH_NAME(MSGPACK_SCHEMA_NAME);
    using VK = UltraKeccakFlavor::VerificationKey;
    validate_vk_size<VK>(verification_key);

    auto vk = std::make_shared<VK>(from_buffer<VK>(verification_key));

    std::string contract = settings.disable_zk ? get_honk_solidity_verifier(vk) : get_honk_zk_solidity_verifier(vk);

// If in wasm, we dont include the optimized solidity verifier - due to its large bundle size
// This will run generate twice, but this should only be run before deployment and not frequently
#ifndef __wasm__
    if (settings.disable_zk && settings.optimized_solidity_verifier) {
        contract = get_optimized_honk_solidity_verifier(vk);
    }
#endif

    return { std::move(contract) };
}

} // namespace bb::bbapi
