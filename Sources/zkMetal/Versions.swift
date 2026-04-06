// Primitive implementation versions
// Bump the version when an engine's implementation changes meaningfully
// (optimization, algorithm change, bug fix, new kernel).
// Format: "major.minor.patch" — major = algorithm/API change, minor = optimization, patch = bug fix

public struct PrimitiveVersion {
    public let version: String
    public let updated: String  // YYYY-MM-DD

    public var description: String { "\(version) (\(updated))" }
}

public enum Versions {
    // --- Hash ---
    public static let poseidon2       = PrimitiveVersion(version: "1.3.0", updated: "2026-04-03")
    public static let poseidon2M31    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let poseidon2BabyBear = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let keccak256       = PrimitiveVersion(version: "1.2.0", updated: "2026-04-02")
    public static let blake3          = PrimitiveVersion(version: "1.1.0", updated: "2026-04-02")
    public static let poseidon2Merkle = PrimitiveVersion(version: "1.2.0", updated: "2026-04-03")
    public static let keccakMerkle    = PrimitiveVersion(version: "1.2.0", updated: "2026-04-02")
    public static let sha256           = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let sha256Merkle     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let blake3Merkle    = PrimitiveVersion(version: "1.1.0", updated: "2026-04-02")
    public static let incrementalMerkle = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")

    // --- MSM ---
    public static let msmBN254       = PrimitiveVersion(version: "2.1.0", updated: "2026-04-03")
    public static let msmBLS12377    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let msmSecp256k1   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let msmPallas      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let msmVesta       = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let msmEd25519     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let msmGrumpkin    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let msmBN254G2    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let msmBLS12381   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    // --- Signatures ---
    public static let eddsa          = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let bjjEdDSA       = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let blsSignature   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let schnorr        = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let batchECDSA     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let batchEd25519   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    // --- BabyJubjub ---
    public static let babyJubjub     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let pedersenBJJ    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")

    // --- NTT ---
    public static let nttBN254       = PrimitiveVersion(version: "1.3.0", updated: "2026-04-02")
    public static let nttGoldilocks  = PrimitiveVersion(version: "1.2.0", updated: "2026-04-02")
    public static let nttBabyBear    = PrimitiveVersion(version: "1.1.0", updated: "2026-04-03")
    public static let nttBLS12377    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-02")
    public static let nttStark252    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let circleNTT      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let rnsNTT         = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let mersenne31     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")

    // --- Polynomial / STARK ---
    public static let fri            = PrimitiveVersion(version: "1.3.0", updated: "2026-04-03")
    public static let friFold        = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let stir            = PrimitiveVersion(version: "2.0.0", updated: "2026-04-05")
    public static let whir           = PrimitiveVersion(version: "2.0.0", updated: "2026-04-05")
    public static let sumcheck       = PrimitiveVersion(version: "1.1.0", updated: "2026-04-02")
    public static let poly           = PrimitiveVersion(version: "1.0.0", updated: "2026-04-01")
    public static let basefold       = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let circleFRI      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let univariateSumcheck = PrimitiveVersion(version: "1.1.0", updated: "2026-04-04")

    // --- BLS12-381 ---
    public static let bls12381       = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")

    // --- Commitment Schemes ---
    public static let kzg            = PrimitiveVersion(version: "1.1.0", updated: "2026-04-03")
    public static let ipa            = PrimitiveVersion(version: "1.1.0", updated: "2026-04-03")
    public static let zeromorph      = PrimitiveVersion(version: "2.0.0", updated: "2026-04-05")
    public static let pedersenCommit = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuVectorCommit = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    // --- GKR ---
    public static let gkr            = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let dataParallel   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let structuredGKR  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let grandProductGKR = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let memoryChecking  = PrimitiveVersion(version: "1.1.0", updated: "2026-04-05")

    // --- Other ---
    public static let ecdsa          = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let radixSort      = PrimitiveVersion(version: "1.1.0", updated: "2026-04-02")
    public static let verkle         = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let lookup         = PrimitiveVersion(version: "1.0.0", updated: "2026-04-01")
    public static let lasso          = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let cqLookup       = PrimitiveVersion(version: "2.0.0", updated: "2026-04-05")
    public static let plookup        = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let unifiedLookup  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let constraint     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let gpuConstraintEval = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let fusedNTTConstraint = PrimitiveVersion(version: "2.0.0", updated: "2026-04-05")
    public static let witness        = PrimitiveVersion(version: "2.0.0", updated: "2026-04-04")
    public static let transcript     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let serialization  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let circleSTARK    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let babyBearSTARK  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let folding        = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let brakedown      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let binaryTower    = PrimitiveVersion(version: "2.0.0", updated: "2026-04-05")
    public static let latticeNTT     = PrimitiveVersion(version: "1.1.0", updated: "2026-04-05")
    public static let reedSolomon    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let kyber          = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let dilithium      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let streamVerify   = PrimitiveVersion(version: "1.1.0", updated: "2026-04-05")
    public static let unifiedVerify  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let batchVerify    = PrimitiveVersion(version: "2.0.0", updated: "2026-04-04")
    public static let marlin         = PrimitiveVersion(version: "2.0.0", updated: "2026-04-04")
    public static let varuna         = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let spartan           = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let multilinearPoly   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuMultilinear    = PrimitiveVersion(version: "2.0.0", updated: "2026-04-05")
    public static let hyraxPCS          = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let tensorCompressor  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let joltVM            = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let groth16Batch      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let proofAggregation  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let sp1Bridge         = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let parallelReduce    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let batchField        = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let prefixScan        = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let riscvExecutor     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let trustedSetup      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuWitness        = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let halo2Permutation  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let witnessSolver     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let protogalaxyDecider = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let marlinProver      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let evmPrecompile     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let customGateLib     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuParallelReduce = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let fflonk            = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let goldilocksSTARK   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let polyIdentity      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let recursiveSNARK    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let proofTranscript   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuFFT            = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let sparsePolyCommit  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuMultiPointEval = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let stark252STARK     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let verkleProofEngine = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let plonkishArith     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuCosetLDE       = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuCosetNTT       = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let plonky2Verifier   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let pedersenHashBN254 = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let airCompiler       = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let novaIVC           = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let multiScalarIPA    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let groth16GPUWitness = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let pcsComparison     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuInnerProduct   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuMerkleTree     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuHornerEval     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuBatchInverse   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuKZG            = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuGrandProduct   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuPrefixSum     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuQuotientEngine = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let interpolation     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuRLC           = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuMatrixTranspose = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuPolyComposition = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuTraceGen      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuPolyArith     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuBatchPCSVerify = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuSparsePoly    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuPermutation   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuRangeProof    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuBatchTranscript = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuCosetFFT      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuProofComposition = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuPolyDivision  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuWitnessGen    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuPlonkGate     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuSumcheckProver = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuLogUp         = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuRecursiveSNARK = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuProofAggregation = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuPlonky2      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuR1CSSolver    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuGroth16Prover = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuIPAEngine     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuZeromorph     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuKZGMultiOpen  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuLasso         = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuFRIProver     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuSTARKVerifier = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuNovaFold      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuHalo2Backend  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuBiniusTower   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuBabyBearSTARKProver = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuGoldilocksSTARKProver = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuSpartanProver = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuzkVM         = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuPoseidon2Chain = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuMerkleBatchProof = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuFieldExtension = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuCircuitOptimizer = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuAIRConstraintCompiler = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuPCSFactory   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuEVMPrecompile = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuFRIVerifier  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuPolyInterpolation = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuProofSerializer = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuWitnessReduction = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuGrandProductProver = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuR1CSToQAP   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuBLSAggregate = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuCommitmentBatch = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuNovaDecider  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuPlonkLookup  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuMultilinearSumcheck = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuVerkleTree   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuSTARKTraceLDE = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuConstraintCompiler = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuRecursiveComposition = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuGroth16VK    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    // --- Wave 12 ---
    public static let gpuPlonkCopyConstraint = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuSTARKDeepComposition = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuJoltSubtable = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuKZGSetup     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuAIRTraceValidator = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuPedersenChain = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuBabyBearExtension = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let gpuGoldilocksExtension = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    /// Print all primitive versions
    public static func printAll() {
        let entries: [(String, PrimitiveVersion)] = [
            ("Poseidon2",         poseidon2),
            ("Poseidon2 M31",    poseidon2M31),
            ("Poseidon2 BB",     poseidon2BabyBear),
            ("Keccak-256",        keccak256),
            ("SHA-256",            sha256),
            ("Blake3",            blake3),
            ("Poseidon2 Merkle",  poseidon2Merkle),
            ("Keccak Merkle",     keccakMerkle),
            ("SHA-256 Merkle",     sha256Merkle),
            ("Blake3 Merkle",     blake3Merkle),
            ("MSM BN254",         msmBN254),
            ("MSM BLS12-377",     msmBLS12377),
            ("MSM secp256k1",     msmSecp256k1),
            ("MSM Pallas",        msmPallas),
            ("MSM Vesta",         msmVesta),
            ("MSM Ed25519",       msmEd25519),
            ("MSM Grumpkin",      msmGrumpkin),
            ("MSM BN254 G2",     msmBN254G2),
            ("MSM BLS12-381",    msmBLS12381),
            ("EdDSA Ed25519",    eddsa),
            ("BLS Signature",    blsSignature),
            ("Schnorr BIP340",   schnorr),
            ("Batch ECDSA",      batchECDSA),
            ("Batch Ed25519",    batchEd25519),
            ("NTT BN254",         nttBN254),
            ("NTT Goldilocks",    nttGoldilocks),
            ("NTT BabyBear",      nttBabyBear),
            ("NTT BLS12-377",     nttBLS12377),
            ("NTT Stark252",      nttStark252),
            ("Circle NTT M31",   circleNTT),
            ("RNS NTT (HE)",    rnsNTT),
            ("Mersenne31",       mersenne31),
            ("FRI",               fri),
            ("FRI Fold Engine",   friFold),
            ("STIR",              stir),
            ("Sumcheck",          sumcheck),
            ("Univ. Sumcheck",   univariateSumcheck),
            ("Polynomial Ops",    poly),
            ("KZG",               kzg),
            ("IPA",               ipa),
            ("Zeromorph",         zeromorph),
            ("GPU Vec Commit",   gpuVectorCommit),
            ("ECDSA",             ecdsa),
            ("Radix Sort",        radixSort),
            ("Verkle Tree",       verkle),
            ("Lookup (LogUp)",    lookup),
            ("Lasso Lookup",      lasso),
            ("cq Lookup",         cqLookup),
            ("Plookup",           plookup),
            ("Unified Lookup",    unifiedLookup),
            ("Constraint IR",    constraint),
            ("GPU Constr Eval",  gpuConstraintEval),
            ("Fused NTT+Constr", fusedNTTConstraint),
            ("Witness Trace",    witness),
            ("Transcript",       transcript),
            ("Serialization",    serialization),
            ("Basefold",         basefold),
            ("Brakedown",        brakedown),
            ("BLS12-381",        bls12381),
            ("Circle STARK",     circleSTARK),
            ("Circle FRI",       circleFRI),
            ("HyperNova Fold",   folding),
            ("GKR",              gkr),
            ("Spartan",          spartan),
            ("Reed-Solomon",     reedSolomon),
            ("Lattice NTT",      latticeNTT),
            ("Kyber KEM",        kyber),
            ("Dilithium Sig",    dilithium),
            ("Jolt VM",          joltVM),
            ("Groth16 Batch",    groth16Batch),
            ("Proof Aggregation", proofAggregation),
            ("Marlin",           marlin),
            ("Varuna",           varuna),
            ("Unified Verify",   unifiedVerify),
            ("GPU Merkle Tree",  gpuMerkleTree),
            ("GPU Horner Eval",  gpuHornerEval),
            ("GPU Batch Inv",    gpuBatchInverse),
            ("GPU KZG",          gpuKZG),
            ("GPU Grand Prod",   gpuGrandProduct),
            ("GPU Prefix Sum",   gpuPrefixSum),
            ("GPU Quotient Eng", gpuQuotientEngine),
            ("GPU Coset NTT",    gpuCosetNTT),
            ("Interpolation",    interpolation),
            ("GPU RLC Engine",   gpuRLC),
            ("GPU Transpose",    gpuMatrixTranspose),
            ("GPU Poly Compose", gpuPolyComposition),
            ("GPU Trace Gen",    gpuTraceGen),
            ("GPU Poly Arith",   gpuPolyArith),
            ("GPU Batch PCS",    gpuBatchPCSVerify),
            ("GPU Multilinear",  gpuMultilinear),
            ("GPU Sparse Poly",  gpuSparsePoly),
            ("GPU Permutation",  gpuPermutation),
            ("GPU Range Proof",  gpuRangeProof),
            ("GPU Batch Txn",    gpuBatchTranscript),
            ("GPU Coset FFT",    gpuCosetFFT),
            ("GPU Proof Comp",   gpuProofComposition),
            ("GPU Poly Div",     gpuPolyDivision),
            ("GPU Witness Gen",  gpuWitnessGen),
            ("GPU Plonk Gate",   gpuPlonkGate),
            ("GPU Sumcheck Pv",  gpuSumcheckProver),
            ("GPU LogUp",        gpuLogUp),
            ("GPU R1CS Solver",  gpuR1CSSolver),
            ("GPU Groth16 Pv",   gpuGroth16Prover),
            ("GPU IPA Engine",   gpuIPAEngine),
            ("GPU Zeromorph",    gpuZeromorph),
            ("GPU KZG MultiOp",  gpuKZGMultiOpen),
            ("GPU Lasso",        gpuLasso),
            ("GPU FRI Prover",   gpuFRIProver),
            ("GPU STARK Verify", gpuSTARKVerifier),
            ("GPU Nova Fold",    gpuNovaFold),
            ("GPU Recursive",    gpuRecursiveSNARK),
            ("GPU Proof Agg",    gpuProofAggregation),
            ("GPU Plonky2",      gpuPlonky2),
            ("GPU Halo2",        gpuHalo2Backend),
            ("GPU Binius",       gpuBiniusTower),
            ("GPU BB STARK Pv",  gpuBabyBearSTARKProver),
            ("GPU GL STARK Pv",  gpuGoldilocksSTARKProver),
            ("GPU Spartan Pv",   gpuSpartanProver),
            ("GPU zkVM",         gpuzkVM),
            ("GPU P2 Chain",     gpuPoseidon2Chain),
            ("GPU Merkle Batch", gpuMerkleBatchProof),
            ("GPU Field Ext",    gpuFieldExtension),
            ("GPU Circuit Opt",  gpuCircuitOptimizer),
            ("GPU AIR Compile",  gpuAIRConstraintCompiler),
            ("GPU PCS Factory",  gpuPCSFactory),
            ("GPU EVM Precomp",  gpuEVMPrecompile),
            ("GPU FRI Verify",   gpuFRIVerifier),
            ("GPU Poly Interp",  gpuPolyInterpolation),
            ("GPU Proof Serial", gpuProofSerializer),
            ("GPU Witness Red",  gpuWitnessReduction),
            ("GPU Grand Prod",   gpuGrandProductProver),
            ("GPU R1CS->QAP",    gpuR1CSToQAP),
            ("GPU BLS Agg",      gpuBLSAggregate),
            ("GPU Commit Batch", gpuCommitmentBatch),
            ("GPU Nova Decide",  gpuNovaDecider),
            ("GPU Plonk Look",   gpuPlonkLookup),
            ("GPU ML Sumcheck",  gpuMultilinearSumcheck),
            ("GPU Verkle Tree",  gpuVerkleTree),
            ("GPU STARK LDE",    gpuSTARKTraceLDE),
            ("GPU Constr Comp",  gpuConstraintCompiler),
            ("GPU Recurs Comp",  gpuRecursiveComposition),
            ("GPU Groth16 VK",   gpuGroth16VK),
            ("GPU Plonk Copy",   gpuPlonkCopyConstraint),
            ("GPU STARK Deep",   gpuSTARKDeepComposition),
            ("GPU Jolt Subtbl",  gpuJoltSubtable),
            ("GPU KZG Setup",    gpuKZGSetup),
            ("GPU AIR Trace V",  gpuAIRTraceValidator),
            ("GPU Ped Chain",    gpuPedersenChain),
            ("GPU BB Ext",       gpuBabyBearExtension),
            ("GPU GL Ext",       gpuGoldilocksExtension),
        ]
        print("=== zkMetal Primitive Versions ===")
        for (name, v) in entries {
            let padded = name.padding(toLength: 18, withPad: " ", startingAt: 0)
            print("  \(padded) \(v.description)")
        }
    }
}
