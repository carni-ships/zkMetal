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

    // --- Signatures ---
    public static let eddsa          = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let bjjEdDSA       = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let blsSignature   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let schnorr        = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")

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
    public static let stir            = PrimitiveVersion(version: "1.0.0", updated: "2026-04-04")
    public static let whir           = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
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
    public static let zeromorph      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")

    // --- GKR ---
    public static let gkr            = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let dataParallel   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")

    // --- Other ---
    public static let ecdsa          = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let radixSort      = PrimitiveVersion(version: "1.1.0", updated: "2026-04-02")
    public static let verkle         = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let lookup         = PrimitiveVersion(version: "1.0.0", updated: "2026-04-01")
    public static let lasso          = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let cqLookup       = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let constraint     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let fusedNTTConstraint = PrimitiveVersion(version: "1.1.0", updated: "2026-04-04")
    public static let witness        = PrimitiveVersion(version: "2.0.0", updated: "2026-04-04")
    public static let transcript     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let serialization  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let circleSTARK    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let folding        = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let brakedown      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let binaryTower    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let latticeNTT     = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let reedSolomon    = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let kyber          = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let dilithium      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let streamVerify   = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let batchVerify    = PrimitiveVersion(version: "2.0.0", updated: "2026-04-04")
    public static let marlin         = PrimitiveVersion(version: "2.0.0", updated: "2026-04-04")
    public static let spartan           = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let tensorCompressor  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let joltVM            = PrimitiveVersion(version: "1.0.0", updated: "2026-04-03")
    public static let groth16Batch      = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")
    public static let proofAggregation  = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

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
            ("EdDSA Ed25519",    eddsa),
            ("BLS Signature",    blsSignature),
            ("Schnorr BIP340",   schnorr),
            ("NTT BN254",         nttBN254),
            ("NTT Goldilocks",    nttGoldilocks),
            ("NTT BabyBear",      nttBabyBear),
            ("NTT BLS12-377",     nttBLS12377),
            ("NTT Stark252",      nttStark252),
            ("Circle NTT M31",   circleNTT),
            ("RNS NTT (HE)",    rnsNTT),
            ("Mersenne31",       mersenne31),
            ("FRI",               fri),
            ("STIR",              stir),
            ("Sumcheck",          sumcheck),
            ("Univ. Sumcheck",   univariateSumcheck),
            ("Polynomial Ops",    poly),
            ("KZG",               kzg),
            ("IPA",               ipa),
            ("Zeromorph",         zeromorph),
            ("ECDSA",             ecdsa),
            ("Radix Sort",        radixSort),
            ("Verkle Tree",       verkle),
            ("Lookup (LogUp)",    lookup),
            ("Lasso Lookup",      lasso),
            ("cq Lookup",         cqLookup),
            ("Constraint IR",    constraint),
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
        ]
        print("=== zkMetal Primitive Versions ===")
        for (name, v) in entries {
            let padded = name.padding(toLength: 18, withPad: " ", startingAt: 0)
            print("  \(padded) \(v.description)")
        }
    }
}
