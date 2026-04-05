// Ethereum ABI encoding for Groth16 proofs.
// Encodes proofs for on-chain verification using Solidity verifiers
// (e.g., snarkjs-generated verifier.sol) and the EVM bn256 precompiles.
//
// ABI layout for verifyProof(uint256[2] a, uint256[2][2] b, uint256[2] c, uint256[] input):
//   - a: [a.x, a.y]                     (2 x 32 bytes)
//   - b: [[b.x.c1, b.x.c0], [b.y.c1, b.y.c0]]  (4 x 32 bytes) — NOTE: c1 before c0
//   - c: [c.x, c.y]                     (2 x 32 bytes)
//   - input: [input[0], input[1], ...]   (n x 32 bytes)
//
// All values are 256-bit big-endian unsigned integers.
// Field elements must be in standard form (NOT Montgomery).

import Foundation

// MARK: - Big-Endian 256-bit Encoding

/// Encode a 4-limb little-endian UInt64 array as a 32-byte big-endian word.
private func uint64LimbsToBE32(_ limbs: [UInt64]) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: 32)
    for i in 0..<4 {
        let limb = limbs[i]
        let base = 24 - i * 8  // big-endian: highest limb first
        bytes[base + 0] = UInt8((limb >> 56) & 0xFF)
        bytes[base + 1] = UInt8((limb >> 48) & 0xFF)
        bytes[base + 2] = UInt8((limb >> 40) & 0xFF)
        bytes[base + 3] = UInt8((limb >> 32) & 0xFF)
        bytes[base + 4] = UInt8((limb >> 24) & 0xFF)
        bytes[base + 5] = UInt8((limb >> 16) & 0xFF)
        bytes[base + 6] = UInt8((limb >> 8) & 0xFF)
        bytes[base + 7] = UInt8(limb & 0xFF)
    }
    return bytes
}

/// Decode a 32-byte big-endian word to 4-limb little-endian UInt64 array.
private func be32ToUint64Limbs(_ bytes: [UInt8]) -> [UInt64] {
    precondition(bytes.count == 32)
    var limbs = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {
        let base = 24 - i * 8
        limbs[i] = (UInt64(bytes[base]) << 56) | (UInt64(bytes[base + 1]) << 48) |
                   (UInt64(bytes[base + 2]) << 40) | (UInt64(bytes[base + 3]) << 32) |
                   (UInt64(bytes[base + 4]) << 24) | (UInt64(bytes[base + 5]) << 16) |
                   (UInt64(bytes[base + 6]) << 8) | UInt64(bytes[base + 7])
    }
    return limbs
}

// MARK: - Fp/Fr -> ABI Words

/// Encode an Fp element as a 32-byte big-endian word (converts from Montgomery form).
private func fpToABIWord(_ a: Fp) -> [UInt8] {
    uint64LimbsToBE32(fpToInt(a))
}

/// Encode an Fr element as a 32-byte big-endian word (converts from Montgomery form).
private func frToABIWord(_ a: Fr) -> [UInt8] {
    uint64LimbsToBE32(frToInt(a))
}

/// Decode a 32-byte big-endian word to an Fp in Montgomery form.
private func abiWordToFp(_ bytes: [UInt8]) -> Fp {
    let limbs = be32ToUint64Limbs(bytes)
    let raw = Fp.from64(limbs)
    return fpMul(raw, Fp.from64(Fp.R2_MOD_P))
}

/// Decode a 32-byte big-endian word to an Fr in Montgomery form.
private func abiWordToFr(_ bytes: [UInt8]) -> Fr {
    let limbs = be32ToUint64Limbs(bytes)
    let raw = Fr.from64(limbs)
    return frMul(raw, Fr.from64(Fr.R2_MOD_R))
}

// MARK: - Ethereum ABI Encoding

public struct EthereumABIEncoder {
    private init() {}

    /// Encode a Groth16 proof and public inputs for Solidity verifyProof().
    ///
    /// Returns the ABI-encoded calldata (without function selector) as bytes:
    ///   uint256[2] a, uint256[2][2] b, uint256[2] c, uint256[N] input
    ///
    /// Total size: (2 + 4 + 2 + N) * 32 bytes for fixed-size encoding.
    public static func encodeProof(proof: Groth16Proof, publicInputs: [Fr]) -> [UInt8]? {
        guard let aAff = pointToAffine(proof.a),
              let bAff = g2ToAffine(proof.b),
              let cAff = pointToAffine(proof.c) else { return nil }

        var data = [UInt8]()
        let wordCount = 2 + 4 + 2 + publicInputs.count
        data.reserveCapacity(wordCount * 32)

        // a: [x, y]
        data.append(contentsOf: fpToABIWord(aAff.x))
        data.append(contentsOf: fpToABIWord(aAff.y))

        // b: [[x.c1, x.c0], [y.c1, y.c0]]
        // NOTE: Ethereum bn256 precompile uses imaginary part first
        data.append(contentsOf: fpToABIWord(bAff.x.c1))
        data.append(contentsOf: fpToABIWord(bAff.x.c0))
        data.append(contentsOf: fpToABIWord(bAff.y.c1))
        data.append(contentsOf: fpToABIWord(bAff.y.c0))

        // c: [x, y]
        data.append(contentsOf: fpToABIWord(cAff.x))
        data.append(contentsOf: fpToABIWord(cAff.y))

        // public inputs
        for input in publicInputs {
            data.append(contentsOf: frToABIWord(input))
        }

        return data
    }

    /// Decode a Groth16 proof and public inputs from ABI-encoded bytes.
    ///
    /// - Parameter data: ABI-encoded bytes (at least 256 bytes for proof only).
    /// - Parameter numPublicInputs: Number of public inputs to decode after the proof.
    /// - Returns: Tuple of (proof, publicInputs), or nil if data is too short or invalid.
    public static func decodeProof(data: [UInt8], numPublicInputs: Int) -> (Groth16Proof, [Fr])? {
        let minSize = (2 + 4 + 2 + numPublicInputs) * 32
        guard data.count >= minSize else { return nil }

        var offset = 0

        func nextWord() -> [UInt8] {
            let word = Array(data[offset..<(offset + 32)])
            offset += 32
            return word
        }

        // a: [x, y]
        let ax = abiWordToFp(nextWord())
        let ay = abiWordToFp(nextWord())
        let aPoint = PointProjective(x: ax, y: ay, z: .one)

        // b: [[x.c1, x.c0], [y.c1, y.c0]]
        let bxc1 = abiWordToFp(nextWord())
        let bxc0 = abiWordToFp(nextWord())
        let byc1 = abiWordToFp(nextWord())
        let byc0 = abiWordToFp(nextWord())
        let bPoint = G2ProjectivePoint(
            x: Fp2(c0: bxc0, c1: bxc1),
            y: Fp2(c0: byc0, c1: byc1),
            z: .one
        )

        // c: [x, y]
        let cx = abiWordToFp(nextWord())
        let cy = abiWordToFp(nextWord())
        let cPoint = PointProjective(x: cx, y: cy, z: .one)

        // public inputs
        var inputs = [Fr]()
        inputs.reserveCapacity(numPublicInputs)
        for _ in 0..<numPublicInputs {
            inputs.append(abiWordToFr(nextWord()))
        }

        let proof = Groth16Proof(a: aPoint, b: bPoint, c: cPoint)
        return (proof, inputs)
    }

    /// Encode proof as a hex string (with 0x prefix) suitable for eth_sendTransaction.
    public static func encodeProofHex(proof: Groth16Proof, publicInputs: [Fr]) -> String? {
        guard let bytes = encodeProof(proof: proof, publicInputs: publicInputs) else {
            return nil
        }
        return "0x" + bytes.map { String(format: "%02x", $0) }.joined()
    }

    /// Encode proof with the Solidity function selector prepended.
    /// Default selector is for verifyProof(uint256[2],uint256[2][2],uint256[2],uint256[N]).
    public static func encodeCalldata(
        proof: Groth16Proof,
        publicInputs: [Fr],
        selector: [UInt8] = [0x43, 0x75, 0x3b, 0x4d]  // verifyProof(uint256[2],uint256[2][2],uint256[2],uint256[2])
    ) -> [UInt8]? {
        guard let encoded = encodeProof(proof: proof, publicInputs: publicInputs) else {
            return nil
        }
        return selector + encoded
    }
}
