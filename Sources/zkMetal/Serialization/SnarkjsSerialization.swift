// snarkjs-compatible JSON serialization for Groth16 proofs and verification keys.
// Converts internal Montgomery-form field elements to decimal strings in standard form.
// Matches the JSON schema used by snarkjs (Circom, Polygon zkEVM, Semaphore).

import Foundation

// MARK: - 256-bit Decimal Conversion

/// Convert a 4-limb little-endian UInt64 representation to a decimal string.
/// Used to serialize field elements in the format snarkjs expects.
private func uint64LimbsToDecimal(_ limbs: [UInt64]) -> String {
    // Work with big-endian bytes for division
    var bytes = [UInt8](repeating: 0, count: 32)
    for i in 0..<4 {
        for j in 0..<8 {
            bytes[31 - (i * 8 + j)] = UInt8((limbs[i] >> (j * 8)) & 0xFF)
        }
    }

    // Repeated division by 10 on big-endian byte array
    var digits = [UInt8]()
    while true {
        var allZero = true
        for b in bytes { if b != 0 { allZero = false; break } }
        if allZero { break }

        var remainder: UInt16 = 0
        for i in 0..<bytes.count {
            let val = UInt16(bytes[i]) + remainder * 256
            bytes[i] = UInt8(val / 10)
            remainder = val % 10
        }
        digits.append(UInt8(remainder))
    }

    if digits.isEmpty { return "0" }
    return String(digits.reversed().map { Character(String($0)) })
}

/// Parse a decimal string into a 4-limb little-endian UInt64 array.
private func decimalToUint64Limbs(_ s: String) -> [UInt64]? {
    // Validate: only digits
    guard !s.isEmpty, s.allSatisfy({ $0.isNumber }) else { return nil }

    // Repeated multiply-and-add in big-endian byte array
    var bytes = [UInt8](repeating: 0, count: 32)

    for ch in s {
        guard let digit = ch.wholeNumberValue else { return nil }

        // Multiply bytes by 10
        var carry: UInt16 = 0
        for i in stride(from: bytes.count - 1, through: 0, by: -1) {
            let val = UInt16(bytes[i]) * 10 + carry
            bytes[i] = UInt8(val & 0xFF)
            carry = val >> 8
        }

        // Add digit
        var addCarry: UInt16 = UInt16(digit)
        for i in stride(from: bytes.count - 1, through: 0, by: -1) {
            let val = UInt16(bytes[i]) + addCarry
            bytes[i] = UInt8(val & 0xFF)
            addCarry = val >> 8
            if addCarry == 0 { break }
        }
    }

    // Convert big-endian bytes to little-endian UInt64 limbs
    var limbs = [UInt64](repeating: 0, count: 4)
    for i in 0..<4 {
        for j in 0..<8 {
            limbs[i] |= UInt64(bytes[31 - (i * 8 + j)]) << (j * 8)
        }
    }
    return limbs
}

// MARK: - Field Element <-> Decimal String

/// Convert an Fp (Montgomery form) to a decimal string in standard form.
public func fpToDecimal(_ a: Fp) -> String {
    uint64LimbsToDecimal(fpToInt(a))
}

/// Convert a decimal string to an Fp in Montgomery form.
public func fpFromDecimal(_ s: String) -> Fp? {
    guard let limbs = decimalToUint64Limbs(s) else { return nil }
    let raw = Fp.from64(limbs)
    return fpMul(raw, Fp.from64(Fp.R2_MOD_P))
}

/// Convert an Fr (Montgomery form) to a decimal string in standard form.
public func frToDecimal(_ a: Fr) -> String {
    uint64LimbsToDecimal(frToInt(a))
}

/// Convert a decimal string to an Fr in Montgomery form.
public func frFromDecimal(_ s: String) -> Fr? {
    guard let limbs = decimalToUint64Limbs(s) else { return nil }
    let raw = Fr.from64(limbs)
    return frMul(raw, Fr.from64(Fr.R2_MOD_R))
}

// MARK: - snarkjs JSON Types (Codable)

/// snarkjs Groth16 proof JSON format.
public struct SnarkjsGroth16Proof: Codable {
    public var pi_a: [String]
    public var pi_b: [[String]]
    public var pi_c: [String]
    public var protocol_type: String

    enum CodingKeys: String, CodingKey {
        case pi_a, pi_b, pi_c
        case protocol_type = "protocol"
    }

    public init(pi_a: [String], pi_b: [[String]], pi_c: [String]) {
        self.pi_a = pi_a
        self.pi_b = pi_b
        self.pi_c = pi_c
        self.protocol_type = "groth16"
    }
}

/// snarkjs public signals format (array of decimal strings).
public struct SnarkjsPublicSignals: Codable {
    public var signals: [String]

    public init(signals: [String]) {
        self.signals = signals
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self.signals = try container.decode([String].self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(signals)
    }
}

/// snarkjs verification key JSON format.
public struct SnarkjsVerificationKey: Codable {
    public var protocol_type: String
    public var curve: String
    public var nPublic: Int
    public var vk_alpha_1: [String]
    public var vk_beta_2: [[String]]
    public var vk_gamma_2: [[String]]
    public var vk_delta_2: [[String]]
    public var IC: [[String]]

    enum CodingKeys: String, CodingKey {
        case protocol_type = "protocol"
        case curve, nPublic
        case vk_alpha_1, vk_beta_2, vk_gamma_2, vk_delta_2, IC
    }

    public init(protocol_type: String = "groth16", curve: String = "bn128",
                nPublic: Int, vk_alpha_1: [String], vk_beta_2: [[String]],
                vk_gamma_2: [[String]], vk_delta_2: [[String]], IC: [[String]]) {
        self.protocol_type = protocol_type
        self.curve = curve
        self.nPublic = nPublic
        self.vk_alpha_1 = vk_alpha_1
        self.vk_beta_2 = vk_beta_2
        self.vk_gamma_2 = vk_gamma_2
        self.vk_delta_2 = vk_delta_2
        self.IC = IC
    }
}

// MARK: - Serialization: zkMetal types -> snarkjs JSON

/// Serialize a G1 affine point to snarkjs format: ["x", "y", "1"] (decimal strings).
private func g1AffineToSnarkjs(_ p: PointAffine) -> [String] {
    [fpToDecimal(p.x), fpToDecimal(p.y), "1"]
}

/// Serialize a G2 affine point to snarkjs format:
/// [["x_c0", "x_c1"], ["y_c0", "y_c1"], ["1", "0"]]
private func g2AffineToSnarkjs(_ p: G2AffinePoint) -> [[String]] {
    [
        [fpToDecimal(p.x.c0), fpToDecimal(p.x.c1)],
        [fpToDecimal(p.y.c0), fpToDecimal(p.y.c1)],
        ["1", "0"]
    ]
}

/// Serialize a G1 projective point (converts to affine first).
/// Returns identity representation ["0", "1", "0"] if point is at infinity.
private func g1ProjectiveToSnarkjs(_ p: PointProjective) -> [String] {
    guard let aff = pointToAffine(p) else { return ["0", "1", "0"] }
    return g1AffineToSnarkjs(aff)
}

/// Serialize a G2 projective point (converts to affine first).
private func g2ProjectiveToSnarkjs(_ p: G2ProjectivePoint) -> [[String]] {
    guard let aff = g2ToAffine(p) else {
        return [["0", "0"], ["1", "0"], ["0", "0"]]
    }
    return g2AffineToSnarkjs(aff)
}

public extension Groth16Proof {
    /// Convert to snarkjs-compatible JSON representation.
    func toSnarkjs() -> SnarkjsGroth16Proof {
        SnarkjsGroth16Proof(
            pi_a: g1ProjectiveToSnarkjs(a),
            pi_b: g2ProjectiveToSnarkjs(b),
            pi_c: g1ProjectiveToSnarkjs(c)
        )
    }

    /// Encode to JSON data in snarkjs format.
    func toSnarkjsJSON(prettyPrint: Bool = true) -> Data? {
        let encoder = JSONEncoder()
        if prettyPrint { encoder.outputFormatting = [.prettyPrinted, .sortedKeys] }
        return try? encoder.encode(toSnarkjs())
    }

    /// Decode from snarkjs JSON data.
    static func fromSnarkjsJSON(_ data: Data) -> Groth16Proof? {
        guard let sj = try? JSONDecoder().decode(SnarkjsGroth16Proof.self, from: data) else {
            return nil
        }
        return fromSnarkjs(sj)
    }

    /// Convert from snarkjs representation back to internal types.
    static func fromSnarkjs(_ sj: SnarkjsGroth16Proof) -> Groth16Proof? {
        guard let a = parseG1FromSnarkjs(sj.pi_a),
              let b = parseG2FromSnarkjs(sj.pi_b),
              let c = parseG1FromSnarkjs(sj.pi_c) else { return nil }
        return Groth16Proof(a: a, b: b, c: c)
    }
}

public extension Groth16VerificationKey {
    /// Convert to snarkjs-compatible JSON representation.
    func toSnarkjs() -> SnarkjsVerificationKey {
        let icSnarkjs = ic.map { g1ProjectiveToSnarkjs($0) }
        return SnarkjsVerificationKey(
            nPublic: ic.count - 1,
            vk_alpha_1: g1ProjectiveToSnarkjs(alpha_g1),
            vk_beta_2: g2ProjectiveToSnarkjs(beta_g2),
            vk_gamma_2: g2ProjectiveToSnarkjs(gamma_g2),
            vk_delta_2: g2ProjectiveToSnarkjs(delta_g2),
            IC: icSnarkjs
        )
    }

    /// Encode to JSON data in snarkjs format.
    func toSnarkjsJSON(prettyPrint: Bool = true) -> Data? {
        let encoder = JSONEncoder()
        if prettyPrint { encoder.outputFormatting = [.prettyPrinted, .sortedKeys] }
        return try? encoder.encode(toSnarkjs())
    }

    /// Decode from snarkjs JSON data.
    static func fromSnarkjsJSON(_ data: Data) -> Groth16VerificationKey? {
        guard let sj = try? JSONDecoder().decode(SnarkjsVerificationKey.self, from: data) else {
            return nil
        }
        return fromSnarkjs(sj)
    }

    /// Convert from snarkjs representation back to internal types.
    static func fromSnarkjs(_ sj: SnarkjsVerificationKey) -> Groth16VerificationKey? {
        guard let alpha = parseG1FromSnarkjs(sj.vk_alpha_1),
              let beta = parseG2FromSnarkjs(sj.vk_beta_2),
              let gamma = parseG2FromSnarkjs(sj.vk_gamma_2),
              let delta = parseG2FromSnarkjs(sj.vk_delta_2) else { return nil }
        var icPoints = [PointProjective]()
        for icArr in sj.IC {
            guard let p = parseG1FromSnarkjs(icArr) else { return nil }
            icPoints.append(p)
        }
        return Groth16VerificationKey(
            alpha_g1: alpha, beta_g2: beta,
            gamma_g2: gamma, delta_g2: delta, ic: icPoints
        )
    }
}

// MARK: - Public Inputs Serialization

public extension Array where Element == Fr {
    /// Serialize Fr public inputs to snarkjs decimal string array.
    func toSnarkjsSignals() -> SnarkjsPublicSignals {
        SnarkjsPublicSignals(signals: self.map { frToDecimal($0) })
    }

    /// Encode public inputs to JSON data.
    func toSnarkjsSignalsJSON(prettyPrint: Bool = true) -> Data? {
        let encoder = JSONEncoder()
        if prettyPrint { encoder.outputFormatting = [.prettyPrinted] }
        return try? encoder.encode(toSnarkjsSignals())
    }
}

/// Parse an array of decimal strings back to Fr elements.
public func frArrayFromSnarkjsSignals(_ signals: SnarkjsPublicSignals) -> [Fr]? {
    var result = [Fr]()
    result.reserveCapacity(signals.signals.count)
    for s in signals.signals {
        guard let fr = frFromDecimal(s) else { return nil }
        result.append(fr)
    }
    return result
}

// MARK: - Parsing Helpers

/// Parse a G1 point from snarkjs format ["x", "y", "z"] -> PointProjective.
/// If z == "1", creates affine-style projective point. If z == "0", returns identity.
private func parseG1FromSnarkjs(_ arr: [String]) -> PointProjective? {
    guard arr.count >= 2 else { return nil }
    let zStr = arr.count >= 3 ? arr[2] : "1"

    if zStr == "0" { return pointIdentity() }

    guard let xFp = fpFromDecimal(arr[0]),
          let yFp = fpFromDecimal(arr[1]) else { return nil }

    // snarkjs always uses z=1 (affine representation)
    return PointProjective(x: xFp, y: yFp, z: .one)
}

/// Parse a G2 point from snarkjs format [["x_c0","x_c1"],["y_c0","y_c1"],["z_c0","z_c1"]].
private func parseG2FromSnarkjs(_ arr: [[String]]) -> G2ProjectivePoint? {
    guard arr.count >= 2,
          arr[0].count == 2, arr[1].count == 2 else { return nil }

    // Check for identity
    if arr.count >= 3 && arr[2].count == 2 && arr[2][0] == "0" && arr[2][1] == "0" {
        return g2Identity()
    }

    guard let xc0 = fpFromDecimal(arr[0][0]),
          let xc1 = fpFromDecimal(arr[0][1]),
          let yc0 = fpFromDecimal(arr[1][0]),
          let yc1 = fpFromDecimal(arr[1][1]) else { return nil }

    return G2ProjectivePoint(x: Fp2(c0: xc0, c1: xc1), y: Fp2(c0: yc0, c1: yc1), z: .one)
}
