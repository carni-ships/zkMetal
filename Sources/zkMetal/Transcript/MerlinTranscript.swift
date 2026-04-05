// MerlinTranscript — STROBE-based construction for Merlin-compatible transcripts
//
// Implements the Merlin transcript protocol (https://merlin.cool) used by:
//   - Spartan (transparent SNARK)
//   - Bulletproofs (range proofs, inner product arguments)
//   - Dalek cryptography ecosystem
//
// Built on a simplified STROBE-128 construction over Keccak-f[1600].
//
// Reference: https://github.com/dalek-cryptography/merlin
//            https://github.com/rozbb/strobe-rs

import Foundation
import NeonFieldOps

// MARK: - TranscriptRng Support (Merlin-compatible)

/// Strobe128 state machine for Merlin transcripts.
///
/// This is a faithful port of strobe-rs's Strobe128:
///   - Security: 128 bits
///   - Rate: R = 200 - 2*(128/8) = 166 bytes
///   - Uses Keccak-f[1600] as the permutation
///
/// The key insight: STROBE uses the duplex sponge in a very specific way.
/// Each operation (AD, PRF, etc.) is framed with flags that describe the
/// data flow direction, preventing cross-operation state confusion.
private struct Strobe128 {
    /// 200-byte Keccak state as 25 UInt64 words
    var st: [UInt8]

    /// Current position in the rate portion
    var pos: UInt8

    /// Position where the current operation began
    var posBegin: UInt8

    /// Current operation flags
    var curFlags: UInt8

    static let rate: UInt8 = 166

    // Operation flag bits
    static let flagI: UInt8 = 1       // Inbound
    static let flagA: UInt8 = 1 << 1  // Application
    static let flagC: UInt8 = 1 << 2  // Cipher
    static let flagT: UInt8 = 1 << 3  // Transport
    static let flagM: UInt8 = 1 << 4  // Meta

    /// Create a new STROBE-128 instance with the given protocol label.
    init(protocolLabel: [UInt8]) {
        // Initialize 200-byte state to zero
        st = [UInt8](repeating: 0, count: 200)
        pos = 0
        posBegin = 0
        curFlags = 0

        // STROBE initialization block:
        // [1, R, 1, 0, 1, 12*8] + b"STRBa"
        // Note: the STROBE spec uses R+2 = 168, not R = 166
        let initBlock: [UInt8] = [1, Strobe128.rate &+ 2, 1, 0, 1, 96]
        let version = Array("STRBa".utf8)
        let fullInit = initBlock + version

        // XOR into state
        for i in 0..<fullInit.count {
            st[i] ^= fullInit[i]
        }

        // Permute
        runF()

        // Absorb protocol label via meta-AD
        metaAD(protocolLabel, more: false)
    }

    // MARK: - Core

    /// Run Keccak-f[1600] permutation on the state, resetting position.
    ///
    /// Before permuting, applies STROBE framing:
    ///   st[pos] ^= posBegin
    ///   st[pos+1] ^= 0x04
    ///   st[rate+1] ^= 0x80
    private mutating func runF() {
        st[Int(pos)] ^= posBegin
        st[Int(pos) + 1] ^= 0x04
        st[Int(Strobe128.rate) + 1] ^= 0x80

        // Convert to UInt64 for keccak, permute, convert back
        var state64 = [UInt64](repeating: 0, count: 25)
        st.withUnsafeBufferPointer { src in
            state64.withUnsafeMutableBufferPointer { dst in
                memcpy(dst.baseAddress!, src.baseAddress!, 200)
            }
        }
        state64.withUnsafeMutableBufferPointer { buf in
            keccak_f1600_neon(buf.baseAddress!)
        }
        state64.withUnsafeBufferPointer { src in
            st.withUnsafeMutableBufferPointer { dst in
                memcpy(dst.baseAddress!, src.baseAddress!, 200)
            }
        }

        pos = 0
        posBegin = 0
    }

    /// Absorb a single byte into the state at the current position.
    /// If position reaches the rate, triggers a permutation.
    private mutating func absorbByte(_ byte: UInt8) {
        st[Int(pos)] ^= byte
        pos += 1
        if pos == Strobe128.rate {
            runF()
        }
    }

    /// Squeeze a single byte from the state at the current position.
    /// The state byte is read and then zeroed (for forward secrecy).
    private mutating func squeezeByte() -> UInt8 {
        let byte = st[Int(pos)]
        st[Int(pos)] = 0  // zero for forward secrecy
        pos += 1
        if pos == Strobe128.rate {
            runF()
        }
        return byte
    }

    /// Overwrite the state byte at the current position.
    /// Used for KEY-like operations.
    private mutating func overwriteByte(_ byte: UInt8) {
        st[Int(pos)] = byte
        pos += 1
        if pos == Strobe128.rate {
            runF()
        }
    }

    /// Begin a new STROBE operation.
    ///
    /// If `more` is false, absorbs the operation framing: [posBegin, flags].
    /// If `more` is true, continues the current operation (no framing).
    private mutating func beginOp(_ flags: UInt8, more: Bool) {
        if more {
            curFlags = flags
            return
        }

        let oldBegin = posBegin
        posBegin = pos + 1
        curFlags = flags

        absorbByte(oldBegin)
        absorbByte(flags)
    }

    // MARK: - STROBE Operations

    /// AD (Associated Data): absorb application data.
    mutating func ad(_ data: [UInt8], more: Bool) {
        beginOp(Strobe128.flagA, more: more)
        for byte in data {
            absorbByte(byte)
        }
    }

    /// meta-AD: absorb metadata (labels, lengths).
    mutating func metaAD(_ data: [UInt8], more: Bool) {
        beginOp(Strobe128.flagM | Strobe128.flagA, more: more)
        for byte in data {
            absorbByte(byte)
        }
    }

    /// PRF: squeeze pseudorandom bytes.
    ///
    /// For PRF (flags = I|A|C), we squeeze from the state.
    /// Forces a permutation before squeezing to ensure all absorbed data
    /// affects the output, even for small inputs that don't fill the rate.
    mutating func prf(_ count: Int, more: Bool) -> [UInt8] {
        beginOp(Strobe128.flagI | Strobe128.flagA | Strobe128.flagC, more: more)
        // Force permutation to mix all absorbed data into the full state.
        // Without this, data absorbed at earlier positions wouldn't affect
        // the squeeze output for inputs smaller than the rate (166 bytes).
        runF()
        var result = [UInt8]()
        result.reserveCapacity(count)
        for _ in 0..<count {
            result.append(squeezeByte())
        }
        return result
    }

    /// KEY: absorb key material (overwrites state).
    mutating func key(_ data: [UInt8], more: Bool) {
        beginOp(Strobe128.flagA | Strobe128.flagC, more: more)
        for byte in data {
            overwriteByte(byte)
        }
    }

    /// Create an independent copy of the current state.
    func clone() -> Strobe128 {
        var copy = Strobe128(raw: ())
        copy.st = self.st
        copy.pos = self.pos
        copy.posBegin = self.posBegin
        copy.curFlags = self.curFlags
        return copy
    }

    /// Private raw init for clone.
    private init(raw: Void) {
        st = [UInt8](repeating: 0, count: 200)
        pos = 0
        posBegin = 0
        curFlags = 0
    }
}

// MARK: - MerlinTranscript

/// Merlin-compatible Fiat-Shamir transcript built on STROBE-128 over Keccak.
///
/// This transcript is wire-compatible with the Rust `merlin::Transcript` crate,
/// enabling cross-language proof generation and verification.
///
/// The Merlin protocol layers on top of STROBE-128:
///   - `append_message(label, data)` -> meta-AD(label_len) || meta-AD(label) || AD(data)
///   - `challenge_bytes(label, count)` -> meta-AD(label_len) || meta-AD(label) || PRF(count)
///
/// Usage:
///   ```
///   var t = MerlinTranscript(label: "Spartan")
///   t.appendMessage(label: "A", data: aBytes)
///   let challengeBytes = t.challengeBytes(label: "c", count: 32)
///   ```
public struct MerlinTranscript: TranscriptEngine {
    private var strobe: Strobe128
    private var _operationCount: UInt64 = 0

    public var operationCount: UInt64 { _operationCount }

    /// Create a new Merlin transcript with the given protocol label.
    ///
    /// This matches `merlin::Transcript::new(b"label")` in Rust.
    public init(label: String) {
        self.strobe = Strobe128(protocolLabel: Array("Merlin v1.0".utf8))
        appendMessageInternal(label: "dom-sep", data: Array(label.utf8))
    }

    // MARK: - Merlin-native API

    /// Append a labeled message (Merlin-compatible).
    public mutating func appendMessage(label: String, data: [UInt8]) {
        appendMessageInternal(label: label, data: data)
        _operationCount &+= 1
    }

    /// Squeeze challenge bytes (Merlin-compatible).
    public mutating func challengeBytes(label: String, count: Int) -> [UInt8] {
        let labelBytes = Array(label.utf8)
        // Match Rust merlin: meta_ad(label, false); meta_ad(dest_len_le32, true); prf(dest_len, false)
        var destLen = UInt32(count)
        let destLenBytes = withUnsafeBytes(of: &destLen) { Array($0) }
        strobe.metaAD(labelBytes, more: false)
        strobe.metaAD(destLenBytes, more: true)
        _operationCount &+= 1
        return strobe.prf(count, more: false)
    }

    // MARK: - TranscriptEngine conformance

    public mutating func appendScalar(label: String, scalar: Fr) {
        var bytes = [UInt8](repeating: 0, count: 32)
        withUnsafeBytes(of: scalar) { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt8.self)
            for i in 0..<32 { bytes[i] = ptr[i] }
        }
        appendMessage(label: label, data: bytes)
    }

    public mutating func appendPoint(label: String, point: PointProjective) {
        var affine = [UInt64](repeating: 0, count: 8)
        withUnsafeBytes(of: point) { pBuf in
            affine.withUnsafeMutableBufferPointer { aBuf in
                bn254_projective_to_affine(
                    pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    aBuf.baseAddress!
                )
            }
        }
        var bytes = [UInt8](repeating: 0, count: 64)
        for i in 0..<8 {
            for j in 0..<8 {
                bytes[i * 8 + j] = UInt8((affine[i] >> (j * 8)) & 0xFF)
            }
        }
        appendMessage(label: label, data: bytes)
    }

    public mutating func squeezeChallenge() -> Fr {
        let bytes = challengeBytes(label: "challenge", count: 32)
        return bytesToFr(bytes)
    }

    public mutating func squeezeChallenges(count: Int) -> [Fr] {
        var result = [Fr]()
        result.reserveCapacity(count)
        for i in 0..<count {
            let bytes = challengeBytes(label: "challenge_\(i)", count: 32)
            result.append(bytesToFr(bytes))
        }
        return result
    }

    public func fork(label: String) -> any TranscriptEngine {
        var child = MerlinTranscript(strobe: strobe.clone())
        child.appendMessage(label: "fork", data: Array(label.utf8))
        return child
    }

    // MARK: - Internal

    private init(strobe: Strobe128) {
        self.strobe = strobe
    }

    private mutating func appendMessageInternal(label: String, data: [UInt8]) {
        let labelBytes = Array(label.utf8)
        // Match Rust merlin: meta_ad(label, false); meta_ad(data_len_le32, true); ad(data, false)
        var dataLen = UInt32(data.count)
        let dataLenBytes = withUnsafeBytes(of: &dataLen) { Array($0) }
        strobe.metaAD(labelBytes, more: false)
        strobe.metaAD(dataLenBytes, more: true)
        strobe.ad(data, more: false)
    }

    private func bytesToFr(_ bytes: [UInt8]) -> Fr {
        var limbs: [UInt64] = [0, 0, 0, 0]
        for i in 0..<4 {
            for j in 0..<8 {
                limbs[i] |= UInt64(bytes[i * 8 + j]) << (j * 8)
            }
        }
        limbs[3] &= 0x3FFFFFFFFFFFFFFF
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}
