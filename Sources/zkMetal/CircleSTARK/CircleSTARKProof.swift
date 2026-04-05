// Circle STARK Proof — self-contained proof data structures for serialization and verification
//
// The proof contains:
// 1. Trace commitments (Merkle roots of LDE columns)
// 2. Composition polynomial commitment
// 3. FRI low-degree proof
// 4. Query responses (openings at random points)
//
// All data is in terms of M31 field elements and byte arrays (Merkle hashes).

import Foundation

// Note: CircleSTARKProof, CircleSTARKQueryResponse, CircleFRIProofData, and CircleFRIRound
// are defined in CircleSTARKProver.swift. This file provides additional proof utilities.

// MARK: - Proof Size Estimation

extension CircleSTARKProof {
    /// Estimated proof size in bytes
    public var estimatedSizeBytes: Int {
        var size = 0
        // Trace commitments: 32 bytes each
        size += traceCommitments.count * 32
        // Composition commitment: 32 bytes
        size += 32
        // FRI rounds
        for round in friProof.rounds {
            size += 32 // commitment
            for (_, _, path) in round.queryResponses {
                size += 8 // two M31 values
                size += path.count * 32 // Merkle path
            }
        }
        size += 4 // final FRI value
        // Query responses
        for qr in queryResponses {
            size += qr.traceValues.count * 4 // M31 values
            for path in qr.tracePaths {
                size += path.count * 32
            }
            size += 4 // composition value
            size += qr.compositionPath.count * 32
        }
        return size
    }

    /// Human-readable proof size
    public var proofSizeDescription: String {
        let bytes = estimatedSizeBytes
        if bytes < 1024 {
            return "\(bytes) B"
        } else if bytes < 1024 * 1024 {
            return String(format: "%.1f KiB", Double(bytes) / 1024.0)
        } else {
            return String(format: "%.1f MiB", Double(bytes) / (1024.0 * 1024.0))
        }
    }
}

// MARK: - Proof Serialization (compact binary format)

extension CircleSTARKProof {
    /// Serialize proof to bytes for transmission
    public func serialize() -> [UInt8] {
        var data = [UInt8]()

        // Header: magic + version
        data.append(contentsOf: [0x43, 0x53, 0x54, 0x4B]) // "CSTK"
        data.append(contentsOf: withUnsafeBytes(of: UInt32(1)) { Array($0) }) // version 1

        // Metadata
        data.append(contentsOf: withUnsafeBytes(of: UInt32(traceLength)) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(numColumns)) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(logBlowup)) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: alpha.v) { Array($0) })

        // Trace commitments
        data.append(contentsOf: withUnsafeBytes(of: UInt32(traceCommitments.count)) { Array($0) })
        for root in traceCommitments {
            data.append(contentsOf: root)
        }

        // Composition commitment
        data.append(contentsOf: compositionCommitment)

        // FRI proof
        data.append(contentsOf: withUnsafeBytes(of: UInt32(friProof.rounds.count)) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: friProof.finalValue.v) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(friProof.queryIndices.count)) { Array($0) })
        for qi in friProof.queryIndices {
            data.append(contentsOf: withUnsafeBytes(of: UInt32(qi)) { Array($0) })
        }

        for round in friProof.rounds {
            data.append(contentsOf: round.commitment)
            data.append(contentsOf: withUnsafeBytes(of: UInt32(round.queryResponses.count)) { Array($0) })
            for (v0, v1, path) in round.queryResponses {
                data.append(contentsOf: withUnsafeBytes(of: v0.v) { Array($0) })
                data.append(contentsOf: withUnsafeBytes(of: v1.v) { Array($0) })
                data.append(contentsOf: withUnsafeBytes(of: UInt32(path.count)) { Array($0) })
                for node in path {
                    data.append(contentsOf: node)
                }
            }
        }

        // Query responses
        data.append(contentsOf: withUnsafeBytes(of: UInt32(queryResponses.count)) { Array($0) })
        for qr in queryResponses {
            data.append(contentsOf: withUnsafeBytes(of: UInt32(qr.queryIndex)) { Array($0) })
            for tv in qr.traceValues {
                data.append(contentsOf: withUnsafeBytes(of: tv.v) { Array($0) })
            }
            for path in qr.tracePaths {
                data.append(contentsOf: withUnsafeBytes(of: UInt32(path.count)) { Array($0) })
                for node in path { data.append(contentsOf: node) }
            }
            data.append(contentsOf: withUnsafeBytes(of: qr.compositionValue.v) { Array($0) })
            data.append(contentsOf: withUnsafeBytes(of: UInt32(qr.compositionPath.count)) { Array($0) })
            for node in qr.compositionPath { data.append(contentsOf: node) }
        }

        return data
    }
}
