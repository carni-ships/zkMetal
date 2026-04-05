// STARKProofSerializer — Compact binary serialization for STARK proofs
//
// Format:
//   [4 bytes] magic: "STRK"
//   [1 byte]  version
//   [4 bytes] traceLength (little-endian u32)
//   [4 bytes] numColumns (little-endian u32)
//   [4 bytes] logBlowup (little-endian u32)
//   [4 bytes] alpha (Bb value)
//   [4 bytes] oodPoint (Bb value)
//   [4 bytes] oodCompositionEval (Bb value)
//   [4 bytes] numTraceCommitments (u32)
//   For each trace commitment: [32 bytes] 8 x Bb
//   [32 bytes] compositionCommitment (8 x Bb)
//   [4 bytes] numOodTraceEvals (u32)
//   For each: [4 bytes] Bb
//   [4 bytes] numOodTraceNextEvals (u32)
//   For each: [4 bytes] Bb
//   FRI proof section
//   Query response section

import Foundation

// MARK: - STARK Proof Serializer

public struct STARKProofSerializer {

    public static let magic: [UInt8] = [0x53, 0x54, 0x52, 0x4B] // "STRK"
    public static let version: UInt8 = 1

    public init() {}

    // MARK: - Serialize

    /// Serialize a STARKProof to compact binary encoding.
    public func serialize(proof: STARKProof) -> [UInt8] {
        var data = [UInt8]()
        data.reserveCapacity(proof.estimatedSizeBytes + 64)

        // Header
        data.append(contentsOf: STARKProofSerializer.magic)
        data.append(STARKProofSerializer.version)

        // Metadata
        appendU32(&data, UInt32(proof.traceLength))
        appendU32(&data, UInt32(proof.numColumns))
        appendU32(&data, UInt32(proof.logBlowup))
        appendBb(&data, proof.alpha)
        appendBb(&data, proof.oodPoint)
        appendBb(&data, proof.oodCompositionEval)

        // Trace commitments
        appendU32(&data, UInt32(proof.traceCommitments.count))
        for commitment in proof.traceCommitments {
            appendBbSlice(&data, commitment)
        }

        // Composition commitment
        appendBbSlice(&data, proof.compositionCommitment)

        // OOD trace evaluations
        appendU32(&data, UInt32(proof.oodTraceEvals.count))
        for eval in proof.oodTraceEvals {
            appendBb(&data, eval)
        }

        // OOD trace next evaluations
        appendU32(&data, UInt32(proof.oodTraceNextEvals.count))
        for eval in proof.oodTraceNextEvals {
            appendBb(&data, eval)
        }

        // FRI proof
        serializeFRIProof(&data, proof.friProof)

        // Query responses
        appendU32(&data, UInt32(proof.queryResponses.count))
        for qr in proof.queryResponses {
            serializeQueryResponse(&data, qr)
        }

        return data
    }

    // MARK: - Deserialize

    /// Deserialize a STARKProof from binary data. Returns nil on parse failure.
    public func deserialize(data: [UInt8]) -> STARKProof? {
        var offset = 0

        // Verify magic
        guard data.count >= 5 else { return nil }
        guard Array(data[0..<4]) == STARKProofSerializer.magic else { return nil }
        guard data[4] == STARKProofSerializer.version else { return nil }
        offset = 5

        // Metadata
        guard let traceLength = readU32(data, &offset) else { return nil }
        guard let numColumns = readU32(data, &offset) else { return nil }
        guard let logBlowup = readU32(data, &offset) else { return nil }
        guard let alpha = readBb(data, &offset) else { return nil }
        guard let oodPoint = readBb(data, &offset) else { return nil }
        guard let oodCompositionEval = readBb(data, &offset) else { return nil }

        // Trace commitments
        guard let numCommitments = readU32(data, &offset) else { return nil }
        var traceCommitments = [[Bb]]()
        for _ in 0..<numCommitments {
            guard let commitment = readBbSlice(data, &offset, count: 8) else { return nil }
            traceCommitments.append(commitment)
        }

        // Composition commitment
        guard let compositionCommitment = readBbSlice(data, &offset, count: 8) else { return nil }

        // OOD trace evals
        guard let numOodEvals = readU32(data, &offset) else { return nil }
        var oodTraceEvals = [Bb]()
        for _ in 0..<numOodEvals {
            guard let eval = readBb(data, &offset) else { return nil }
            oodTraceEvals.append(eval)
        }

        // OOD trace next evals
        guard let numOodNextEvals = readU32(data, &offset) else { return nil }
        var oodTraceNextEvals = [Bb]()
        for _ in 0..<numOodNextEvals {
            guard let eval = readBb(data, &offset) else { return nil }
            oodTraceNextEvals.append(eval)
        }

        // FRI proof
        guard let friProof = deserializeFRIProof(data, &offset) else { return nil }

        // Query responses
        guard let numQueries = readU32(data, &offset) else { return nil }
        var queryResponses = [BabyBearSTARKQueryResponse]()
        for _ in 0..<numQueries {
            guard let qr = deserializeQueryResponse(data, &offset) else { return nil }
            queryResponses.append(qr)
        }

        return STARKProof(
            traceCommitments: traceCommitments,
            compositionCommitment: compositionCommitment,
            friProof: friProof,
            queryResponses: queryResponses,
            oodPoint: oodPoint,
            oodTraceEvals: oodTraceEvals,
            oodTraceNextEvals: oodTraceNextEvals,
            oodCompositionEval: oodCompositionEval,
            alpha: alpha,
            traceLength: Int(traceLength),
            numColumns: Int(numColumns),
            logBlowup: Int(logBlowup)
        )
    }

    // MARK: - Proof Size Reporting

    /// Report proof size breakdown.
    public func proofSizeReport(proof: STARKProof) -> STARKProofSizeReport {
        let serialized = serialize(proof: proof)
        let totalBytes = serialized.count

        // Compute component sizes
        let headerBytes = 5 + 4 * 3 + 4 * 3 // magic+ver + metadata + alpha/ood/comp
        let commitmentBytes = 4 + proof.traceCommitments.count * 32 + 32
        let oodBytes = 4 + proof.oodTraceEvals.count * 4 + 4 + proof.oodTraceNextEvals.count * 4

        // FRI proof size estimate
        var friBytes = 4 // numRounds
        for round in proof.friProof.rounds {
            friBytes += 32 // commitment
            friBytes += 4  // numOpenings
            for (_, _, path) in round.queryOpenings {
                friBytes += 8 // value + sibling
                friBytes += 4 + path.count * 32 // path
            }
        }
        friBytes += 4 + proof.friProof.finalPoly.count * 4 // finalPoly
        friBytes += 4 + proof.friProof.queryIndices.count * 4 // queryIndices

        // Query response size
        var queryBytes = 4
        for qr in proof.queryResponses {
            queryBytes += 4 + qr.traceValues.count * 4 // values
            queryBytes += 4 // numOpenings
            for opening in qr.traceOpenings {
                queryBytes += 4 + 4 + opening.path.count * 32 // index + pathLen + path
            }
            queryBytes += 4 // compositionValue
            queryBytes += 4 + 4 + qr.compositionOpening.path.count * 32 // comp opening
            queryBytes += 4 // queryIndex
        }

        return STARKProofSizeReport(
            totalBytes: totalBytes,
            headerBytes: headerBytes,
            commitmentBytes: commitmentBytes,
            oodEvalBytes: oodBytes,
            friProofBytes: friBytes,
            queryResponseBytes: queryBytes,
            numFRIRounds: proof.friProof.rounds.count,
            numQueries: proof.queryResponses.count,
            traceLength: proof.traceLength,
            numColumns: proof.numColumns
        )
    }

    // MARK: - FRI Proof Serialization

    private func serializeFRIProof(_ data: inout [UInt8], _ fri: BabyBearFRIProof) {
        // Rounds
        appendU32(&data, UInt32(fri.rounds.count))
        for round in fri.rounds {
            appendBbSlice(&data, round.commitment)
            appendU32(&data, UInt32(round.queryOpenings.count))
            for (value, siblingValue, path) in round.queryOpenings {
                appendBb(&data, value)
                appendBb(&data, siblingValue)
                appendU32(&data, UInt32(path.count))
                for node in path {
                    appendBbSlice(&data, node)
                }
            }
        }

        // Final polynomial
        appendU32(&data, UInt32(fri.finalPoly.count))
        for coeff in fri.finalPoly {
            appendBb(&data, coeff)
        }

        // Query indices
        appendU32(&data, UInt32(fri.queryIndices.count))
        for qi in fri.queryIndices {
            appendU32(&data, UInt32(qi))
        }
    }

    private func deserializeFRIProof(_ data: [UInt8], _ offset: inout Int) -> BabyBearFRIProof? {
        guard let numRounds = readU32(data, &offset) else { return nil }
        var rounds = [BabyBearFRIRound]()
        for _ in 0..<numRounds {
            guard let commitment = readBbSlice(data, &offset, count: 8) else { return nil }
            guard let numOpenings = readU32(data, &offset) else { return nil }
            var queryOpenings = [(value: Bb, siblingValue: Bb, path: [[Bb]])]()
            for _ in 0..<numOpenings {
                guard let value = readBb(data, &offset) else { return nil }
                guard let siblingValue = readBb(data, &offset) else { return nil }
                guard let pathLen = readU32(data, &offset) else { return nil }
                var path = [[Bb]]()
                for _ in 0..<pathLen {
                    guard let node = readBbSlice(data, &offset, count: 8) else { return nil }
                    path.append(node)
                }
                queryOpenings.append((value: value, siblingValue: siblingValue, path: path))
            }
            rounds.append(BabyBearFRIRound(commitment: commitment, queryOpenings: queryOpenings))
        }

        guard let numFinalCoeffs = readU32(data, &offset) else { return nil }
        var finalPoly = [Bb]()
        for _ in 0..<numFinalCoeffs {
            guard let coeff = readBb(data, &offset) else { return nil }
            finalPoly.append(coeff)
        }

        guard let numQueryIndices = readU32(data, &offset) else { return nil }
        var queryIndices = [Int]()
        for _ in 0..<numQueryIndices {
            guard let qi = readU32(data, &offset) else { return nil }
            queryIndices.append(Int(qi))
        }

        return BabyBearFRIProof(rounds: rounds, finalPoly: finalPoly, queryIndices: queryIndices)
    }

    // MARK: - Query Response Serialization

    private func serializeQueryResponse(_ data: inout [UInt8], _ qr: BabyBearSTARKQueryResponse) {
        // Trace values
        appendU32(&data, UInt32(qr.traceValues.count))
        for v in qr.traceValues {
            appendBb(&data, v)
        }

        // Trace openings
        appendU32(&data, UInt32(qr.traceOpenings.count))
        for opening in qr.traceOpenings {
            appendU32(&data, UInt32(opening.index))
            appendU32(&data, UInt32(opening.path.count))
            for node in opening.path {
                appendBbSlice(&data, node)
            }
        }

        // Composition value
        appendBb(&data, qr.compositionValue)

        // Composition opening
        appendU32(&data, UInt32(qr.compositionOpening.index))
        appendU32(&data, UInt32(qr.compositionOpening.path.count))
        for node in qr.compositionOpening.path {
            appendBbSlice(&data, node)
        }

        // Query index
        appendU32(&data, UInt32(qr.queryIndex))
    }

    private func deserializeQueryResponse(_ data: [UInt8], _ offset: inout Int) -> BabyBearSTARKQueryResponse? {
        // Trace values
        guard let numTraceValues = readU32(data, &offset) else { return nil }
        var traceValues = [Bb]()
        for _ in 0..<numTraceValues {
            guard let v = readBb(data, &offset) else { return nil }
            traceValues.append(v)
        }

        // Trace openings
        guard let numOpenings = readU32(data, &offset) else { return nil }
        var traceOpenings = [BbMerkleOpeningProof]()
        for _ in 0..<numOpenings {
            guard let index = readU32(data, &offset) else { return nil }
            guard let pathLen = readU32(data, &offset) else { return nil }
            var path = [[Bb]]()
            for _ in 0..<pathLen {
                guard let node = readBbSlice(data, &offset, count: 8) else { return nil }
                path.append(node)
            }
            traceOpenings.append(BbMerkleOpeningProof(path: path, index: Int(index)))
        }

        // Composition value
        guard let compositionValue = readBb(data, &offset) else { return nil }

        // Composition opening
        guard let compIndex = readU32(data, &offset) else { return nil }
        guard let compPathLen = readU32(data, &offset) else { return nil }
        var compPath = [[Bb]]()
        for _ in 0..<compPathLen {
            guard let node = readBbSlice(data, &offset, count: 8) else { return nil }
            compPath.append(node)
        }
        let compositionOpening = BbMerkleOpeningProof(path: compPath, index: Int(compIndex))

        // Query index
        guard let queryIndex = readU32(data, &offset) else { return nil }

        return BabyBearSTARKQueryResponse(
            traceValues: traceValues,
            traceOpenings: traceOpenings,
            compositionValue: compositionValue,
            compositionOpening: compositionOpening,
            queryIndex: Int(queryIndex)
        )
    }

    // MARK: - Primitive Encoding Helpers

    private func appendU32(_ data: inout [UInt8], _ value: UInt32) {
        data.append(UInt8(value & 0xFF))
        data.append(UInt8((value >> 8) & 0xFF))
        data.append(UInt8((value >> 16) & 0xFF))
        data.append(UInt8((value >> 24) & 0xFF))
    }

    private func appendBb(_ data: inout [UInt8], _ value: Bb) {
        appendU32(&data, value.v)
    }

    private func appendBbSlice(_ data: inout [UInt8], _ values: [Bb]) {
        for v in values {
            appendBb(&data, v)
        }
    }

    private func readU32(_ data: [UInt8], _ offset: inout Int) -> UInt32? {
        guard offset + 4 <= data.count else { return nil }
        let value = UInt32(data[offset]) |
                    (UInt32(data[offset + 1]) << 8) |
                    (UInt32(data[offset + 2]) << 16) |
                    (UInt32(data[offset + 3]) << 24)
        offset += 4
        return value
    }

    private func readBb(_ data: [UInt8], _ offset: inout Int) -> Bb? {
        guard let val = readU32(data, &offset) else { return nil }
        return Bb(v: val)
    }

    private func readBbSlice(_ data: [UInt8], _ offset: inout Int, count: Int) -> [Bb]? {
        var result = [Bb]()
        result.reserveCapacity(count)
        for _ in 0..<count {
            guard let val = readBb(data, &offset) else { return nil }
            result.append(val)
        }
        return result
    }
}

// MARK: - Proof Size Report

/// Breakdown of STARK proof size by component.
public struct STARKProofSizeReport: CustomStringConvertible {
    public let totalBytes: Int
    public let headerBytes: Int
    public let commitmentBytes: Int
    public let oodEvalBytes: Int
    public let friProofBytes: Int
    public let queryResponseBytes: Int
    public let numFRIRounds: Int
    public let numQueries: Int
    public let traceLength: Int
    public let numColumns: Int

    public var description: String {
        var lines = [String]()
        lines.append("STARK Proof Size Report:")
        lines.append("  Total:            \(totalBytes) bytes (\(String(format: "%.1f", Double(totalBytes) / 1024.0)) KB)")
        lines.append("  Header/metadata:  \(headerBytes) bytes")
        lines.append("  Commitments:      \(commitmentBytes) bytes")
        lines.append("  OOD evaluations:  \(oodEvalBytes) bytes")
        lines.append("  FRI proof:        \(friProofBytes) bytes (\(numFRIRounds) rounds)")
        lines.append("  Query responses:  \(queryResponseBytes) bytes (\(numQueries) queries)")
        lines.append("  Trace: \(traceLength) rows x \(numColumns) columns")
        return lines.joined(separator: "\n")
    }
}
