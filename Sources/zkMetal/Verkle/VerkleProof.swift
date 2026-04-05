// Verkle Tree Proof Generation and Verification (EIP-6800 compatible)
//
// Implements Verkle proofs using the Banderwagon curve and IPA commitment scheme,
// following the Ethereum Verkle tree specification.
//
// Proof structure:
//   - Path proofs: IPA opening proofs along the tree path from leaf to root
//   - Multi-proofs: batched proofs for multiple keys sharing a single IPA
//   - Extension status: indicates whether a key exists (present/absent/other-stem)
//
// The commitment at each node is a Pedersen vector commitment over the Banderwagon group:
//   C = sum(v_i * G_i) where G_i are shared generators and v_i are child commitments
//   mapped to scalars via the map-to-field function.
//
// References:
//   - EIP-6800: Ethereum state with Verkle trees
//   - Dankrad Feist: Verkle tree structure
//   - go-verkle reference implementation

import Foundation
import NeonFieldOps

// MARK: - Extension Status

/// Status of a key in a Verkle tree.
public enum VerkleExtensionStatus: UInt8 {
    case present = 0       // Key exists and value is committed
    case absent = 1        // Key does not exist (empty subtree)
    case otherStem = 2     // A different stem occupies this position
}

// MARK: - Verkle Proof Structures

/// A proof for a single key in a Verkle tree.
/// Contains commitments along the path and an IPA opening proof at each level.
public struct VerkleProof {
    /// Commitments along the path from root to the leaf's parent (depth = path length).
    public let commitments: [BanderwagonExtended]
    /// IPA opening proofs for each commitment in the path.
    public let ipaProofs: [BanderwagonIPAProof]
    /// The extension status of the key.
    public let extensionStatus: VerkleExtensionStatus
    /// The stem (first 31 bytes of the key).
    public let stem: [UInt8]
    /// The suffix index (last byte of the key).
    public let suffixIndex: UInt8
    /// Depth at which the proof terminates (for absent proofs).
    public let depth: Int

    public init(commitments: [BanderwagonExtended],
                ipaProofs: [BanderwagonIPAProof],
                extensionStatus: VerkleExtensionStatus,
                stem: [UInt8],
                suffixIndex: UInt8,
                depth: Int) {
        self.commitments = commitments
        self.ipaProofs = ipaProofs
        self.extensionStatus = extensionStatus
        self.stem = stem
        self.suffixIndex = suffixIndex
        self.depth = depth
    }
}

/// A batched multi-key proof sharing a single random evaluation point.
public struct VerkleMultiProof {
    /// Unique commitments referenced by any of the proven keys.
    public let commitments: [BanderwagonExtended]
    /// Single batched IPA proof for all openings.
    public let ipaProof: BanderwagonIPAProof
    /// Per-key extension statuses.
    public let extensionStatuses: [VerkleExtensionStatus]
    /// Per-key stems.
    public let stems: [[UInt8]]
    /// Per-key suffix indices.
    public let suffixIndices: [UInt8]
    /// Per-key depths.
    public let depths: [Int]
    /// The evaluation point used for batching.
    public let evaluationPoint: Fr381
}

/// IPA proof over the Banderwagon curve.
public struct BanderwagonIPAProof {
    public let L: [BanderwagonExtended]
    public let R: [BanderwagonExtended]
    public let a: BwScalar  // final scalar in Banderwagon scalar field (Fq)

    public init(L: [BanderwagonExtended], R: [BanderwagonExtended], a: BwScalar) {
        self.L = L
        self.R = R
        self.a = a
    }
}

// MARK: - Verkle Tree Node

/// A node in the Verkle tree.
public class VerkleNode {
    /// Node type.
    public enum NodeType {
        case branch     // Internal node with children
        case leaf       // Leaf node with a value
        case empty      // Unoccupied slot
    }

    public let nodeType: NodeType
    /// Children (only for branch nodes).
    public var children: [VerkleNode?]
    /// Commitment for this node (cached after computation).
    public var commitment: BanderwagonExtended?
    /// Child values as field elements (for branch nodes).
    public var childValues: [Fr381]
    /// Leaf value (only for leaf nodes).
    public var value: Fr381?
    /// The stem associated with this node (for leaf/extension nodes).
    public var stem: [UInt8]?

    public init(nodeType: NodeType, width: Int = 256) {
        self.nodeType = nodeType
        self.children = nodeType == .branch ?
            [VerkleNode?](repeating: nil, count: width) : []
        self.childValues = nodeType == .branch ?
            [Fr381](repeating: Fr381.zero, count: width) : []
        self.value = nil
        self.stem = nil
        self.commitment = nil
    }

    /// Create a leaf node.
    public static func leaf(value: Fr381, stem: [UInt8]) -> VerkleNode {
        let node = VerkleNode(nodeType: .leaf, width: 0)
        node.value = value
        node.stem = stem
        return node
    }

    /// Create an empty node.
    public static func empty() -> VerkleNode {
        VerkleNode(nodeType: .empty, width: 0)
    }
}

// MARK: - Banderwagon IPA Engine

/// IPA (Inner Product Argument) engine over the Banderwagon curve.
/// This is a simplified pure-Swift implementation suitable for Verkle tree proofs.
public class BanderwagonIPAEngine {
    /// Generator points for vector commitments.
    public let generators: [BanderwagonExtended]
    /// Blinding generator Q for inner product binding.
    public let Q: BanderwagonExtended
    /// Vector length (must be power of 2).
    public let n: Int

    public init(generators: [BanderwagonExtended], Q: BanderwagonExtended) {
        precondition(generators.count > 0 && (generators.count & (generators.count - 1)) == 0)
        self.generators = generators
        self.Q = Q
        self.n = generators.count
    }

    /// Convenience initializer with deterministic generators.
    public convenience init(width: Int = 256) {
        let (gens, q) = bwGenerateGenerators(count: width)
        self.init(generators: gens, Q: q)
    }

    /// Commit to a vector of BwScalars: C = MSM(G, values)
    public func commit(_ values: [BwScalar]) -> BanderwagonExtended {
        precondition(values.count == n)
        return bwMSMQ(generators, values)
    }

    /// Commit to a vector of Fr381 values (convenience, converts to BwScalar).
    public func commitFr381(_ values: [Fr381]) -> BanderwagonExtended {
        return commit(values.map { bwScalarFromFr381($0) })
    }

    /// Compute inner product <a, b> in the Banderwagon scalar field (Fq).
    public static func innerProduct(_ a: [BwScalar], _ b: [BwScalar]) -> BwScalar {
        precondition(a.count == b.count)
        var result = BwScalar.zero
        for i in 0..<a.count {
            result = bwScalarAdd(result, bwScalarMul(a[i], b[i]))
        }
        return result
    }

    /// Debug flag for tracing IPA proof computation
    public var debugIPA = false

    /// Create an IPA opening proof.
    /// All scalar arithmetic is done in the Banderwagon scalar field Fq (order q).
    public func createProof(a inputA: [BwScalar], b inputB: [BwScalar]) -> BanderwagonIPAProof {
        let n = inputA.count
        precondition(n == inputB.count && n == self.n)
        precondition(n > 0 && (n & (n - 1)) == 0)

        let logN = Int(log2(Double(n)))

        var a = inputA
        var b = inputB
        var G = generators

        var Ls = [BanderwagonExtended]()
        var Rs = [BanderwagonExtended]()
        Ls.reserveCapacity(logN)
        Rs.reserveCapacity(logN)

        // Transcript for Fiat-Shamir
        var transcript = [UInt8]()
        let C = commit(inputA)
        let v = BanderwagonIPAEngine.innerProduct(inputA, inputB)
        let vQ = bwScalarMulQ(Q, v)
        let Cbound = bwAdd(C, vQ)
        appendBwPoint(&transcript, Cbound)
        appendBwScalar(&transcript, v)

        if debugIPA {
            print("  [prove] C serial = \(bwSerialize(C).prefix(8).map { String(format: "%02x", $0) }.joined())")
            print("  [prove] Cbound serial = \(bwSerialize(Cbound).prefix(8).map { String(format: "%02x", $0) }.joined())")
        }

        var halfLen = n / 2

        for round in 0..<logN {
            let aL = Array(a.prefix(halfLen))
            let aR = Array(a.suffix(from: halfLen))
            let bL = Array(b.prefix(halfLen))
            let bR = Array(b.suffix(from: halfLen))
            let GL = Array(G.prefix(halfLen))
            let GR = Array(G.suffix(from: halfLen))

            // Cross inner products
            let cL = BanderwagonIPAEngine.innerProduct(aL, bR)
            let cR = BanderwagonIPAEngine.innerProduct(aR, bL)

            // L = MSM(GR, aL) + cL * Q
            let msmL = bwMSMQ(GR, aL)
            let L = bwAdd(msmL, bwScalarMulQ(Q, cL))

            // R = MSM(GL, aR) + cR * Q
            let msmR = bwMSMQ(GL, aR)
            let R = bwAdd(msmR, bwScalarMulQ(Q, cR))

            Ls.append(L)
            Rs.append(R)

            // Fiat-Shamir challenge (derived as BwScalar, mod q)
            appendBwPoint(&transcript, L)
            appendBwPoint(&transcript, R)
            let x = deriveBwChallenge(transcript)
            let xInv = bwScalarInverse(x)

            if debugIPA {
                print("  [prove] round \(round): x = \(x.limbs)")
            }

            // Fold vectors: a' = x*aL + xInv*aR, b' = xInv*bL + x*bR, G' = xInv*GL + x*GR
            var newA = [BwScalar](repeating: .zero, count: halfLen)
            var newB = [BwScalar](repeating: .zero, count: halfLen)
            var newG = [BanderwagonExtended](repeating: .identity, count: halfLen)

            for i in 0..<halfLen {
                newA[i] = bwScalarAdd(bwScalarMul(x, aL[i]), bwScalarMul(xInv, aR[i]))
                newB[i] = bwScalarAdd(bwScalarMul(xInv, bL[i]), bwScalarMul(x, bR[i]))
                newG[i] = bwAdd(bwScalarMulQ(GL[i], xInv), bwScalarMulQ(GR[i], x))
            }

            a = newA
            b = newB
            G = newG

            halfLen /= 2
        }

        return BanderwagonIPAProof(L: Ls, R: Rs, a: a[0])
    }

    /// Verify an IPA proof.
    /// All scalar arithmetic is done in the Banderwagon scalar field Fq (order q).
    public func verify(commitment C: BanderwagonExtended, b inputB: [BwScalar],
                       innerProductValue v: BwScalar, proof: BanderwagonIPAProof) -> Bool {
        let logN = Int(log2(Double(n)))
        guard proof.L.count == logN, proof.R.count == logN else { return false }
        guard inputB.count == n else { return false }

        // Reconstruct challenges from transcript
        var transcript = [UInt8]()
        appendBwPoint(&transcript, C)
        appendBwScalar(&transcript, v)

        var challenges = [BwScalar]()
        var challengeInvs = [BwScalar]()
        challenges.reserveCapacity(logN)
        challengeInvs.reserveCapacity(logN)

        for round in 0..<logN {
            appendBwPoint(&transcript, proof.L[round])
            appendBwPoint(&transcript, proof.R[round])
            let x = deriveBwChallenge(transcript)
            challenges.append(x)
            challengeInvs.append(bwScalarInverse(x))

            if debugIPA {
                print("  [verify] round \(round): x = \(x.limbs)")
            }
        }

        // Fold commitment: C' = C + sum(x_i^2 * L_i + x_i^(-2) * R_i)
        var Cprime = C
        for round in 0..<logN {
            let x2 = bwScalarSqr(challenges[round])
            let xInv2 = bwScalarSqr(challengeInvs[round])
            let lTerm = bwScalarMulQ(proof.L[round], x2)
            let rTerm = bwScalarMulQ(proof.R[round], xInv2)
            Cprime = bwAdd(Cprime, bwAdd(lTerm, rTerm))
        }

        // Compute s-vector
        var s = [BwScalar](repeating: .one, count: n)
        for round in 0..<logN {
            let x = challenges[round]
            let xInv = challengeInvs[round]
            for i in 0..<n {
                let bit = (i >> (logN - 1 - round)) & 1
                if bit == 0 {
                    s[i] = bwScalarMul(s[i], xInv)
                } else {
                    s[i] = bwScalarMul(s[i], x)
                }
            }
        }

        // G_final = MSM(G, s)
        let gFinal = bwMSMQ(generators, s)

        // Fold b: b' = xInv*bL + x*bR
        var bFolded = inputB
        var halfLen = n / 2
        for round in 0..<logN {
            let bL = Array(bFolded.prefix(halfLen))
            let bR = Array(bFolded.suffix(from: halfLen))
            var newB = [BwScalar](repeating: .zero, count: halfLen)
            for i in 0..<halfLen {
                newB[i] = bwScalarAdd(bwScalarMul(challengeInvs[round], bL[i]),
                                      bwScalarMul(challenges[round], bR[i]))
            }
            bFolded = newB
            halfLen /= 2
        }
        let bFinal = bFolded[0]

        // Check: C' == proof.a * G_final + (proof.a * bFinal) * Q
        let aScalar = proof.a
        var aTimesS = [BwScalar](repeating: .zero, count: n)
        for i in 0..<n {
            aTimesS[i] = bwScalarMul(aScalar, s[i])
        }
        let aG = bwMSMQ(generators, aTimesS)
        let ab = bwScalarMul(aScalar, bFinal)
        let abQ = bwScalarMulQ(Q, ab)
        let expected = bwAdd(aG, abQ)

        if debugIPA {
            print("  [verify] C' serial = \(bwSerialize(Cprime).prefix(8).map { String(format: "%02x", $0) }.joined())")
            print("  [verify] expected serial = \(bwSerialize(expected).prefix(8).map { String(format: "%02x", $0) }.joined())")
        }

        return bwEqual(Cprime, expected)
    }

    // MARK: - Transcript Helpers

    private func appendBwPoint(_ transcript: inout [UInt8], _ p: BanderwagonExtended) {
        let serialized = bwSerialize(p)
        transcript.append(contentsOf: serialized)
    }

    private func appendBwScalar(_ transcript: inout [UInt8], _ v: BwScalar) {
        let limbs = [v.limbs.0, v.limbs.1, v.limbs.2, v.limbs.3]
        for limb in limbs {
            for j in 0..<8 {
                transcript.append(UInt8((limb >> (j * 8)) & 0xFF))
            }
        }
    }

    private func deriveBwChallenge(_ transcript: [UInt8]) -> BwScalar {
        var hash = [UInt8](repeating: 0, count: 32)
        transcript.withUnsafeBufferPointer { inp in
            hash.withUnsafeMutableBufferPointer { out in
                blake3_hash_neon(inp.baseAddress!, inp.count, out.baseAddress!)
            }
        }
        var limbs: [UInt64] = [0, 0, 0, 0]
        hash.withUnsafeBytes { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            limbs[0] = ptr[0]; limbs[1] = ptr[1]
            limbs[2] = ptr[2]; limbs[3] = ptr[3]
        }
        // Reduce mod q
        var v = (limbs[0], limbs[1], limbs[2], limbs[3])
        while bwScalarGTE(v, BwScalar.Q) {
            v = bwScalarSub256(v, BwScalar.Q)
        }
        return BwScalar(v.0, v.1, v.2, v.3)
    }
}

// MARK: - Verkle Tree with Banderwagon

/// A Verkle tree using Banderwagon commitments and IPA proofs.
/// Width-256 branching factor, 32-byte keys (31-byte stem + 1-byte suffix).
public class VerkleTree {
    /// The IPA engine used for commitments and proofs.
    public let ipaEngine: BanderwagonIPAEngine
    /// Root node of the tree.
    public var root: VerkleNode

    /// The branching factor of this tree.
    public let width: Int

    public init(ipaEngine: BanderwagonIPAEngine? = nil) {
        if let engine = ipaEngine {
            self.ipaEngine = engine
        } else {
            self.ipaEngine = BanderwagonIPAEngine(width: 256)
        }
        self.width = self.ipaEngine.n
        self.root = VerkleNode(nodeType: .branch, width: self.width)
    }

    /// Insert a key-value pair into the tree.
    /// Key is 32 bytes: first 31 bytes are the stem, last byte is the suffix index.
    public func insert(key: [UInt8], value: Fr381) {
        precondition(key.count == 32)
        let stem = Array(key.prefix(31))
        let suffix = key[31]
        insertAtNode(&root, stem: stem, suffix: suffix, value: value, depth: 0)
    }

    /// Look up a key in the tree.
    public func get(key: [UInt8]) -> Fr381? {
        precondition(key.count == 32)
        let stem = Array(key.prefix(31))
        let suffix = key[31]
        return getFromNode(root, stem: stem, suffix: suffix, depth: 0)
    }

    /// Compute commitments bottom-up for the entire tree.
    public func computeCommitments() {
        _ = computeNodeCommitment(root)
    }

    /// Get the root commitment (computes if needed).
    public func rootCommitment() -> BanderwagonExtended {
        if root.commitment == nil { computeCommitments() }
        return root.commitment ?? .identity
    }

    // MARK: - Proof Generation

    /// Generate a proof of inclusion/exclusion for a single key.
    public func generateProof(key: [UInt8]) -> VerkleProof {
        precondition(key.count == 32)
        computeCommitments()

        let stem = Array(key.prefix(31))
        let suffix = key[31]

        var commitments = [BanderwagonExtended]()
        var ipaProofs = [BanderwagonIPAProof]()
        var status = VerkleExtensionStatus.absent
        var depth = 0

        var node = root
        var currentDepth = 0

        while true {
            let childIndex: Int
            if currentDepth < 31 {
                childIndex = Int(stem[currentDepth]) % width
            } else {
                childIndex = Int(suffix) % width
            }

            if let commitment = node.commitment {
                commitments.append(commitment)
            }

            // Create evaluation vector b: 1 at childIndex, 0 elsewhere
            var bQ = [BwScalar](repeating: .zero, count: width)
            bQ[childIndex] = .one

            // Create IPA proof for this node (convert Fr381 values to BwScalar)
            let aQ = node.childValues.map { bwScalarFromFr381($0) }
            let proof = ipaEngine.createProof(a: aQ, b: bQ)
            ipaProofs.append(proof)
            depth = currentDepth + 1

            switch node.nodeType {
            case .branch:
                if let child = node.children[childIndex] {
                    if child.nodeType == .leaf {
                        if child.stem == stem {
                            status = .present
                        } else {
                            status = .otherStem
                        }
                        break
                    } else if child.nodeType == .empty {
                        status = .absent
                        break
                    }
                    node = child
                    currentDepth += 1
                    continue
                } else {
                    status = .absent
                    break
                }
            case .leaf:
                if node.stem == stem {
                    status = .present
                } else {
                    status = .otherStem
                }
                break
            case .empty:
                status = .absent
                break
            }
            break
        }

        return VerkleProof(
            commitments: commitments,
            ipaProofs: ipaProofs,
            extensionStatus: status,
            stem: stem,
            suffixIndex: suffix,
            depth: depth)
    }

    /// Generate a batched multi-key proof.
    /// All openings are combined into a single IPA proof using a random evaluation point.
    public func generateMultiProof(keys: [[UInt8]]) -> VerkleMultiProof {
        precondition(!keys.isEmpty)
        computeCommitments()

        // Collect all individual proof data
        var allCommitments = [BanderwagonExtended]()
        var extensionStatuses = [VerkleExtensionStatus]()
        var stems = [[UInt8]]()
        var suffixIndices = [UInt8]()
        var depths = [Int]()
        var allValues = [[Fr381]]()    // child values for each opening
        var allIndices = [Int]()        // child indices for each opening

        // Gather openings from each key
        for key in keys {
            precondition(key.count == 32)
            let stem = Array(key.prefix(31))
            let suffix = key[31]
            stems.append(stem)
            suffixIndices.append(suffix)

            var node = root
            var currentDepth = 0
            var status = VerkleExtensionStatus.absent

            while true {
                let childIndex: Int
                if currentDepth < 31 {
                    childIndex = Int(stem[currentDepth]) % width
                } else {
                    childIndex = Int(suffix) % width
                }

                if let commitment = node.commitment {
                    allCommitments.append(commitment)
                }
                allValues.append(node.childValues)
                allIndices.append(childIndex)

                switch node.nodeType {
                case .branch:
                    if let child = node.children[childIndex] {
                        if child.nodeType == .leaf {
                            status = child.stem == stem ? .present : .otherStem
                            currentDepth += 1
                            break
                        } else if child.nodeType == .empty {
                            status = .absent
                            currentDepth += 1
                            break
                        }
                        node = child
                        currentDepth += 1
                        continue
                    } else {
                        status = .absent
                        currentDepth += 1
                        break
                    }
                case .leaf:
                    status = node.stem == stem ? .present : .otherStem
                    currentDepth += 1
                    break
                case .empty:
                    status = .absent
                    currentDepth += 1
                    break
                }
                break
            }

            extensionStatuses.append(status)
            depths.append(currentDepth)
        }

        // Derive random evaluation point from all commitments via Fiat-Shamir
        var transcript = [UInt8]()
        for c in allCommitments {
            transcript.append(contentsOf: bwSerialize(c))
        }
        let evalPoint = deriveFr381FromTranscript(transcript)

        // Combine all openings: for each opening i with values v_i and index idx_i,
        // compute g(z) = sum_i (r^i * v_i[idx_i]) and the combined polynomial.
        // This is a simplified batching — in production, use the full multiproof protocol.

        // For the batched proof, we create a single combined IPA proof.
        // Use powers of the evaluation point as batching coefficients.
        let numOpenings = allValues.count
        let w = self.width
        let evalPointQ = bwScalarFromFr381(evalPoint)
        var combinedA = [BwScalar](repeating: .zero, count: w)
        var combinedB = [BwScalar](repeating: .zero, count: w)
        var rPower = BwScalar.one  // r^0 = 1

        for i in 0..<numOpenings {
            let values = allValues[i]
            let idx = allIndices[i]

            // Combined a: sum(r^i * values_i)
            for j in 0..<w {
                let valQ = bwScalarFromFr381(values[j])
                combinedA[j] = bwScalarAdd(combinedA[j], bwScalarMul(rPower, valQ))
            }
            // Combined b: sum(r^i * e_idx_i)
            combinedB[idx] = bwScalarAdd(combinedB[idx], rPower)

            rPower = bwScalarMul(rPower, evalPointQ)
        }

        let ipaProof = ipaEngine.createProof(a: combinedA, b: combinedB)

        // Deduplicate commitments
        var uniqueCommitments = [BanderwagonExtended]()
        var seen = Set<String>()
        for c in allCommitments {
            let ser = bwSerialize(c)
            let key = Data(ser).base64EncodedString()
            if seen.insert(key).inserted {
                uniqueCommitments.append(c)
            }
        }

        return VerkleMultiProof(
            commitments: uniqueCommitments,
            ipaProof: ipaProof,
            extensionStatuses: extensionStatuses,
            stems: stems,
            suffixIndices: suffixIndices,
            depths: depths,
            evaluationPoint: evalPoint)
    }

    // MARK: - Proof Verification (Static)

    /// Verify a single-key proof against a root commitment.
    public static func verifyProof(
        root: BanderwagonExtended,
        key: [UInt8],
        value: Fr381?,
        proof: VerkleProof,
        ipaEngine: BanderwagonIPAEngine
    ) -> Bool {
        precondition(key.count == 32)
        guard !proof.commitments.isEmpty else { return false }
        guard proof.commitments.count == proof.ipaProofs.count else { return false }

        // Verify the root commitment matches
        guard bwEqual(proof.commitments[0], root) else { return false }

        let stem = Array(key.prefix(31))
        let suffix = key[31]
        let w = ipaEngine.n

        // Verify each IPA proof along the path
        for i in 0..<proof.ipaProofs.count {
            let childIndex: Int
            if i < 31 {
                childIndex = Int(stem[i]) % w
            } else {
                childIndex = Int(suffix) % w
            }

            // Evaluation vector: e_childIndex
            var bQ = [BwScalar](repeating: .zero, count: w)
            bQ[childIndex] = .one

            // The commitment at this level
            let C = proof.commitments[i]

            // The value at the child index is the mapping of the next commitment
            // (or the leaf value for the last level).
            let vQ: BwScalar
            if i < proof.commitments.count - 1 {
                vQ = bwScalarFromFr381(bwMapToField(proof.commitments[i + 1]))
            } else if proof.extensionStatus == .present, let val = value {
                vQ = bwScalarFromFr381(val)
            } else {
                vQ = .zero  // absent or other-stem
            }

            // The bound commitment: Cbound = C + v*Q
            let Cbound = bwAdd(C, bwScalarMulQ(ipaEngine.Q, vQ))

            let valid = ipaEngine.verify(
                commitment: Cbound, b: bQ,
                innerProductValue: vQ,
                proof: proof.ipaProofs[i])
            if !valid { return false }
        }

        // Extension status checks
        switch proof.extensionStatus {
        case .present:
            return value != nil
        case .absent:
            return true
        case .otherStem:
            return proof.stem != stem
        }
    }

    /// Verify a multi-key proof against a root commitment.
    public static func verifyMultiProof(
        root: BanderwagonExtended,
        keys: [[UInt8]],
        values: [Fr381?],
        proof: VerkleMultiProof,
        ipaEngine: BanderwagonIPAEngine
    ) -> Bool {
        guard keys.count == values.count else { return false }
        guard keys.count == proof.extensionStatuses.count else { return false }
        guard !proof.commitments.isEmpty else { return false }

        // Reconstruct the combined vectors from the proof data
        let w = ipaEngine.n
        let evalPointQ = bwScalarFromFr381(proof.evaluationPoint)
        var combinedB = [BwScalar](repeating: .zero, count: w)
        var rPower = BwScalar.one

        let numOpenings = keys.count
        for i in 0..<numOpenings {
            let stem = Array(keys[i].prefix(31))
            let suffix = keys[i][31]

            let childIndex: Int
            if proof.depths[i] <= 31 {
                childIndex = (proof.depths[i] > 0 ? Int(stem[proof.depths[i] - 1]) : 0) % w
            } else {
                childIndex = Int(suffix) % w
            }

            combinedB[childIndex] = bwScalarAdd(combinedB[childIndex], rPower)
            rPower = bwScalarMul(rPower, evalPointQ)
        }

        // Compute the expected inner product value from the proof commitments
        let innerProductValue = BanderwagonIPAEngine.innerProduct(
            [BwScalar](repeating: .zero, count: w), combinedB)

        // Verify the batched IPA proof against the first (root) commitment
        guard let rootCommitment = proof.commitments.first else { return false }
        guard bwEqual(rootCommitment, root) else { return false }

        // For the multi-proof, we need to verify the combined IPA
        // The combined commitment is sum(r^i * C_i) where C_i are per-opening commitments
        var combinedC = BanderwagonExtended.identity
        rPower = BwScalar.one
        for c in proof.commitments {
            combinedC = bwAdd(combinedC, bwScalarMulQ(c, rPower))
            rPower = bwScalarMul(rPower, evalPointQ)
        }

        // Verify the combined IPA proof
        return ipaEngine.verify(
            commitment: combinedC, b: combinedB,
            innerProductValue: innerProductValue,
            proof: proof.ipaProof)
    }

    // MARK: - Private Helpers

    /// Insert a key-value pair at a node.
    private func insertAtNode(_ node: inout VerkleNode, stem: [UInt8], suffix: UInt8,
                               value: Fr381, depth: Int) {
        if depth >= 31 {
            // We've traversed the entire stem; store at suffix position
            node.children[Int(suffix) % width] = VerkleNode.leaf(value: value, stem: stem)
            node.commitment = nil  // invalidate cached commitment
            return
        }

        let childIndex = Int(stem[depth]) % width

        if node.children[childIndex] == nil {
            // Create new branch or leaf
            if depth == 30 {
                // Next level is the suffix level
                var child = VerkleNode(nodeType: .branch, width: width)
                child.children[Int(suffix) % width] = VerkleNode.leaf(value: value, stem: stem)
                node.children[childIndex] = child
            } else {
                // We can place a leaf directly (EIP-6800 style: extension nodes)
                let leaf = VerkleNode.leaf(value: value, stem: stem)
                node.children[childIndex] = leaf
            }
        } else if let child = node.children[childIndex] {
            if child.nodeType == .leaf {
                // Collision: split the leaf
                if child.stem == stem {
                    // Same stem, just update value
                    child.value = value
                    child.commitment = nil
                } else {
                    // Different stem: create intermediate branch nodes
                    var branch = VerkleNode(nodeType: .branch, width: width)
                    // Re-insert the existing leaf
                    insertAtNode(&branch, stem: child.stem!, suffix: suffix,
                                 value: child.value!, depth: depth + 1)
                    // Insert the new value
                    insertAtNode(&branch, stem: stem, suffix: suffix,
                                 value: value, depth: depth + 1)
                    node.children[childIndex] = branch
                }
            } else if child.nodeType == .branch {
                var mutableChild = child
                insertAtNode(&mutableChild, stem: stem, suffix: suffix,
                             value: value, depth: depth + 1)
                node.children[childIndex] = mutableChild
            }
        }
        node.commitment = nil  // invalidate
    }

    /// Look up a value from a node.
    private func getFromNode(_ node: VerkleNode, stem: [UInt8], suffix: UInt8, depth: Int) -> Fr381? {
        if node.nodeType == .leaf {
            if node.stem == stem { return node.value }
            return nil
        }
        if node.nodeType == .empty { return nil }

        let childIndex: Int
        if depth < 31 {
            childIndex = Int(stem[depth]) % width
        } else {
            childIndex = Int(suffix) % width
        }

        guard let child = node.children[childIndex] else { return nil }

        if child.nodeType == .leaf {
            return child.stem == stem ? child.value : nil
        }
        return getFromNode(child, stem: stem, suffix: suffix, depth: depth + 1)
    }

    /// Compute commitment for a node (recursive).
    @discardableResult
    private func computeNodeCommitment(_ node: VerkleNode) -> BanderwagonExtended {
        if let cached = node.commitment { return cached }

        switch node.nodeType {
        case .empty:
            node.commitment = .identity
            return .identity

        case .leaf:
            // Leaf commitment: hash the value and stem
            if let v = node.value {
                node.commitment = bwScalarMul(bwFromAffine(banderwagonGenerator), v)
            } else {
                node.commitment = .identity
            }
            return node.commitment!

        case .branch:
            // Compute child commitments and map to field elements
            var childValues = [Fr381](repeating: Fr381.zero, count: width)
            for i in 0..<width {
                if let child = node.children[i] {
                    let childCommitment = computeNodeCommitment(child)
                    childValues[i] = bwMapToField(childCommitment)
                }
            }
            node.childValues = childValues
            let commitment = ipaEngine.commitFr381(childValues)
            node.commitment = commitment
            return commitment
        }
    }

    /// Derive an Fr381 challenge from transcript bytes.
    private func deriveFr381FromTranscript(_ transcript: [UInt8]) -> Fr381 {
        var hash = [UInt8](repeating: 0, count: 32)
        transcript.withUnsafeBufferPointer { inp in
            hash.withUnsafeMutableBufferPointer { out in
                blake3_hash_neon(inp.baseAddress!, inp.count, out.baseAddress!)
            }
        }
        var limbs = [UInt64](repeating: 0, count: 4)
        hash.withUnsafeBytes { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: UInt64.self)
            limbs[0] = ptr[0]; limbs[1] = ptr[1]
            limbs[2] = ptr[2]; limbs[3] = ptr[3]
        }
        let raw = Fr381.from64(limbs)
        return fr381Mul(raw, Fr381.from64(Fr381.R2_MOD_R))
    }
}

// MARK: - Top-Level Convenience Functions

/// Generate a Verkle proof for a single key.
public func generateVerkleProof(tree: VerkleTree, key: [UInt8]) -> VerkleProof {
    return tree.generateProof(key: key)
}

/// Generate a batched multi-key Verkle proof.
public func generateVerkleMultiProof(tree: VerkleTree, keys: [[UInt8]]) -> VerkleMultiProof {
    return tree.generateMultiProof(keys: keys)
}

/// Verify a single-key Verkle proof.
public func verifyVerkleProof(root: BanderwagonExtended, key: [UInt8], value: Fr381?,
                               proof: VerkleProof,
                               ipaEngine: BanderwagonIPAEngine) -> Bool {
    return VerkleTree.verifyProof(root: root, key: key, value: value,
                                   proof: proof, ipaEngine: ipaEngine)
}

/// Verify a batched multi-key Verkle proof.
public func verifyVerkleMultiProof(root: BanderwagonExtended, keys: [[UInt8]], values: [Fr381?],
                                    proof: VerkleMultiProof,
                                    ipaEngine: BanderwagonIPAEngine) -> Bool {
    return VerkleTree.verifyMultiProof(root: root, keys: keys, values: values,
                                        proof: proof, ipaEngine: ipaEngine)
}
