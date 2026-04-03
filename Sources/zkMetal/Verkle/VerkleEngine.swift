// Verkle Tree Engine
// Implements Verkle trees with Pedersen (MSM-based) vector commitments and IPA opening proofs.
//
// Structure:
//   - Width-256 tree: each internal node commits to 256 children
//   - Commitment: C = MSM(G, children_values) where G are shared generators
//   - Opening: IPA proof that C opens to the child value at a given index
//   - Path proof: chain of IPA proofs from leaf to root
//
// References: Ethereum Verkle Tree EIP, Kuszmaul 2019

import Foundation
import Metal
import NeonFieldOps

/// A Verkle tree node commitment (a curve point)
public typealias VerkleCommitment = PointProjective

/// A single-level opening proof: proves that a commitment opens to a value at a given index
public struct VerkleOpeningProof {
    public let index: Int           // child index (0..width-1)
    public let value: Fr            // the committed value at this index
    public let ipaProof: IPAProof   // IPA proof for the opening
    public let commitment: VerkleCommitment  // the node commitment
}

/// A Verkle path proof from a leaf to the root
public struct VerklePathProof {
    public let openings: [VerkleOpeningProof]  // from leaf level up to root
}

public class VerkleEngine {
    /// Tree branching factor (children per node)
    public let width: Int
    /// IPA engine with width-many generators
    public let ipaEngine: IPAEngine
    /// Shared generators for all commitments
    public let generators: [PointAffine]
    /// Blinding point Q
    public let Q: PointAffine

    /// Create a Verkle engine with the given width (must be power of 2).
    /// Generators are created deterministically for testing.
    public init(width: Int = 256) throws {
        precondition(width > 0 && (width & (width - 1)) == 0, "Width must be power of 2")
        self.width = width
        let (gens, q) = IPAEngine.generateTestGenerators(count: width)
        self.generators = gens
        self.Q = q
        self.ipaEngine = try IPAEngine(generators: gens, Q: q)
    }

    /// Commit to a vector of field elements (length must equal width).
    /// Returns the Pedersen commitment C = MSM(G, values).
    public func commit(_ values: [Fr]) throws -> VerkleCommitment {
        precondition(values.count == width)
        return try ipaEngine.commit(values)
    }

    /// Create an opening proof for a specific index in a committed vector.
    /// Proves that commitment C opens to values[index].
    public func createOpeningProof(values: [Fr], index: Int) throws -> VerkleOpeningProof {
        precondition(values.count == width)
        precondition(index >= 0 && index < width)

        let C = try commit(values)

        // The evaluation point vector b has 1 at `index` and 0 elsewhere
        // So <a, b> = a[index] = values[index]
        var b = [Fr](repeating: Fr.zero, count: width)
        b[index] = Fr.one

        let v = values[index]  // inner product <values, b> = values[index]

        let proof = try ipaEngine.createProof(a: values, b: b)

        return VerkleOpeningProof(
            index: index,
            value: v,
            ipaProof: proof,
            commitment: C
        )
    }

    /// Verify a single opening proof.
    public func verifyOpeningProof(_ proof: VerkleOpeningProof) -> Bool {
        var b = [Fr](repeating: Fr.zero, count: width)
        b[proof.index] = Fr.one

        // Reconstruct bound commitment: C_bound = C + v*Q
        let vQ = cPointScalarMul(pointFromAffine(Q), proof.value)
        let Cbound = pointAdd(proof.commitment, vQ)

        return ipaEngine.verify(
            commitment: Cbound,
            b: b,
            innerProductValue: proof.value,
            proof: proof.ipaProof
        )
    }

    // MARK: - Tree operations

    /// Build a Verkle tree from leaf values.
    /// Returns array of levels: level[0] = leaf commitments, level[depth-1] = [root].
    /// Each level contains the commitments for that level's nodes.
    /// Leaf values are grouped into chunks of `width`.
    public func buildTree(leaves: [Fr]) throws -> (levels: [[VerkleCommitment]], leafChunks: [[Fr]]) {
        let numLeaves = leaves.count
        precondition(numLeaves > 0 && numLeaves % width == 0, "Leaf count must be multiple of width")

        // Level 0: commit each chunk of `width` leaves
        let numChunks = numLeaves / width
        var leafChunks = [[Fr]]()
        leafChunks.reserveCapacity(numChunks)
        var currentLevel = [VerkleCommitment]()
        currentLevel.reserveCapacity(numChunks)

        for i in 0..<numChunks {
            let chunk = Array(leaves[i * width ..< (i + 1) * width])
            leafChunks.append(chunk)
            let c = try commit(chunk)
            currentLevel.append(c)
        }

        var levels = [currentLevel]

        // Build upper levels until we have a single root
        while currentLevel.count > 1 {
            let numNodes = currentLevel.count
            // Pad to multiple of width if needed
            let padded = numNodes % width == 0 ? numNodes : numNodes + (width - numNodes % width)
            var childValues = [Fr](repeating: Fr.zero, count: padded)

            // Convert commitments to field elements (use x-coordinate)
            for i in 0..<numNodes {
                let aff = batchToAffine([currentLevel[i]])[0]
                childValues[i] = commitmentToFr(aff)
            }

            var nextLevel = [VerkleCommitment]()
            let nextCount = padded / width
            nextLevel.reserveCapacity(nextCount)

            for i in 0..<nextCount {
                let chunk = Array(childValues[i * width ..< (i + 1) * width])
                let c = try commit(chunk)
                nextLevel.append(c)
            }

            levels.append(nextLevel)
            currentLevel = nextLevel
        }

        return (levels: levels, leafChunks: leafChunks)
    }

    /// Create a path proof for a specific leaf index.
    /// Returns opening proofs from leaf level up to root.
    public func createPathProof(leaves: [Fr], leafIndex: Int) throws -> VerklePathProof {
        let (levels, leafChunks) = try buildTree(leaves: leaves)

        var openings = [VerkleOpeningProof]()
        var nodeIndex = leafIndex / width  // which chunk contains this leaf
        var childIndex = leafIndex % width // position within chunk

        // Level 0: open the leaf chunk
        let opening0 = try createOpeningProof(values: leafChunks[nodeIndex], index: childIndex)
        openings.append(opening0)

        // Upper levels: open each parent node
        for level in 0..<(levels.count - 1) {
            let numNodes = levels[level].count
            let padded = numNodes % width == 0 ? numNodes : numNodes + (width - numNodes % width)
            var childValues = [Fr](repeating: Fr.zero, count: padded)

            for i in 0..<numNodes {
                let aff = batchToAffine([levels[level][i]])[0]
                childValues[i] = commitmentToFr(aff)
            }

            childIndex = nodeIndex % width
            nodeIndex = nodeIndex / width

            let chunk = Array(childValues[(nodeIndex * width) ..< ((nodeIndex + 1) * width)])
            let opening = try createOpeningProof(values: chunk, index: childIndex)
            openings.append(opening)
        }

        return VerklePathProof(openings: openings)
    }

    /// Verify a path proof against a known root commitment.
    public func verifyPathProof(_ proof: VerklePathProof, root: VerkleCommitment) -> Bool {
        guard !proof.openings.isEmpty else { return false }

        // Verify each opening
        for opening in proof.openings {
            if !verifyOpeningProof(opening) { return false }
        }

        // Check that the last opening's commitment matches the root
        let lastCommitment = proof.openings.last!.commitment
        return pointEqual(lastCommitment, root)
    }

    // MARK: - Helpers

    /// Convert a commitment (curve point) to a field element for use as a child value.
    /// Uses the x-coordinate of the affine representation.
    private func commitmentToFr(_ p: PointAffine) -> Fr {
        // Interpret the Fp x-coordinate as an Fr element
        // Both are 256-bit, but moduli differ. We just reinterpret the bits.
        let xInt = fpToInt(p.x)
        let raw = Fr.from64(xInt)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))  // to Montgomery form
    }
}
