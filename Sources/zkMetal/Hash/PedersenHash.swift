// Pedersen Hash over BabyJubjub curve
//
// H(m) = sum(m_i * G_i) where G_i are fixed generator points
// Standard 4-bit window Pedersen (Zcash/Circom style):
//   Split message into 4-bit chunks, each chunk selects from a precomputed table.
//
// Generator points are derived deterministically from hash-to-curve.

import Foundation
#if canImport(CryptoKit)
import CryptoKit
#endif

// MARK: - Pedersen Hash

public class PedersenBJJ {
    /// Number of generators (supports up to numGenerators * 4 bits = numGenerators * 4 bit messages)
    public let numGenerators: Int
    /// Base generators G_0, G_1, ... (on BabyJubjub)
    public let generators: [BJJPointExtended]
    /// Precomputed tables: for each generator, [0*G, 1*G, 2*G, ..., 15*G] (16 entries per 4-bit window)
    public let tables: [[BJJPointExtended]]

    /// Initialize with a given number of generators (determines max message size)
    /// Default: 62 generators = 248 bits = enough for a BN254 Fr element
    public init(numGenerators: Int = 62) {
        self.numGenerators = numGenerators
        var gens = [BJJPointExtended]()
        var tabs = [[BJJPointExtended]]()
        gens.reserveCapacity(numGenerators)
        tabs.reserveCapacity(numGenerators)

        for i in 0..<numGenerators {
            let g = PedersenBJJ.deriveGenerator(index: i)
            let gExt = bjjPointFromAffine(g)
            gens.append(gExt)

            // Build lookup table: [0*G, 1*G, ..., 15*G]
            var table = [BJJPointExtended]()
            table.reserveCapacity(16)
            table.append(bjjPointIdentity())  // 0*G
            var acc = gExt
            table.append(acc)  // 1*G
            for _ in 2..<16 {
                acc = bjjPointAdd(acc, gExt)
                table.append(acc)
            }
            tabs.append(table)
        }

        self.generators = gens
        self.tables = tabs
    }

    /// Pedersen hash of a byte array (up to numGenerators * 4 bits / 8 = numGenerators/2 bytes)
    public func hash(_ data: [UInt8]) -> BJJPointAffine {
        let ext = hashExtended(data)
        return bjjPointToAffine(ext)
    }

    /// Hash returning extended point (avoids inversion for intermediate results)
    public func hashExtended(_ data: [UInt8]) -> BJJPointExtended {
        // Convert bytes to 4-bit chunks
        var chunks = [Int]()
        chunks.reserveCapacity(data.count * 2)
        for byte in data {
            chunks.append(Int(byte & 0x0F))
            chunks.append(Int((byte >> 4) & 0x0F))
        }

        precondition(chunks.count <= numGenerators,
                      "Message too long: \(chunks.count) nibbles > \(numGenerators) generators")

        var result = bjjPointIdentity()
        for i in 0..<chunks.count {
            let idx = chunks[i]
            if idx != 0 {
                result = bjjPointAdd(result, tables[i][idx])
            }
        }
        return result
    }

    /// Hash a single Fr element (252 bits split into 4-bit chunks = 63 chunks)
    public func hashFr(_ element: Fr) -> BJJPointAffine {
        let limbs = frToInt(element)
        var result = bjjPointIdentity()
        var chunkIdx = 0
        for i in 0..<4 {
            var word = limbs[i]
            for _ in 0..<16 {  // 16 nibbles per 64-bit word
                let nibble = Int(word & 0xF)
                if nibble != 0 && chunkIdx < numGenerators {
                    result = bjjPointAdd(result, tables[chunkIdx][nibble])
                }
                word >>= 4
                chunkIdx += 1
            }
        }
        return bjjPointToAffine(result)
    }

    /// Hash two Fr elements (Pedersen commitment / 2-to-1 compression)
    public func hashTwo(_ a: Fr, _ b: Fr) -> BJJPointAffine {
        let aLimbs = frToInt(a)
        let bLimbs = frToInt(b)
        var result = bjjPointIdentity()
        var chunkIdx = 0

        // First element
        for i in 0..<4 {
            var word = aLimbs[i]
            for _ in 0..<16 {
                let nibble = Int(word & 0xF)
                if nibble != 0 && chunkIdx < numGenerators {
                    result = bjjPointAdd(result, tables[chunkIdx][nibble])
                }
                word >>= 4
                chunkIdx += 1
            }
        }

        // Second element (continues from where first left off)
        // Need 2 * 64 nibbles = 128 generators total for two full Fr elements
        // With default 62 generators, we cover ~124 bits per element
        // For full coverage, use numGenerators >= 128
        for i in 0..<4 {
            var word = bLimbs[i]
            for _ in 0..<16 {
                let nibble = Int(word & 0xF)
                if nibble != 0 && chunkIdx < numGenerators {
                    result = bjjPointAdd(result, tables[chunkIdx][nibble])
                }
                word >>= 4
                chunkIdx += 1
            }
        }
        return bjjPointToAffine(result)
    }

    /// Derive a generator point deterministically using hash-to-curve
    /// from seed string "BabyJubjub_PedersenHash_{index}"
    public static func deriveGenerator(index: Int) -> BJJPointAffine {
        let seed = "BabyJubjub_PedersenHash_\(index)"
        var counter: UInt32 = 0

        while true {
            // Hash seed + counter to get a candidate y-coordinate
            var input = Array(seed.utf8)
            input.append(contentsOf: withUnsafeBytes(of: counter) { Array($0) })

            let hash = sha256Bytes(input)

            // Interpret hash as a field element (mod r)
            var limbs: [UInt64] = [0, 0, 0, 0]
            for i in 0..<4 {
                for j in 0..<8 {
                    let byteIdx = i * 8 + j
                    if byteIdx < hash.count {
                        limbs[i] |= UInt64(hash[byteIdx]) << (j * 8)
                    }
                }
            }
            // Reduce mod r
            let raw = Fr.from64(limbs)
            let yCandidate = frMul(raw, Fr.from64(Fr.R2_MOD_R))  // to Montgomery

            // Try to find x from curve equation: ax^2 + y^2 = 1 + dx^2y^2
            // => x^2 * (a - d*y^2) = 1 - y^2
            // => x^2 = (1 - y^2) / (a - d*y^2)
            let y2 = frSqr(yCandidate)
            let aConst = bjjA()
            let dConst = bjjD()
            let num = frSub(Fr.one, y2)  // 1 - y^2
            let den = frSub(aConst, frMul(dConst, y2))  // a - d*y^2

            if frToInt(den) == [0, 0, 0, 0] {
                counter += 1
                continue
            }

            let denInv = frInverse(den)
            let x2 = frMul(num, denInv)

            if let x = frSqrt(x2) {
                let p = BJJPointAffine(x: x, y: yCandidate)
                // Multiply by cofactor to ensure in the prime-order subgroup
                let pExt = bjjPointFromAffine(p)
                let pSubgroup = bjjPointMulInt(pExt, Int(BJJ_COFACTOR))
                if !bjjPointIsIdentity(pSubgroup) {
                    return bjjPointToAffine(pSubgroup)
                }
            }
            counter += 1
        }
    }
}

// MARK: - SHA-256 helper

func sha256Bytes(_ data: [UInt8]) -> [UInt8] {
    #if canImport(CryptoKit)
    let digest = SHA256.hash(data: data)
    return Array(digest)
    #else
    fatalError("CryptoKit not available")
    #endif
}
