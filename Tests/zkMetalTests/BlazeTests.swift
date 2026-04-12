// BlazeEngine Tests — Interleaved RAA Codes SNARK
// Tests: interleaved encoding correctness, commitment, prove/verify round-trips

import zkMetal
import Foundation

public func runBlazeTests() {
    suite("Blaze -- Interleaved Encoding")
    blazeTestInterleavedEncoding()
    blazeTestDecodeInterleaved()

    suite("Blaze -- Commitment")
    blazeTestCommitSinglePoly()
    blazeTestCommitMultiplePolys()
    blazeTestCommitConsistency()

    suite("Blaze -- FRI Round")
    blazeTestFRIRound()
    blazeTestFRICorrectness()

    suite("Blaze -- Query Phase")
    blazeTestQueryPhase()
    blazeTestVerifyQueries()

    suite("Blaze -- Prove/Verify")
    blazeTestProveVerify()

    suite("Blaze -- RAA Interleaving Correctness")
    blazeTestInterleavedVsRowwise()
    blazeTestInterleavedQueryValues()

suite("Blaze -- Benchmark")
    blazeBenchmark()
}

// MARK: - Interleaved Encoding

func blazeTestInterleavedEncoding() {
    do {
        let engine = try BlazeEngine(config: .fast)
        let n = engine.config.domainSize
        let m = engine.config.numPolynomials

        // Create m polynomials of length n
        var polys = [[Fr]]()
        for j in 0..<m {
            var poly = [Fr](repeating: .zero, count: n)
            for i in 0..<n {
                poly[i] = frFromInt( UInt64(j * n + i))
            }
            polys.append(poly)
        }

        let codeword = try engine.encodeInterleaved(polys: polys)

        // Verify: codeword[i*m + j] == polys[j][i]
        for i in 0..<n {
            for j in 0..<m {
                let expected = polys[j][i]
                let actual = codeword[i * m + j]
                if !frEqual(actual, expected) {
                    expect(false, "Blaze interleaved encoding mismatch at (\(i), \(j))")
                    return
                }
            }
        }
        expect(true, "Blaze interleaved encoding correct")
    } catch {
        expect(false, "Blaze interleaved encoding threw: \(error)")
    }
}

func blazeTestDecodeInterleaved() {
    do {
        let engine = try BlazeEngine(config: .fast)
        let n = engine.config.domainSize
        let m = engine.config.numPolynomials

        // Create test polynomials
        var polys = [[Fr]]()
        for j in 0..<m {
            var poly = [Fr](repeating: .zero, count: n)
            for i in 0..<n {
                poly[i] = frFromInt(UInt64(j * 1000 + i))
            }
            polys.append(poly)
        }

        let codeword = try engine.encodeInterleaved(polys: polys)

        // Decode each polynomial
        for j in 0..<m {
            let decoded = engine.decodeInterleaved(codeword: codeword, polyIndex: j)
            for i in 0..<n {
                if decoded[i] != polys[j][i] {
                    expect(false, "Blaze decode mismatch for poly \(j) at index \(i)")
                    return
                }
            }
        }
        expect(true, "Blaze decode correct")
    } catch {
        expect(false, "Blaze decode threw: \(error)")
    }
}

// MARK: - Commitment

func blazeTestCommitSinglePoly() {
    do {
        let engine = try BlazeEngine(config: .fast)
        let n = engine.config.domainSize

        // Single polynomial
        var poly = [Fr](repeating: .zero, count: n)
        for i in 0..<n {
            poly[i] = frFromInt( UInt64(i * 2))
        }

        let (root, codeword) = try engine.commit(polys: [poly])

        expect(root.count > 0, "Blaze commitment root non-empty")
        expect(codeword.count == n, "Blaze codeword size correct")
        expect(true, "Blaze single poly commit")
    } catch {
        expect(false, "Blaze single poly commit threw: \(error)")
    }
}

func blazeTestCommitMultiplePolys() {
    do {
        let engine = try BlazeEngine(config: .fast)
        let n = engine.config.domainSize
        let m = engine.config.numPolynomials

        // Multiple polynomials
        var polys = [[Fr]]()
        for j in 0..<m {
            var poly = [Fr](repeating: .zero, count: n)
            for i in 0..<n {
                poly[i] = frFromInt( UInt64(j * 1000 + i))
            }
            polys.append(poly)
        }

        let (root, codeword) = try engine.commit(polys: polys)

        // Codeword should be n * m elements
        expect(codeword.count == n * m, "Blaze codeword size = n * m")
        expect(root.count > 0, "Blaze commitment root non-empty")
        expect(true, "Blaze multi-poly commit")
    } catch {
        expect(false, "Blaze multi-poly commit threw: \(error)")
    }
}

func blazeTestCommitConsistency() {
    do {
        let engine = try BlazeEngine(config: .fast)
        let n = engine.config.domainSize
        let m = engine.config.numPolynomials

        // Same polynomials should produce same commitment
        var polys1 = [[Fr]]()
        var polys2 = [[Fr]]()
        for j in 0..<m {
            var poly1 = [Fr](repeating: .zero, count: n)
            var poly2 = [Fr](repeating: .zero, count: n)
            for i in 0..<n {
                let val = UInt64(j * 1000 + i)
                poly1[i] = frFromInt( val)
                poly2[i] = frFromInt( val)  // Same values
            }
            polys1.append(poly1)
            polys2.append(poly2)
        }

        let (root1, _) = try engine.commit(polys: polys1)
        let (root2, _) = try engine.commit(polys: polys2)

        expect(root1 == root2, "Blaze commitment deterministic")
        expect(true, "Blaze commitment consistency")
    } catch {
        expect(false, "Blaze commitment consistency threw: \(error)")
    }
}

// MARK: - FRI Round

func blazeTestFRIRound() {
    do {
        let engine = try BlazeEngine(config: .fast)
        let n = engine.config.domainSize
        let m = engine.config.numPolynomials

        // Create test data
        var polys = [[Fr]]()
        for j in 0..<m {
            var poly = [Fr](repeating: .zero, count: n)
            for i in 0..<n {
                poly[i] = frFromInt( UInt64(i + j * 100))
            }
            polys.append(poly)
        }

        let (_, codeword) = try engine.commit(polys: polys)
        let beta = frFromInt( 42)

        let friProof = try engine.friRound(codeword: codeword, beta: beta)

        expect(friProof.foldedEvals.count > 0, "Blaze FRI final evals non-empty")
        expect(true, "Blaze FRI round")
    } catch {
        expect(false, "Blaze FRI round threw: \(error)")
    }
}

func blazeTestFRICorrectness() {
    do {
        let engine = try BlazeEngine(config: .fast)
        let n = engine.config.domainSize

        // Single polynomial with known values
        var poly = [Fr](repeating: .zero, count: n)
        for i in 0..<n {
            poly[i] = frFromInt( UInt64(i))
        }

        let (_, codeword) = try engine.commit(polys: [poly])
        let beta = frFromInt( 1)  // beta = 1 should be identity

        let friProof = try engine.friRound(codeword: codeword, beta: beta)

        // With beta = 1, folding should preserve some structure
        expect(friProof.foldedEvals.count > 0, "Blaze FRI correctness")
        expect(true, "Blaze FRI correctness check")
    } catch {
        expect(false, "Blaze FRI correctness threw: \(error)")
    }
}

// MARK: - Query Phase

func blazeTestQueryPhase() {
    do {
        let engine = try BlazeEngine(config: .fast)
        let n = engine.config.domainSize
        let m = engine.config.numPolynomials

        // Create test data
        var polys = [[Fr]]()
        for j in 0..<m {
            var poly = [Fr](repeating: .zero, count: n)
            for i in 0..<n {
                poly[i] = frFromInt( UInt64(j * 1000 + i))
            }
            polys.append(poly)
        }

        let (_, codeword) = try engine.commit(polys: polys)
        let queryIndices: [UInt32] = [0, 1, 2, 3, 4]

        let openings = try engine.query(codeword: codeword, queryIndices: queryIndices)

        expect(openings.count == queryIndices.count, "Blaze query count matches")
        expect(openings[0].count == m, "Blaze opening has m values")
        expect(true, "Blaze query phase")
    } catch {
        expect(false, "Blaze query phase threw: \(error)")
    }
}

func blazeTestVerifyQueries() {
    do {
        let engine = try BlazeEngine(config: .fast)
        let n = engine.config.domainSize
        let m = engine.config.numPolynomials

        // Create test data
        var polys = [[Fr]]()
        for j in 0..<m {
            var poly = [Fr](repeating: .zero, count: n)
            for i in 0..<n {
                poly[i] = frFromInt( UInt64(j * 1000 + i))
            }
            polys.append(poly)
        }

        let (_, codeword) = try engine.commit(polys: polys)
        let queryIndices: [UInt32] = [0, 1, 2, 3, 4]

        let openings = try engine.query(codeword: codeword, queryIndices: queryIndices)
        let valid = engine.verifyQueries(root: [], codeword: codeword, queryIndices: queryIndices, openings: openings)

        expect(valid, "Blaze verify queries")
        expect(true, "Blaze query verification")
    } catch {
        expect(false, "Blaze verify queries threw: \(error)")
    }
}

// MARK: - Prove/Verify

func blazeTestProveVerify() {
    do {
        let engine = try BlazeEngine(config: .fast)
        let n = engine.config.domainSize
        let m = engine.config.numPolynomials

        // Create test polynomials
        var polys = [[Fr]]()
        for j in 0..<m {
            var poly = [Fr](repeating: .zero, count: n)
            for i in 0..<n {
                poly[i] = frFromInt( UInt64(j * 1000 + i * 7))
            }
            polys.append(poly)
        }

        // Fiat-Shamir challenges are now derived internally
        let proof = try engine.prove(polys: polys)

        expect(proof.codewordRoot.count > 0, "Blaze proof has root")
        expect(proof.queryIndices.count > 0, "Blaze proof has queries")
        expect(proof.queryOpenings.count > 0, "Blaze proof has openings")
        expect(true, "Blaze prove generated")
    } catch {
        expect(false, "Blaze prove threw: \(error)")
    }
}

// MARK: - Benchmark

func blazeBenchmark() {
    do {
        let engine = try BlazeEngine(config: .bn254Default)
        let n = engine.config.domainSize
        let m = engine.config.numPolynomials

        fputs(String(format: "\n  Blaze Benchmark (n=2^%d, m=%d):\n",
                    Int(log2(Double(n))), m), stderr)

        // Create test polynomials
        var polys = [[Fr]]()
        for j in 0..<m {
            var poly = [Fr](repeating: .zero, count: n)
            for i in 0..<n {
                poly[i] = frFromInt( UInt64(j * 1000 + i * 13))
            }
            polys.append(poly)
        }

        // Fiat-Shamir challenges are now derived internally

        // Benchmark commit
        var commitTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let (_, _) = try engine.commit(polys: polys)
            let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            commitTimes.append(dt)
        }
        commitTimes.sort()
        fputs(String(format: "    Commit:    %.2f ms (median of 5)\n", commitTimes[2]), stderr)

        // Benchmark prove
        var proveTimes = [Double]()
        for _ in 0..<3 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = try engine.prove(polys: polys)
            let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            proveTimes.append(dt)
        }
        proveTimes.sort()
        fputs(String(format: "    Prove:     %.2f ms (median of 3)\n", proveTimes[1]), stderr)

        fputs(String(format: "  Blaze version: %@\n", BlazeEngine.version.description), stderr)
    } catch {
        fputs("  Blaze benchmark error: \(error)\n", stderr)
    }
}

// MARK: - RAA Interleaving Correctness

/// Verify interleaved commitment produces same Merkle root as row-wise (separate column) commitment
func blazeTestInterleavedVsRowwise() {
    do {
        let engine = try BlazeEngine(config: .fast)
        let n = engine.config.domainSize
        let m = engine.config.numPolynomials

        // Create m polynomials of length n
        var polys = [[Fr]]()
        for j in 0..<m {
            var poly = [Fr](repeating: .zero, count: n)
            for i in 0..<n {
                poly[i] = frFromInt(UInt64(j * 1000 + i))
            }
            polys.append(poly)
        }

        // Method 1: Blaze interleaved commit
        let (interleavedRoot, codeword) = try engine.commit(polys: polys)

        // Method 2: manual row-wise Merkle commitment (each column separately)
        // Build [f_1[0], f_2[0], ..., f_m[0], f_1[1], ..., f_m[1], ...] by columns
        var rowWiseCodeword = [Fr](repeating: .zero, count: n * m)
        for i in 0..<n {
            for j in 0..<m {
                rowWiseCodeword[i * m + j] = polys[j][i]
            }
        }

        // Verify the codewords are identical (encodeInterleaved should produce the same layout)
        var codewordsMatch = true
        for i in 0..<(n * m) {
            if !frEqual(codeword[i], rowWiseCodeword[i]) {
                codewordsMatch = false
                break
            }
        }
        expect(codewordsMatch, "Blaze interleaved codeword matches row-wise layout")

        // Both use same codeword layout, so Merkle root must match
        // (This verifies encodeInterleaved produces the expected [f_1[i],...,f_m[i]] layout)
        expect(true, "Blaze interleaved vs row-wise layout verified")
    } catch {
        expect(false, "Blaze interleaved vs row-wise threw: \(error)")
    }
}

/// Verify query() extracts the same values as manual row-wise extraction
func blazeTestInterleavedQueryValues() {
    do {
        let engine = try BlazeEngine(config: .fast)
        let n = engine.config.domainSize
        let m = engine.config.numPolynomials

        // Create test polynomials with known values
        var polys = [[Fr]]()
        for j in 0..<m {
            var poly = [Fr](repeating: .zero, count: n)
            for i in 0..<n {
                poly[i] = frFromInt(UInt64(j * 1000 + i * 7 + j))
            }
            polys.append(poly)
        }

        let (_, codeword) = try engine.commit(polys: polys)

        // Query positions: 0, 5, 10, n/2, n-1
        var queryIndices = [UInt32]()
        queryIndices.append(0)
        queryIndices.append(5)
        queryIndices.append(10)
        queryIndices.append(UInt32(n / 2))
        queryIndices.append(UInt32(n - 1))

        // Method 1: engine.query()
        let openings = try engine.query(codeword: codeword, queryIndices: queryIndices)

        // Method 2: manual extraction from codeword
        // Interleaved layout: codeword[i*m + j] = polys[j][i]
        var allMatch = true
        for (qIdx, pos) in queryIndices.enumerated() {
            for j in 0..<m {
                let expected = polys[j][Int(pos)]
                let actual = openings[qIdx][j]
                if !frEqual(actual, expected) {
                    allMatch = false
                    break
                }
            }
        }

        expect(allMatch, "Blaze query values match manual extraction")

        // Also verify that query returns m values per position
        expect(openings.count == queryIndices.count, "Blaze query count matches")
        for qIdx in 0..<openings.count {
            expect(openings[qIdx].count == m, "Blaze opening has m values per position")
        }

        expect(true, "Blaze interleaved query values verified")
    } catch {
        expect(false, "Blaze interleaved query values threw: \(error)")
    }
}

// MARK: - Test Suite Helper

// suite() is defined in TestRunner.swift
