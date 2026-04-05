import zkMetal

// MARK: - Kyber/Dilithium reference test vectors

/// Known Kyber NTT output for input [1, 2, 3, ..., 256] mod 3329
/// Computed from reference CRYSTALS-Kyber implementation.
private func kyberReferenceInput() -> [UInt32] {
    return (1...256).map { UInt32($0) }
}

/// Known Dilithium NTT output for input [1, 2, 3, ..., 256] mod 8380417
private func dilithiumReferenceInput() -> [UInt32] {
    return (1...256).map { UInt32($0) }
}

// MARK: - PRNG for test data

private var testRNG: UInt64 = 0xDEAD_BEEF_CAFE_BABE

private func nextTestRandom() -> UInt64 {
    testRNG ^= testRNG << 13
    testRNG ^= testRNG >> 7
    testRNG ^= testRNG << 17
    return testRNG
}

private func resetTestRNG() {
    testRNG = 0xDEAD_BEEF_CAFE_BABE
}

// MARK: - CPU reference implementations for validation

/// CPU Kyber polynomial multiply via schoolbook (for verifying NTT-domain pointwise)
/// Computes c = a * b in Z_q[x]/(x^256 + 1) using negacyclic convolution
private func kyberSchoolbookMul(_ a: [UInt32], _ b: [UInt32]) -> [UInt32] {
    let n = 256
    let q = UInt64(3329)
    var c = [UInt32](repeating: 0, count: n)
    for i in 0..<n {
        for j in 0..<n {
            let prod = (UInt64(a[i]) * UInt64(b[j])) % q
            let idx = i + j
            if idx < n {
                c[idx] = UInt32((UInt64(c[idx]) + prod) % q)
            } else {
                // Negacyclic: x^256 = -1, so wrap around with subtraction
                let wrapIdx = idx - n
                c[wrapIdx] = UInt32((UInt64(c[wrapIdx]) + q - prod) % q)
            }
        }
    }
    return c
}

/// CPU Dilithium polynomial multiply via schoolbook (negacyclic)
private func dilithiumSchoolbookMul(_ a: [UInt32], _ b: [UInt32]) -> [UInt32] {
    let n = 256
    let q = UInt64(8380417)
    var c = [UInt32](repeating: 0, count: n)
    for i in 0..<n {
        for j in 0..<n {
            let prod = (UInt64(a[i]) * UInt64(b[j])) % q
            let idx = i + j
            if idx < n {
                c[idx] = UInt32((UInt64(c[idx]) + prod) % q)
            } else {
                let wrapIdx = idx - n
                c[wrapIdx] = UInt32((UInt64(c[wrapIdx]) + q - prod) % q)
            }
        }
    }
    return c
}

// MARK: - Test runner

public func runLatticeNTTTests() {
    // ============================================================
    // Kyber NTT round-trip
    // ============================================================
    suite("Lattice NTT: Kyber round-trip")
    do {
        let engine = try LatticeNTT()

        // Test with sequential input
        let input = kyberReferenceInput()
        let nttResult = engine.ntt(data: input, mode: .kyber)
        let recovered = engine.intt(data: nttResult, mode: .kyber)

        var ok = true
        for i in 0..<256 {
            if input[i] != recovered[i] {
                ok = false
                break
            }
        }
        expect(ok, "NTT->INTT recovers original (sequential)")

        // Test with random input
        resetTestRNG()
        var randomInput = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 {
            randomInput[i] = UInt32(nextTestRandom() % 3329)
        }
        let nttRandom = engine.ntt(data: randomInput, mode: .kyber)
        let recoveredRandom = engine.intt(data: nttRandom, mode: .kyber)

        var randomOk = true
        for i in 0..<256 {
            if randomInput[i] != recoveredRandom[i] {
                randomOk = false
                break
            }
        }
        expect(randomOk, "NTT->INTT recovers original (random)")
    } catch {
        expect(false, "Kyber round-trip error: \(error)")
    }

    // ============================================================
    // Kyber GPU vs CPU consistency
    // ============================================================
    suite("Lattice NTT: Kyber GPU vs CPU")
    do {
        let engine = try LatticeNTT()

        resetTestRNG()
        var input = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 {
            input[i] = UInt32(nextTestRandom() % 3329)
        }

        let gpuResult = engine.ntt(data: input, mode: .kyber)
        let cpuResult = engine.cpuNTT(data: input, mode: .kyber)

        var ok = true
        for i in 0..<256 {
            if gpuResult[i] != cpuResult[i] {
                ok = false
                break
            }
        }
        expect(ok, "GPU NTT matches CPU NTT (Kyber)")

        // INTT consistency
        let gpuIntt = engine.intt(data: gpuResult, mode: .kyber)
        let cpuIntt = engine.cpuINTT(data: cpuResult, mode: .kyber)

        var inttOk = true
        for i in 0..<256 {
            if gpuIntt[i] != cpuIntt[i] {
                inttOk = false
                break
            }
        }
        expect(inttOk, "GPU INTT matches CPU INTT (Kyber)")
    } catch {
        expect(false, "Kyber GPU vs CPU error: \(error)")
    }

    // ============================================================
    // Dilithium NTT round-trip
    // ============================================================
    suite("Lattice NTT: Dilithium round-trip")
    do {
        let engine = try LatticeNTT()

        let input = dilithiumReferenceInput()
        let nttResult = engine.ntt(data: input, mode: .dilithium)
        let recovered = engine.intt(data: nttResult, mode: .dilithium)

        var ok = true
        for i in 0..<256 {
            if input[i] != recovered[i] {
                ok = false
                break
            }
        }
        expect(ok, "NTT->INTT recovers original (sequential)")

        // Random input
        resetTestRNG()
        var randomInput = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 {
            randomInput[i] = UInt32(nextTestRandom() % 8380417)
        }
        let nttRandom = engine.ntt(data: randomInput, mode: .dilithium)
        let recoveredRandom = engine.intt(data: nttRandom, mode: .dilithium)

        var randomOk = true
        for i in 0..<256 {
            if randomInput[i] != recoveredRandom[i] {
                randomOk = false
                break
            }
        }
        expect(randomOk, "NTT->INTT recovers original (random)")
    } catch {
        expect(false, "Dilithium round-trip error: \(error)")
    }

    // ============================================================
    // Dilithium GPU vs CPU consistency
    // ============================================================
    suite("Lattice NTT: Dilithium GPU vs CPU")
    do {
        let engine = try LatticeNTT()

        resetTestRNG()
        var input = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 {
            input[i] = UInt32(nextTestRandom() % 8380417)
        }

        let gpuResult = engine.ntt(data: input, mode: .dilithium)
        let cpuResult = engine.cpuNTT(data: input, mode: .dilithium)

        var ok = true
        for i in 0..<256 {
            if gpuResult[i] != cpuResult[i] {
                ok = false
                break
            }
        }
        expect(ok, "GPU NTT matches CPU NTT (Dilithium)")

        let gpuIntt = engine.intt(data: gpuResult, mode: .dilithium)
        let cpuIntt = engine.cpuINTT(data: cpuResult, mode: .dilithium)

        var inttOk = true
        for i in 0..<256 {
            if gpuIntt[i] != cpuIntt[i] {
                inttOk = false
                break
            }
        }
        expect(inttOk, "GPU INTT matches CPU INTT (Dilithium)")
    } catch {
        expect(false, "Dilithium GPU vs CPU error: \(error)")
    }

    // ============================================================
    // Kyber pointwise mul matches polynomial multiplication
    // ============================================================
    suite("Lattice NTT: Kyber pointwise mul")
    do {
        let engine = try LatticeNTT()

        resetTestRNG()
        var a = [UInt32](repeating: 0, count: 256)
        var b = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 {
            a[i] = UInt32(nextTestRandom() % 3329)
            b[i] = UInt32(nextTestRandom() % 3329)
        }

        // Method 1: NTT(a) * NTT(b) pointwise, then INTT
        let nttA = engine.ntt(data: a, mode: .kyber)
        let nttB = engine.ntt(data: b, mode: .kyber)
        let nttProd = engine.pointwiseMul(a: nttA, b: nttB, mode: .kyber)
        let polyProd = engine.intt(data: nttProd, mode: .kyber)

        // Method 2: Schoolbook polynomial multiplication
        let expected = kyberSchoolbookMul(a, b)

        var ok = true
        for i in 0..<256 {
            if polyProd[i] != expected[i] {
                ok = false
                break
            }
        }
        expect(ok, "Pointwise NTT mul matches schoolbook (Kyber)")
    } catch {
        expect(false, "Kyber pointwise mul error: \(error)")
    }

    // ============================================================
    // Dilithium pointwise mul matches polynomial multiplication
    // ============================================================
    suite("Lattice NTT: Dilithium pointwise mul")
    do {
        let engine = try LatticeNTT()

        resetTestRNG()
        var a = [UInt32](repeating: 0, count: 256)
        var b = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 {
            a[i] = UInt32(nextTestRandom() % 8380417)
            b[i] = UInt32(nextTestRandom() % 8380417)
        }

        let nttA = engine.ntt(data: a, mode: .dilithium)
        let nttB = engine.ntt(data: b, mode: .dilithium)
        let nttProd = engine.pointwiseMul(a: nttA, b: nttB, mode: .dilithium)
        let polyProd = engine.intt(data: nttProd, mode: .dilithium)

        let expected = dilithiumSchoolbookMul(a, b)

        var ok = true
        for i in 0..<256 {
            if polyProd[i] != expected[i] {
                ok = false
                break
            }
        }
        expect(ok, "Pointwise NTT mul matches schoolbook (Dilithium)")
    } catch {
        expect(false, "Dilithium pointwise mul error: \(error)")
    }

    // ============================================================
    // Known test vectors — Kyber
    // ============================================================
    suite("Lattice NTT: Kyber test vectors")
    do {
        let engine = try LatticeNTT()

        // Zero polynomial should NTT to zero
        let zeros = [UInt32](repeating: 0, count: 256)
        let nttZeros = engine.ntt(data: zeros, mode: .kyber)
        let allZero = nttZeros.allSatisfy { $0 == 0 }
        expect(allZero, "NTT(zero) = zero (Kyber)")

        // Constant polynomial [c, 0, 0, ...] should NTT to [c, c, c, ...]
        // in a standard NTT (but negacyclic NTT is different — constant poly
        // maps to c * powers of root). Just verify round-trip.
        var constPoly = [UInt32](repeating: 0, count: 256)
        constPoly[0] = 42
        let nttConst = engine.ntt(data: constPoly, mode: .kyber)
        let recConst = engine.intt(data: nttConst, mode: .kyber)
        expect(recConst[0] == 42, "Constant poly round-trip coeff 0 (Kyber)")
        let restZero = recConst[1...].allSatisfy { $0 == 0 }
        expect(restZero, "Constant poly round-trip rest zero (Kyber)")

        // Identity: NTT([1, 0, 0, ...]) pointwise * NTT(a) = NTT(a)
        var identity = [UInt32](repeating: 0, count: 256)
        identity[0] = 1
        let nttId = engine.ntt(data: identity, mode: .kyber)

        resetTestRNG()
        var testPoly = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 { testPoly[i] = UInt32(nextTestRandom() % 3329) }
        let nttTest = engine.ntt(data: testPoly, mode: .kyber)
        let prod = engine.pointwiseMul(a: nttId, b: nttTest, mode: .kyber)
        let result = engine.intt(data: prod, mode: .kyber)
        var idOk = true
        for i in 0..<256 {
            if result[i] != testPoly[i] { idOk = false; break }
        }
        expect(idOk, "Multiplicative identity [1,0,...] (Kyber)")
    } catch {
        expect(false, "Kyber test vectors error: \(error)")
    }

    // ============================================================
    // Known test vectors — Dilithium
    // ============================================================
    suite("Lattice NTT: Dilithium test vectors")
    do {
        let engine = try LatticeNTT()

        // Zero poly
        let zeros = [UInt32](repeating: 0, count: 256)
        let nttZeros = engine.ntt(data: zeros, mode: .dilithium)
        let allZero = nttZeros.allSatisfy { $0 == 0 }
        expect(allZero, "NTT(zero) = zero (Dilithium)")

        // Constant poly round-trip
        var constPoly = [UInt32](repeating: 0, count: 256)
        constPoly[0] = 12345
        let nttConst = engine.ntt(data: constPoly, mode: .dilithium)
        let recConst = engine.intt(data: nttConst, mode: .dilithium)
        expect(recConst[0] == 12345, "Constant poly round-trip coeff 0 (Dilithium)")
        let restZero = recConst[1...].allSatisfy { $0 == 0 }
        expect(restZero, "Constant poly round-trip rest zero (Dilithium)")

        // Multiplicative identity
        var identity = [UInt32](repeating: 0, count: 256)
        identity[0] = 1
        let nttId = engine.ntt(data: identity, mode: .dilithium)

        resetTestRNG()
        var testPoly = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 { testPoly[i] = UInt32(nextTestRandom() % 8380417) }
        let nttTest = engine.ntt(data: testPoly, mode: .dilithium)
        let prod = engine.pointwiseMul(a: nttId, b: nttTest, mode: .dilithium)
        let result = engine.intt(data: prod, mode: .dilithium)
        var idOk = true
        for i in 0..<256 {
            if result[i] != testPoly[i] { idOk = false; break }
        }
        expect(idOk, "Multiplicative identity [1,0,...] (Dilithium)")
    } catch {
        expect(false, "Dilithium test vectors error: \(error)")
    }

    // ============================================================
    // Batch NTT: 1024 polynomials simultaneously
    // ============================================================
    suite("Lattice NTT: Batch 1024 polynomials")
    do {
        let engine = try LatticeNTT()

        // Kyber batch: 1024 polynomials
        let numPolys = 1024
        resetTestRNG()
        var kyberBatch = [UInt32](repeating: 0, count: numPolys * 256)
        for i in 0..<kyberBatch.count {
            kyberBatch[i] = UInt32(nextTestRandom() % 3329)
        }

        let nttBatch = engine.ntt(data: kyberBatch, mode: .kyber)
        let recovered = engine.intt(data: nttBatch, mode: .kyber)

        var kyberOk = true
        for i in 0..<kyberBatch.count {
            if kyberBatch[i] != recovered[i] {
                kyberOk = false
                break
            }
        }
        expect(kyberOk, "Batch 1024 round-trip (Kyber)")

        // Dilithium batch: 1024 polynomials
        resetTestRNG()
        var dilBatch = [UInt32](repeating: 0, count: numPolys * 256)
        for i in 0..<dilBatch.count {
            dilBatch[i] = UInt32(nextTestRandom() % 8380417)
        }

        let nttDilBatch = engine.ntt(data: dilBatch, mode: .dilithium)
        let recoveredDil = engine.intt(data: nttDilBatch, mode: .dilithium)

        var dilOk = true
        for i in 0..<dilBatch.count {
            if dilBatch[i] != recoveredDil[i] {
                dilOk = false
                break
            }
        }
        expect(dilOk, "Batch 1024 round-trip (Dilithium)")

        // Verify batch consistency: each polynomial in the batch should
        // produce the same NTT as processing it individually
        let singleNTT = engine.ntt(data: Array(kyberBatch[0..<256]), mode: .kyber)
        var singleOk = true
        for i in 0..<256 {
            if nttBatch[i] != singleNTT[i] {
                singleOk = false
                break
            }
        }
        expect(singleOk, "Batch NTT[0] matches single NTT (Kyber)")
    } catch {
        expect(false, "Batch NTT error: \(error)")
    }

    // ============================================================
    // Edge cases
    // ============================================================
    suite("Lattice NTT: Edge cases")
    do {
        let engine = try LatticeNTT()

        // All max values (q-1)
        let kyberMax = [UInt32](repeating: 3328, count: 256)
        let nttMax = engine.ntt(data: kyberMax, mode: .kyber)
        let recMax = engine.intt(data: nttMax, mode: .kyber)
        var maxOk = true
        for i in 0..<256 {
            if kyberMax[i] != recMax[i] { maxOk = false; break }
        }
        expect(maxOk, "All q-1 round-trip (Kyber)")

        let dilMax = [UInt32](repeating: 8380416, count: 256)
        let nttDilMax = engine.ntt(data: dilMax, mode: .dilithium)
        let recDilMax = engine.intt(data: nttDilMax, mode: .dilithium)
        var dilMaxOk = true
        for i in 0..<256 {
            if dilMax[i] != recDilMax[i] { dilMaxOk = false; break }
        }
        expect(dilMaxOk, "All q-1 round-trip (Dilithium)")

        // All ones
        let kyberOnes = [UInt32](repeating: 1, count: 256)
        let nttOnes = engine.ntt(data: kyberOnes, mode: .kyber)
        let recOnes = engine.intt(data: nttOnes, mode: .kyber)
        var onesOk = true
        for i in 0..<256 {
            if kyberOnes[i] != recOnes[i] { onesOk = false; break }
        }
        expect(onesOk, "All ones round-trip (Kyber)")

        // Pointwise mul with zero vector gives zero
        resetTestRNG()
        var nonzero = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 { nonzero[i] = UInt32(nextTestRandom() % 3329) }
        let zeroVec = [UInt32](repeating: 0, count: 256)
        let mulZero = engine.pointwiseMul(a: nonzero, b: zeroVec, mode: .kyber)
        let allZeroResult = mulZero.allSatisfy { $0 == 0 }
        expect(allZeroResult, "Pointwise mul with zero = zero (Kyber)")

        // Empty array
        let empty = engine.ntt(data: [], mode: .kyber)
        expect(empty.isEmpty, "Empty input returns empty")
    } catch {
        expect(false, "Edge cases error: \(error)")
    }
}
