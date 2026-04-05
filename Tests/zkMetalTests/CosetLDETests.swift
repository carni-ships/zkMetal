import zkMetal

func runCosetLDETests() {
    suite("Coset LDE BabyBear")
    do {
        let engine = try CosetLDEEngine()

        // Test 1: Small polynomial, blowup 2, CPU path
        let n = 16
        var poly = [Bb](repeating: Bb.zero, count: n)
        for i in 0..<n { poly[i] = Bb(v: UInt32(i + 1)) }
        let lde2 = try engine.cosetLDE(poly: poly, blowupFactor: 2)
        expectEqual(lde2.count, 32, "BabyBear LDE x2 size")

        // Verify against CPU reference
        let ldeRef = try engine.cpuCosetLDEBb(poly: poly, blowupFactor: 2)
        var match = true
        for i in 0..<lde2.count {
            if lde2[i].v != ldeRef[i].v { match = false; break }
        }
        expect(match, "BabyBear LDE x2 CPU consistency (n=16)")

        // Test 2: GPU path (n=1024), blowup 4
        let n2 = 1024
        var poly2 = [Bb](repeating: Bb.zero, count: n2)
        var rng: UInt64 = 0xDEAD_BEEF
        for i in 0..<n2 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            poly2[i] = Bb(v: UInt32(rng >> 33) % Bb.P)
        }
        let lde4 = try engine.cosetLDE(poly: poly2, blowupFactor: 4)
        expectEqual(lde4.count, 4096, "BabyBear LDE x4 size")

        // Verify round-trip size
        let bbNTT = try BabyBearNTTEngine()
        let coeffsBack = try bbNTT.intt(lde4)
        expectEqual(coeffsBack.count, 4096, "BabyBear LDE x4 round-trip size")

        // Test 3: Blowup 8
        let lde8 = try engine.cosetLDE(poly: poly, blowupFactor: 8)
        expectEqual(lde8.count, 128, "BabyBear LDE x8 size")
        let lde8Ref = try engine.cpuCosetLDEBb(poly: poly, blowupFactor: 8)
        var match8 = true
        for i in 0..<lde8.count {
            if lde8[i].v != lde8Ref[i].v { match8 = false; break }
        }
        expect(match8, "BabyBear LDE x8 CPU consistency")

    } catch { expect(false, "Coset LDE BabyBear error: \(error)") }

    suite("Coset LDE Goldilocks")
    do {
        let engine = try CosetLDEEngine()

        let n = 16
        var poly = [Gl](repeating: Gl.zero, count: n)
        for i in 0..<n { poly[i] = Gl(v: UInt64(i + 1)) }
        let lde = try engine.cosetLDE(poly: poly, blowupFactor: 2)
        expectEqual(lde.count, 32, "Goldilocks LDE x2 size")

        let ldeRef = try engine.cpuCosetLDEGl(poly: poly, blowupFactor: 2)
        var match = true
        for i in 0..<lde.count {
            if lde[i].v != ldeRef[i].v { match = false; break }
        }
        expect(match, "Goldilocks LDE x2 CPU consistency")

        // GPU path
        let n2 = 1024
        var poly2 = [Gl](repeating: Gl.zero, count: n2)
        var rng: UInt64 = 0xCAFE1234
        for i in 0..<n2 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            poly2[i] = Gl(v: rng % Gl.P)
        }
        let lde4 = try engine.cosetLDE(poly: poly2, blowupFactor: 4)
        expectEqual(lde4.count, 4096, "Goldilocks LDE x4 size")

    } catch { expect(false, "Coset LDE Goldilocks error: \(error)") }

    suite("Coset LDE BN254 Fr")
    do {
        let engine = try CosetLDEEngine()

        let n = 16
        var poly = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { poly[i] = frFromInt(UInt64(i + 1)) }
        let lde = try engine.cosetLDE(poly: poly, blowupFactor: 2)
        expectEqual(lde.count, 32, "BN254 Fr LDE x2 size")

        let ldeRef = try engine.cpuCosetLDEFr(poly: poly, blowupFactor: 2)
        var match = true
        for i in 0..<lde.count {
            let a = frToInt(lde[i])
            let b = frToInt(ldeRef[i])
            if a != b { match = false; break }
        }
        expect(match, "BN254 Fr LDE x2 CPU consistency")

        // GPU path
        let n2 = 1024
        var poly2 = [Fr](repeating: Fr.zero, count: n2)
        for i in 0..<n2 { poly2[i] = frFromInt(UInt64(i + 7)) }
        let lde4 = try engine.cosetLDE(poly: poly2, blowupFactor: 4)
        expectEqual(lde4.count, 4096, "BN254 Fr LDE x4 size")

    } catch { expect(false, "Coset LDE BN254 Fr error: \(error)") }

    suite("Batch Coset LDE BabyBear")
    do {
        let engine = try CosetLDEEngine()

        let n = 1024
        let numCols = 4
        var polys = [[Bb]]()
        var rng: UInt64 = 0xABCD1234
        for _ in 0..<numCols {
            var col = [Bb](repeating: Bb.zero, count: n)
            for j in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                col[j] = Bb(v: UInt32(rng >> 33) % Bb.P)
            }
            polys.append(col)
        }

        let results = try engine.batchCosetLDE(polys: polys, blowupFactor: 2)
        expectEqual(results.count, numCols, "Batch LDE column count")

        // Verify each column against single-column LDE
        var allMatch = true
        for c in 0..<numCols {
            let single = try engine.cosetLDE(poly: polys[c], blowupFactor: 2)
            expectEqual(results[c].count, single.count, "Batch LDE col \(c) size")
            for i in 0..<single.count {
                if results[c][i].v != single[i].v {
                    allMatch = false
                    break
                }
            }
            if !allMatch { break }
        }
        expect(allMatch, "Batch LDE matches single-column LDE")

    } catch { expect(false, "Batch Coset LDE error: \(error)") }
}
