import zkMetal

/// Tests verifying the trivial twiddle skip optimization in NTT butterfly operations.
/// When twiddle == 1 (identity), the Montgomery multiplication is skipped and only
/// modular add/sub is performed. These tests ensure correctness is maintained.
func runNTTTwiddleSkipTests() {
    suite("NTT Twiddle Skip — BabyBear")
    do {
        // Round-trip at multiple sizes (exercises different stage counts)
        for logN in [1, 2, 3, 4, 8, 10, 12] {
            let n = 1 << logN
            var input = [Bb](repeating: Bb.zero, count: n)
            var rng: UInt64 = 0xDEAD_BEEF + UInt64(logN)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = Bb(v: UInt32(rng >> 33) % 2013265921)
            }
            let fwd = neonNTT_Bb(input, logN: logN)
            let rec = neonINTT_Bb(fwd, logN: logN)
            var ok = true
            for i in 0..<n { if input[i].v != rec[i].v { ok = false; break } }
            expect(ok, "BabyBear NEON round-trip 2^\(logN)")
        }

        // Cross-check: NEON (with twiddle skip) vs vanilla Swift NTT
        let logN = 10
        let n = 1 << logN
        var input = [Bb](repeating: Bb.zero, count: n)
        for i in 0..<n { input[i] = Bb(v: UInt32(i + 1)) }
        let vanilla = BabyBearNTTEngine.cpuNTT(input, logN: logN)
        let neon = neonNTT_Bb(input, logN: logN)
        var matchOk = true
        for i in 0..<n { if vanilla[i].v != neon[i].v { matchOk = false; break } }
        expect(matchOk, "BabyBear NEON matches vanilla 2^10")
    }

    suite("NTT Twiddle Skip — Goldilocks")
    do {
        // Round-trip at multiple sizes
        for logN in [1, 2, 3, 4, 8, 10, 12] {
            let n = 1 << logN
            var input = [Gl](repeating: Gl.zero, count: n)
            var rng: UInt64 = 0xCAFE_1234 + UInt64(logN)
            let glP: UInt64 = 0xFFFFFFFF00000001
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = Gl(v: rng % glP)
            }
            let fwd = cNTT_Gl(input, logN: logN)
            let rec = cINTT_Gl(fwd, logN: logN)
            var ok = true
            for i in 0..<n { if input[i].v != rec[i].v { ok = false; break } }
            expect(ok, "Goldilocks C round-trip 2^\(logN)")
        }

        // Cross-check against vanilla
        let logN = 10
        let n = 1 << logN
        var input = [Gl](repeating: Gl.zero, count: n)
        for i in 0..<n { input[i] = Gl(v: UInt64(i + 1)) }
        let vanilla = GoldilocksNTTEngine.cpuNTT(input, logN: logN)
        let c = cNTT_Gl(input, logN: logN)
        var matchOk = true
        for i in 0..<n { if vanilla[i].v != c[i].v { matchOk = false; break } }
        expect(matchOk, "Goldilocks C matches vanilla 2^10")
    }

    suite("NTT Twiddle Skip — BN254 Fr")
    do {
        // Round-trip at multiple sizes
        for logN in [1, 2, 3, 4, 8, 10] {
            let n = 1 << logN
            var input = [Fr](repeating: Fr.zero, count: n)
            var rng: UInt64 = 0xBEEF_CAFE + UInt64(logN)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = frFromInt(rng >> 32)
            }
            let fwd = cNTT_Fr(input, logN: logN)
            let rec = cINTT_Fr(fwd, logN: logN)
            var ok = true
            for i in 0..<n { if frToInt(input[i]) != frToInt(rec[i]) { ok = false; break } }
            expect(ok, "BN254 Fr C round-trip 2^\(logN)")
        }

        // Cross-check against vanilla
        let logN = 10
        let n = 1 << logN
        var input = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { input[i] = frFromInt(UInt64(i + 1)) }
        let vanilla = NTTEngine.cpuNTT(input, logN: logN)
        let c = cNTT_Fr(input, logN: logN)
        var matchOk = true
        for i in 0..<n { if frToInt(vanilla[i]) != frToInt(c[i]) { matchOk = false; break } }
        expect(matchOk, "BN254 Fr C matches vanilla 2^10")
    }

    suite("NTT Twiddle Skip — BLS12-377 Fr")
    do {
        for logN in [1, 2, 3, 4, 8, 10] {
            let n = 1 << logN
            var input = [Fr377](repeating: Fr377.zero, count: n)
            var rng: UInt64 = 0xABCD_EF01 + UInt64(logN)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = fr377FromInt(rng >> 32)
            }
            let fwd = cNTT_Fr377(input, logN: logN)
            let rec = cINTT_Fr377(fwd, logN: logN)
            var ok = true
            for i in 0..<n {
                let a = fr377ToInt(input[i])
                let b = fr377ToInt(rec[i])
                if a != b { ok = false; break }
            }
            expect(ok, "BLS12-377 Fr C round-trip 2^\(logN)")
        }
    }

    suite("NTT Twiddle Skip — Edge Cases")
    do {
        // Size 2 (logN=1): every butterfly has twiddle==1 for stage 0
        var bb2 = [Bb(v: 42), Bb(v: 17)]
        let fwd2 = neonNTT_Bb(bb2, logN: 1)
        let rec2 = neonINTT_Bb(fwd2, logN: 1)
        expect(bb2[0].v == rec2[0].v && bb2[1].v == rec2[1].v, "BabyBear N=2 round-trip")

        // Size 4 (logN=2): stage 0 has all twiddle==1
        var bb4 = [Bb](repeating: Bb.zero, count: 4)
        for i in 0..<4 { bb4[i] = Bb(v: UInt32(100 + i * 37)) }
        let fwd4 = neonNTT_Bb(bb4, logN: 2)
        let rec4 = neonINTT_Bb(fwd4, logN: 2)
        var ok4 = true
        for i in 0..<4 { if bb4[i].v != rec4[i].v { ok4 = false; break } }
        expect(ok4, "BabyBear N=4 round-trip")

        // Goldilocks N=2
        var gl2 = [Gl(v: 123456789), Gl(v: 987654321)]
        let gfwd2 = cNTT_Gl(gl2, logN: 1)
        let grec2 = cINTT_Gl(gfwd2, logN: 1)
        expect(gl2[0].v == grec2[0].v && gl2[1].v == grec2[1].v, "Goldilocks N=2 round-trip")

        // BN254 Fr N=2
        var fr2 = [frFromInt(42), frFromInt(17)]
        let frfwd2 = cNTT_Fr(fr2, logN: 1)
        let frrec2 = cINTT_Fr(frfwd2, logN: 1)
        expect(frToInt(fr2[0]) == frToInt(frrec2[0]) && frToInt(fr2[1]) == frToInt(frrec2[1]),
               "BN254 Fr N=2 round-trip")
    }
}
