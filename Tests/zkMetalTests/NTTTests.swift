import zkMetal

func runNTTTests() {
    suite("NTT BN254 Fr")
    do {
        let engine = try NTTEngine()
        let n = 1024
        var input = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { input[i] = frFromInt(UInt64(i + 1)) }

        let fwd = try engine.ntt(input)
        let rec = try engine.intt(fwd)
        var ok = true
        for i in 0..<n { if frToInt(input[i]) != frToInt(rec[i]) { ok = false; break } }
        expect(ok, "GPU round-trip 2^10")

        let cpuNTT = NTTEngine.cpuNTT(input, logN: 10)
        var cpuOk = true
        for i in 0..<n { if frToInt(fwd[i]) != frToInt(cpuNTT[i]) { cpuOk = false; break } }
        expect(cpuOk, "GPU vs CPU 2^10")

        // Four-step round-trip 2^20
        let n2 = 1 << 20
        var input2 = [Fr](repeating: Fr.zero, count: n2)
        var rng: UInt64 = 0xCAFE_BABE
        for i in 0..<n2 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; input2[i] = frFromInt(rng >> 32) }
        let fwd2 = try engine.ntt(input2)
        let rec2 = try engine.intt(fwd2)
        var mm = 0
        for i in 0..<n2 { if frToInt(input2[i]) != frToInt(rec2[i]) { mm += 1 } }
        expect(mm == 0, "Four-step round-trip 2^20")
    } catch { expect(false, "NTT BN254 error: \(error)") }

    suite("NTT BabyBear")
    do {
        let engine = try BabyBearNTTEngine()
        let n = 1024
        var input = [Bb](repeating: Bb.zero, count: n)
        for i in 0..<n { input[i] = Bb(v: UInt32(i + 1)) }
        let fwd = try engine.ntt(input)
        let rec = try engine.intt(fwd)
        var ok = true
        for i in 0..<n { if input[i].v != rec[i].v { ok = false; break } }
        expect(ok, "BabyBear round-trip 2^10")
    } catch { expect(false, "NTT BabyBear error: \(error)") }

    suite("NTT Goldilocks")
    do {
        let engine = try GoldilocksNTTEngine()
        let n = 1024
        var input = [Gl](repeating: Gl.zero, count: n)
        for i in 0..<n { input[i] = Gl(v: UInt64(i + 1)) }
        let fwd = try engine.ntt(input)
        let rec = try engine.intt(fwd)
        var ok = true
        for i in 0..<n { if input[i].v != rec[i].v { ok = false; break } }
        expect(ok, "Goldilocks round-trip 2^10")
    } catch { expect(false, "NTT GL error: \(error)") }

    suite("Circle NTT GPU")
    do {
        let engine = try CircleNTTEngine()
        for logN in 1...10 {
            let n = 1 << logN
            var coeffs = [M31](repeating: M31.zero, count: n)
            var rng: UInt64 = 0xCAFE + UInt64(logN)
            for i in 0..<n { rng = rng &* 6364136223846793005 &+ 1442695040888963407; coeffs[i] = M31(v: UInt32(rng >> 33) % M31.P) }
            let cpu = CircleNTTEngine.cpuNTT(coeffs, logN: logN)
            let gpu = try engine.ntt(coeffs)
            var fwdOk = true
            for i in 0..<n { if gpu[i].v != cpu[i].v { fwdOk = false; break } }
            let rec = try engine.intt(gpu)
            var invOk = true
            for i in 0..<n { if rec[i].v != coeffs[i].v { invOk = false; break } }
            expect(fwdOk && invOk, "Circle NTT N=\(n)")
        }
    } catch { expect(false, "Circle NTT error: \(error)") }

    suite("C NTT Cross-Checks")
    do {
        let n = 1024
        var frIn = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { frIn[i] = frFromInt(UInt64(i + 1)) }
        let vanilla = NTTEngine.cpuNTT(frIn, logN: 10)
        let c = cNTT_Fr(frIn, logN: 10)
        var ok = true
        for i in 0..<n { if frToInt(vanilla[i]) != frToInt(c[i]) { ok = false; break } }
        expect(ok, "C NTT Fr matches vanilla")
        let rt = cINTT_Fr(c, logN: 10)
        var rtOk = true
        for i in 0..<n { if frToInt(frIn[i]) != frToInt(rt[i]) { rtOk = false; break } }
        expect(rtOk, "C NTT Fr round-trip")

        var bbIn = [Bb](repeating: Bb.zero, count: n)
        for i in 0..<n { bbIn[i] = Bb(v: UInt32(i + 1)) }
        let vBb = BabyBearNTTEngine.cpuNTT(bbIn, logN: 10)
        let nBb = neonNTT_Bb(bbIn, logN: 10)
        var bbOk = true
        for i in 0..<n { if vBb[i].v != nBb[i].v { bbOk = false; break } }
        expect(bbOk, "NEON BabyBear matches vanilla")

        var glIn = [Gl](repeating: Gl.zero, count: n)
        for i in 0..<n { glIn[i] = Gl(v: UInt64(i + 1)) }
        let vGl = GoldilocksNTTEngine.cpuNTT(glIn, logN: 10)
        let cGl = cNTT_Gl(glIn, logN: 10)
        var glOk = true
        for i in 0..<n { if vGl[i].v != cGl[i].v { glOk = false; break } }
        expect(glOk, "C Goldilocks matches vanilla")
    }
}
