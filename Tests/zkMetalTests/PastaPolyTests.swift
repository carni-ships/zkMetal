// Pasta polynomial operations tests — Pallas Fr and Vesta Fr
import zkMetal

func runPastaPolyTests() {
    suite("Pallas Fr Polynomial (via PallasPolyEngine)")

    // Helper: compare VestaFp elements by converting to integer
    func vestaEq(_ a: VestaFp, _ b: VestaFp) -> Bool {
        let ai = vestaToInt(a)
        let bi = vestaToInt(b)
        return ai[0] == bi[0] && ai[1] == bi[1] && ai[2] == bi[2] && ai[3] == bi[3]
    }

    // --- Horner evaluation ---
    // p(x) = 3x^2 + 2x + 1, evaluate at x=5, expect 86
    do {
        let c0 = vestaFromInt(1)
        let c1 = vestaFromInt(2)
        let c2 = vestaFromInt(3)
        let coeffs = [c0, c1, c2]
        let z = vestaFromInt(5)
        let result = PallasPolyEngine.evaluate(coeffs, at: z)
        expect(vestaToInt(result)[0] == 86, "Horner: 3*25+2*5+1=86")
    }

    // p(x) = 7x^3 + 0x^2 + 0x + 42, evaluate at x=10, expect 7042
    do {
        let coeffs = [vestaFromInt(42), VestaFp.zero, VestaFp.zero, vestaFromInt(7)]
        let z = vestaFromInt(10)
        let result = PallasPolyEngine.evaluate(coeffs, at: z)
        expect(vestaToInt(result)[0] == 7042, "Horner: 7*1000+42=7042")
    }

    // Constant polynomial: p(x) = 99
    do {
        let result = PallasPolyEngine.evaluate([vestaFromInt(99)], at: vestaFromInt(123))
        expect(vestaToInt(result)[0] == 99, "Horner: constant poly")
    }

    // Empty polynomial = 0
    do {
        let result = PallasPolyEngine.evaluate([], at: vestaFromInt(5))
        expect(vestaEq(result, VestaFp.zero), "Horner: empty poly = 0")
    }

    // --- Synthetic division ---
    // (x^3 - 1) / (x - 1) = x^2 + x + 1
    // coeffs of x^3 - 1 = [-1, 0, 0, 1]
    do {
        let negOne = vestaNeg(vestaFromInt(1))
        let coeffs = [negOne, VestaFp.zero, VestaFp.zero, vestaFromInt(1)]
        let z = vestaFromInt(1)
        let q = PallasPolyEngine.syntheticDiv(coeffs, z: z)
        // q should be [1, 1, 1] = x^2 + x + 1
        expect(q.count == 3, "SynDiv: quotient has 3 coeffs")
        expect(vestaToInt(q[0])[0] == 1, "SynDiv: q[0]=1")
        expect(vestaToInt(q[1])[0] == 1, "SynDiv: q[1]=1")
        expect(vestaToInt(q[2])[0] == 1, "SynDiv: q[2]=1")
    }

    // --- Fused eval + div ---
    // p(x) = x^2 - 9 = (x-3)(x+3), z=3, p(3)=0, q(x) = x+3
    do {
        let negNine = vestaNeg(vestaFromInt(9))
        let coeffs = [negNine, VestaFp.zero, vestaFromInt(1)]
        let z = vestaFromInt(3)
        let (eval, q) = PallasPolyEngine.evalAndDiv(coeffs, z: z)
        expect(vestaEq(eval, VestaFp.zero), "EvalAndDiv: p(3)=0")
        expect(q.count == 2, "EvalAndDiv: quotient has 2 coeffs")
        expect(vestaToInt(q[0])[0] == 3, "EvalAndDiv: q[0]=3")
        expect(vestaToInt(q[1])[0] == 1, "EvalAndDiv: q[1]=1")
    }

    // --- Inner product ---
    // a = [1, 2, 3], b = [4, 5, 6], <a,b> = 4+10+18 = 32
    do {
        let a = [vestaFromInt(1), vestaFromInt(2), vestaFromInt(3)]
        let b = [vestaFromInt(4), vestaFromInt(5), vestaFromInt(6)]
        let ip = PallasPolyEngine.innerProduct(a, b)
        expect(vestaToInt(ip)[0] == 32, "InnerProduct: 1*4+2*5+3*6=32")
    }

    // --- Batch add ---
    do {
        let a = [vestaFromInt(10), vestaFromInt(20)]
        let b = [vestaFromInt(3), vestaFromInt(7)]
        let r = PallasPolyEngine.batchAdd(a, b)
        expect(r.count == 2, "BatchAdd: count")
        expect(vestaToInt(r[0])[0] == 13, "BatchAdd: 10+3=13")
        expect(vestaToInt(r[1])[0] == 27, "BatchAdd: 20+7=27")
    }

    // --- Batch sub ---
    do {
        let a = [vestaFromInt(100), vestaFromInt(50)]
        let b = [vestaFromInt(30), vestaFromInt(20)]
        let r = PallasPolyEngine.batchSub(a, b)
        expect(vestaToInt(r[0])[0] == 70, "BatchSub: 100-30=70")
        expect(vestaToInt(r[1])[0] == 30, "BatchSub: 50-20=30")
    }

    // --- Batch mul scalar ---
    do {
        var data = [vestaFromInt(5), vestaFromInt(10), vestaFromInt(15)]
        PallasPolyEngine.batchMulScalar(&data, scalar: vestaFromInt(3))
        expect(vestaToInt(data[0])[0] == 15, "BatchMulScalar: 5*3=15")
        expect(vestaToInt(data[1])[0] == 30, "BatchMulScalar: 10*3=30")
        expect(vestaToInt(data[2])[0] == 45, "BatchMulScalar: 15*3=45")
    }

    // --- Polynomial multiplication ---
    // (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2
    do {
        let a = [vestaFromInt(1), vestaFromInt(2)]
        let b = [vestaFromInt(3), vestaFromInt(4)]
        let c = PallasPolyEngine.multiply(a, b)
        expect(c.count == 3, "PolyMul: degree check")
        expect(vestaToInt(c[0])[0] == 3, "PolyMul: c0=3")
        expect(vestaToInt(c[1])[0] == 10, "PolyMul: c1=10")
        expect(vestaToInt(c[2])[0] == 8, "PolyMul: c2=8")
    }

    // --- Lagrange interpolation ---
    // Interpolate through (1,1), (2,4), (3,9) -> p(x) = x^2
    do {
        let xs = [vestaFromInt(1), vestaFromInt(2), vestaFromInt(3)]
        let ys = [vestaFromInt(1), vestaFromInt(4), vestaFromInt(9)]
        let poly = PallasPolyEngine.lagrangeInterpolation(xs: xs, ys: ys)
        expect(poly.count == 3, "Lagrange: 3 coefficients")
        // p(x) = 0 + 0*x + 1*x^2
        expect(vestaEq(poly[0], VestaFp.zero), "Lagrange: c0=0")
        expect(vestaEq(poly[1], VestaFp.zero), "Lagrange: c1=0")
        expect(vestaToInt(poly[2])[0] == 1, "Lagrange: c2=1")
    }

    // --- Round-trip: evaluate, divide, check remainder is zero ---
    // p(x) = 2x^3 + 5x^2 + 3x + 7, z=4
    // Evaluate, then divide by (x - 4), re-evaluate quotient at 4 and check
    // p(x) = (x - z) * q(x) + p(z), so p(z) should equal coeffs[0] + z*q(0) via fused
    do {
        let coeffs = [vestaFromInt(7), vestaFromInt(3), vestaFromInt(5), vestaFromInt(2)]
        let z = vestaFromInt(4)
        let pz = PallasPolyEngine.evaluate(coeffs, at: z)
        // p(4) = 2*64 + 5*16 + 3*4 + 7 = 128 + 80 + 12 + 7 = 227
        expect(vestaToInt(pz)[0] == 227, "RoundTrip: p(4)=227")

        let (evalFused, q) = PallasPolyEngine.evalAndDiv(coeffs, z: z)
        expect(vestaToInt(evalFused)[0] == 227, "RoundTrip: fused p(4)=227")

        // q(x) * (x - z) + p(z) should give back original p(x)
        // Verify: q(4) * 0 + p(4) = p(4) (trivially true)
        // Better: verify q has no remainder by checking eval matches
        let qz = PallasPolyEngine.evaluate(q, at: z)
        // q(z) should satisfy: p(z) = coeffs[0] + z * q[0]
        // This is automatically true from the fused algo. Let's verify a different way:
        // Reconstruct: (x - z) * q(x) + p(z) should equal p(x)
        // Check at x = 10:
        let x10 = vestaFromInt(10)
        let p10 = PallasPolyEngine.evaluate(coeffs, at: x10)
        let q10 = PallasPolyEngine.evaluate(q, at: x10)
        // (10 - 4) * q(10) + p(4)
        let six = vestaFromInt(6)
        let reconstructed = vestaAdd(vestaMul(six, q10), pz)
        expect(vestaEq(p10, reconstructed), "RoundTrip: (x-z)*q(x)+p(z) = p(x)")
    }

    // ================================================================
    suite("Vesta Fr Polynomial (via VestaPolyEngine)")
    // ================================================================

    func pallasEq(_ a: PallasFp, _ b: PallasFp) -> Bool {
        let ai = pallasToInt(a)
        let bi = pallasToInt(b)
        return ai[0] == bi[0] && ai[1] == bi[1] && ai[2] == bi[2] && ai[3] == bi[3]
    }

    // --- Horner evaluation ---
    // p(x) = 3x^2 + 2x + 1, evaluate at x=5, expect 86
    do {
        let coeffs = [pallasFromInt(1), pallasFromInt(2), pallasFromInt(3)]
        let z = pallasFromInt(5)
        let result = VestaPolyEngine.evaluate(coeffs, at: z)
        expect(pallasToInt(result)[0] == 86, "Vesta Horner: 3*25+2*5+1=86")
    }

    // --- Synthetic division ---
    // (x^3 - 1) / (x - 1) = x^2 + x + 1
    do {
        let negOne = pallasNeg(pallasFromInt(1))
        let coeffs = [negOne, PallasFp.zero, PallasFp.zero, pallasFromInt(1)]
        let z = pallasFromInt(1)
        let q = VestaPolyEngine.syntheticDiv(coeffs, z: z)
        expect(q.count == 3, "Vesta SynDiv: count")
        expect(pallasToInt(q[0])[0] == 1, "Vesta SynDiv: q[0]=1")
        expect(pallasToInt(q[1])[0] == 1, "Vesta SynDiv: q[1]=1")
        expect(pallasToInt(q[2])[0] == 1, "Vesta SynDiv: q[2]=1")
    }

    // --- Inner product ---
    do {
        let a = [pallasFromInt(1), pallasFromInt(2), pallasFromInt(3)]
        let b = [pallasFromInt(4), pallasFromInt(5), pallasFromInt(6)]
        let ip = VestaPolyEngine.innerProduct(a, b)
        expect(pallasToInt(ip)[0] == 32, "Vesta InnerProduct: 32")
    }

    // --- Fused eval + div ---
    do {
        let negNine = pallasNeg(pallasFromInt(9))
        let coeffs = [negNine, PallasFp.zero, pallasFromInt(1)]
        let z = pallasFromInt(3)
        let (eval, q) = VestaPolyEngine.evalAndDiv(coeffs, z: z)
        expect(pallasEq(eval, PallasFp.zero), "Vesta EvalAndDiv: p(3)=0")
        expect(q.count == 2, "Vesta EvalAndDiv: count")
        expect(pallasToInt(q[0])[0] == 3, "Vesta EvalAndDiv: q[0]=3")
        expect(pallasToInt(q[1])[0] == 1, "Vesta EvalAndDiv: q[1]=1")
    }

    // --- Batch ops ---
    do {
        let a = [pallasFromInt(10), pallasFromInt(20)]
        let b = [pallasFromInt(3), pallasFromInt(7)]
        let r = VestaPolyEngine.batchAdd(a, b)
        expect(pallasToInt(r[0])[0] == 13, "Vesta BatchAdd: 13")
        expect(pallasToInt(r[1])[0] == 27, "Vesta BatchAdd: 27")

        let s = VestaPolyEngine.batchSub(a, b)
        expect(pallasToInt(s[0])[0] == 7, "Vesta BatchSub: 7")
        expect(pallasToInt(s[1])[0] == 13, "Vesta BatchSub: 13")
    }

    // --- Polynomial multiplication ---
    do {
        let a = [pallasFromInt(1), pallasFromInt(2)]
        let b = [pallasFromInt(3), pallasFromInt(4)]
        let c = VestaPolyEngine.multiply(a, b)
        expect(c.count == 3, "Vesta PolyMul: count")
        expect(pallasToInt(c[0])[0] == 3, "Vesta PolyMul: c0=3")
        expect(pallasToInt(c[1])[0] == 10, "Vesta PolyMul: c1=10")
        expect(pallasToInt(c[2])[0] == 8, "Vesta PolyMul: c2=8")
    }

    // --- Round-trip for Vesta ---
    do {
        let coeffs = [pallasFromInt(7), pallasFromInt(3), pallasFromInt(5), pallasFromInt(2)]
        let z = pallasFromInt(4)
        let pz = VestaPolyEngine.evaluate(coeffs, at: z)
        expect(pallasToInt(pz)[0] == 227, "Vesta RoundTrip: p(4)=227")

        let (evalFused, q) = VestaPolyEngine.evalAndDiv(coeffs, z: z)
        expect(pallasToInt(evalFused)[0] == 227, "Vesta RoundTrip: fused p(4)=227")

        let x10 = pallasFromInt(10)
        let p10 = VestaPolyEngine.evaluate(coeffs, at: x10)
        let q10 = VestaPolyEngine.evaluate(q, at: x10)
        let six = pallasFromInt(6)
        let reconstructed = pallasAdd(pallasMul(six, q10), pz)
        expect(pallasEq(p10, reconstructed), "Vesta RoundTrip: (x-z)*q(x)+p(z)=p(x)")
    }
}
