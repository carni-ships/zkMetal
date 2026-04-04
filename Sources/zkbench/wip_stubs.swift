// Stubs for bench functions whose implementations are in .wip files
import Foundation
import zkMetal

public func runDataParallelBench() { fputs("[stub] DataParallel bench not yet available\n", stderr) }
public func runBatchVerifyBench() { fputs("[stub] BatchVerify bench not yet available\n", stderr) }
public func runJoltBench() { fputs("[stub] Jolt bench not yet available\n", stderr) }

public func runAccumulationBench() {
    fputs("\n=== IPA Accumulation Scheme (Pallas) ===\n", stderr)
    fputs("\n--- Pallas Curve Sanity ---\n", stderr)
    let g = pallasGenerator()
    let gProj = pallasPointFromAffine(g)
    let g2 = pallasPointDouble(gProj)
    let g2add = pallasPointAdd(gProj, gProj)
    let dblOk = pallasPointEqual(g2, g2add)
    fputs("  2G == G+G: \(dblOk ? "PASS" : "FAIL")\n", stderr)
    let three = vestaFromInt(3)
    let threeG = pallasPointScalarMul(gProj, three)
    let g3add = pallasPointAdd(gProj, pallasPointAdd(gProj, gProj))
    let smulOk = pallasPointEqual(threeG, g3add)
    fputs("  3*G == G+G+G: \(smulOk ? "PASS" : "FAIL")\n", stderr)
    var allPass = dblOk && smulOk
    for logN in [2, 4] {
        let n = 1 << logN
        fputs("\n--- Pallas IPA n=\(n) ---\n", stderr)
        let (gens, qPt) = PallasAccumulationEngine.generateTestGenerators(count: n)
        let eng = PallasAccumulationEngine(generators: gens, Q: qPt)
        var a = [VestaFp](); var b = [VestaFp]()
        for i in 0..<n { a.append(vestaFromInt(UInt64(i + 1))); b.append(vestaFromInt(UInt64(n - i))) }
        let v = PallasAccumulationEngine.innerProduct(a, b)
        let cm = eng.commit(a)
        let qP = pallasPointFromAffine(qPt)
        let cB = pallasPointAdd(cm, pallasPointScalarMul(qP, v))
        let t0 = CFAbsoluteTimeGetCurrent()
        let prf = eng.createProof(a: a, b: b)
        fputs("  Prove: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - t0) * 1000)) ms\n", stderr)
        let t1 = CFAbsoluteTimeGetCurrent()
        let ok = eng.verify(commitment: cB, b: b, innerProductValue: v, proof: prf)
        fputs("  Verify: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - t1) * 1000)) ms -- \(ok ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && ok
        let rej = !eng.verify(commitment: pallasPointAdd(cm, pallasPointScalarMul(qP, vestaFromInt(999))), b: b, innerProductValue: vestaFromInt(999), proof: prf)
        fputs("  Reject wrong v: \(rej ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && rej
        let t2 = CFAbsoluteTimeGetCurrent()
        let ac = eng.accumulate(proof: prf, commitment: cB, b: b, innerProductValue: v)
        fputs("  Accumulate: \(String(format: "%.3f", (CFAbsoluteTimeGetCurrent() - t2) * 1000)) ms\n", stderr)
        let t3 = CFAbsoluteTimeGetCurrent()
        let dok = eng.decide(ac)
        fputs("  Decide: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - t3) * 1000)) ms -- \(dok ? "PASS" : "FAIL")\n", stderr)
        allPass = allPass && dok
    }
    fputs("\n--- Batch Decide ---\n", stderr)
    let n4 = 4
    let (g4, q4) = PallasAccumulationEngine.generateTestGenerators(count: n4)
    let e4 = PallasAccumulationEngine(generators: g4, Q: q4)
    var accs4 = [IPAAccumulator]()
    for p in 0..<5 {
        var a4 = [VestaFp](); var b4 = [VestaFp]()
        for i in 0..<n4 { a4.append(vestaFromInt(UInt64(p * n4 + i + 1))); b4.append(vestaFromInt(UInt64(i + 1))) }
        let prf4 = e4.createProof(a: a4, b: b4)
        let v4 = PallasAccumulationEngine.innerProduct(a4, b4)
        let cm4 = e4.commit(a4)
        let cB4 = pallasPointAdd(cm4, pallasPointScalarMul(pallasPointFromAffine(q4), v4))
        accs4.append(e4.accumulate(proof: prf4, commitment: cB4, b: b4, innerProductValue: v4))
    }
    var iTotal = 0.0; var iPass = true
    for i in 0..<5 { let t = CFAbsoluteTimeGetCurrent(); if !e4.decide(accs4[i]) { iPass = false }; iTotal += CFAbsoluteTimeGetCurrent() - t }
    fputs("  Individual 5x: \(String(format: "%.1f", iTotal * 1000)) ms -- \(iPass ? "ALL PASS" : "FAIL")\n", stderr)
    allPass = allPass && iPass
    let tb = CFAbsoluteTimeGetCurrent()
    let bok = e4.batchDecide(accs4)
    let bt = (CFAbsoluteTimeGetCurrent() - tb) * 1000
    fputs("  Batch 5x: \(String(format: "%.1f", bt)) ms -- \(bok ? "PASS" : "FAIL") (\(String(format: "%.2f", (iTotal * 1000) / bt))x)\n", stderr)
    allPass = allPass && bok
    fputs("\n--- Recursive Hash ---\n", stderr)
    let tr = CFAbsoluteTimeGetCurrent()
    let (_, hv) = IteratedHashDemo.run(steps: 3, generatorCount: 4)
    fputs("  3 steps: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - tr) * 1000)) ms -- \(hv ? "PASS" : "FAIL")\n", stderr)
    allPass = allPass && hv
    fputs("\n  Result: \(allPass ? "ALL PASS" : "SOME FAILED")\n", stderr)
}
