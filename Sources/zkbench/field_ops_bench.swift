// Comprehensive field operations benchmark suite
// Measures single-core throughput for all C/NEON and Swift field implementations
// Reports ns/op, ops/sec, and theoretical cycle floors

import Foundation
import NeonFieldOps
import zkMetal

// MARK: - Anti-optimization sink

/// Global sink to prevent dead-code elimination of small field benchmarks.
/// The compiler cannot prove this is unused because it's a public global.
public var _fieldBenchSink32: UInt32 = 0
public var _fieldBenchSink64: UInt64 = 0

// MARK: - Result struct

private struct R {
    let field: String
    let op: String
    let ns: Double
    var mops: Double { 1e3 / ns }
}

private func emit(_ r: R) {
    let opPad = r.op.padding(toLength: 12, withPad: " ", startingAt: 0)
    fputs("  \(opPad): \(String(format: "%7.1f", r.ns)) ns/op  (\(String(format: "%6.1f", r.mops)) M ops/s)\n", stderr)
}

// MARK: - Inline timing macro (no closures to avoid Swift optimizer issues)

/// Time N iterations of inline code. Use via the bench_ functions below.
private func measure(_ n: Int, warmup: Int, body: () -> Void) -> Double {
    for _ in 0..<warmup { body() }
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<n { body() }
    let end = CFAbsoluteTimeGetCurrent()
    return (end - start) * 1e9 / Double(n)
}

// MARK: - 4-limb bench helper

private func bench4(
    field: String,
    aInit: [UInt64], bInit: [UInt64],
    mul: ((UnsafePointer<UInt64>?, UnsafePointer<UInt64>?, UnsafeMutablePointer<UInt64>?) -> Void)? = nil,
    sqr: ((UnsafePointer<UInt64>?, UnsafeMutablePointer<UInt64>?) -> Void)? = nil,
    add: ((UnsafePointer<UInt64>?, UnsafePointer<UInt64>?, UnsafeMutablePointer<UInt64>?) -> Void)? = nil,
    sub: ((UnsafePointer<UInt64>?, UnsafePointer<UInt64>?, UnsafeMutablePointer<UInt64>?) -> Void)? = nil,
    inv: ((UnsafePointer<UInt64>?, UnsafeMutablePointer<UInt64>?) -> Void)? = nil
) -> [R] {
    let iters = 1_000_000
    let warmupN = 100_000
    var results = [R]()

    let ap = UnsafeMutablePointer<UInt64>.allocate(capacity: 4)
    let bp = UnsafeMutablePointer<UInt64>.allocate(capacity: 4)
    let rp = UnsafeMutablePointer<UInt64>.allocate(capacity: 4)
    defer { ap.deallocate(); bp.deallocate(); rp.deallocate() }

    func loadA() { for i in 0..<4 { ap[i] = aInit[i] } }
    func loadB() { for i in 0..<4 { bp[i] = bInit[i] } }
    func chain() { ap[0]=rp[0]; ap[1]=rp[1]; ap[2]=rp[2]; ap[3]=rp[3] }

    if let fn = mul {
        loadA(); loadB()
        let ns = measure(iters, warmup: warmupN) { fn(ap, bp, rp); chain() }
        results.append(R(field: field, op: "mul", ns: ns))
    }

    if let fn = sqr {
        loadA()
        let ns = measure(iters, warmup: warmupN) { fn(ap, rp); chain() }
        results.append(R(field: field, op: "sqr", ns: ns))
    }

    if let fn = add {
        loadA(); loadB()
        let ns = measure(iters, warmup: warmupN) { fn(ap, bp, rp); chain() }
        results.append(R(field: field, op: "add", ns: ns))
    }

    if let fn = sub {
        loadA(); loadB()
        let ns = measure(iters, warmup: warmupN) { fn(ap, bp, rp); chain() }
        results.append(R(field: field, op: "sub", ns: ns))
    }

    if let fn = inv {
        loadA()
        let ns = measure(10_000, warmup: 1000) { fn(ap, rp); chain() }
        results.append(R(field: field, op: "inverse", ns: ns))
    }

    return results
}

// MARK: - 6-limb bench helper

private func bench6(
    field: String,
    aInit: [UInt64], bInit: [UInt64],
    mul: ((UnsafePointer<UInt64>?, UnsafePointer<UInt64>?, UnsafeMutablePointer<UInt64>?) -> Void)? = nil,
    sqr: ((UnsafePointer<UInt64>?, UnsafeMutablePointer<UInt64>?) -> Void)? = nil,
    add: ((UnsafePointer<UInt64>?, UnsafePointer<UInt64>?, UnsafeMutablePointer<UInt64>?) -> Void)? = nil,
    sub: ((UnsafePointer<UInt64>?, UnsafePointer<UInt64>?, UnsafeMutablePointer<UInt64>?) -> Void)? = nil,
    inv: ((UnsafePointer<UInt64>?, UnsafeMutablePointer<UInt64>?) -> Void)? = nil
) -> [R] {
    let iters = 1_000_000
    let warmupN = 100_000
    var results = [R]()

    let ap = UnsafeMutablePointer<UInt64>.allocate(capacity: 6)
    let bp = UnsafeMutablePointer<UInt64>.allocate(capacity: 6)
    let rp = UnsafeMutablePointer<UInt64>.allocate(capacity: 6)
    defer { ap.deallocate(); bp.deallocate(); rp.deallocate() }

    func loadA() { for i in 0..<6 { ap[i] = aInit[i] } }
    func loadB() { for i in 0..<6 { bp[i] = bInit[i] } }
    func chain() { for i in 0..<6 { ap[i] = rp[i] } }

    if let fn = mul {
        loadA(); loadB()
        let ns = measure(iters, warmup: warmupN) { fn(ap, bp, rp); chain() }
        results.append(R(field: field, op: "mul", ns: ns))
    }

    if let fn = sqr {
        loadA()
        let ns = measure(iters, warmup: warmupN) { fn(ap, rp); chain() }
        results.append(R(field: field, op: "sqr", ns: ns))
    }

    if let fn = add {
        loadA(); loadB()
        let ns = measure(iters, warmup: warmupN) { fn(ap, bp, rp); chain() }
        results.append(R(field: field, op: "add", ns: ns))
    }

    if let fn = sub {
        loadA(); loadB()
        let ns = measure(iters, warmup: warmupN) { fn(ap, bp, rp); chain() }
        results.append(R(field: field, op: "sub", ns: ns))
    }

    if let fn = inv {
        loadA()
        let ns = measure(10_000, warmup: 1000) { fn(ap, rp); chain() }
        results.append(R(field: field, op: "inverse", ns: ns))
    }

    return results
}

// MARK: - Small field benchmarks (Swift value types)

private func benchBabyBear() -> [R] {
    let f = "BabyBear (31b)"
    let n = 1_000_000
    let w = 100_000
    var res = [R]()

    var a = Bb(v: 0x12345678 % Bb.P)
    let b = Bb(v: 0x07654321 % Bb.P)

    res.append(R(field: f, op: "mul", ns: measure(n, warmup: w) { a = bbMul(a, b); _fieldBenchSink32 = a.v }))
    a = Bb(v: 0x12345678 % Bb.P)
    res.append(R(field: f, op: "sqr", ns: measure(n, warmup: w) { a = bbSqr(a); _fieldBenchSink32 = a.v }))
    a = Bb(v: 0x12345678 % Bb.P)
    res.append(R(field: f, op: "add", ns: measure(n, warmup: w) { a = bbAdd(a, b); _fieldBenchSink32 = a.v }))
    a = Bb(v: 0x12345678 % Bb.P)
    res.append(R(field: f, op: "sub", ns: measure(n, warmup: w) { a = bbSub(a, b); _fieldBenchSink32 = a.v }))
    a = Bb(v: 42)
    res.append(R(field: f, op: "inverse", ns: measure(100_000, warmup: 10_000) { a = bbInverse(a); _fieldBenchSink32 = a.v }))

    return res
}

private func benchGoldilocks() -> [R] {
    let f = "Goldilocks (64b)"
    let n = 1_000_000
    let w = 100_000
    var res = [R]()

    var a = Gl(v: 0xDEADBEEFCAFEBABE % Gl.P)
    let b = Gl(v: 0x123456789ABCDEF0 % Gl.P)

    res.append(R(field: f, op: "mul", ns: measure(n, warmup: w) { a = glMul(a, b); _fieldBenchSink64 = a.v }))
    a = Gl(v: 0xDEADBEEFCAFEBABE % Gl.P)
    res.append(R(field: f, op: "sqr", ns: measure(n, warmup: w) { a = glSqr(a); _fieldBenchSink64 = a.v }))
    a = Gl(v: 0xDEADBEEFCAFEBABE % Gl.P)
    res.append(R(field: f, op: "add", ns: measure(n, warmup: w) { a = glAdd(a, b); _fieldBenchSink64 = a.v }))
    a = Gl(v: 0xDEADBEEFCAFEBABE % Gl.P)
    res.append(R(field: f, op: "sub", ns: measure(n, warmup: w) { a = glSub(a, b); _fieldBenchSink64 = a.v }))
    a = Gl(v: 42)
    res.append(R(field: f, op: "inverse", ns: measure(10_000, warmup: 1000) { a = glInverse(a); _fieldBenchSink64 = a.v }))

    return res
}

private func benchMersenne31() -> [R] {
    let f = "Mersenne31 (31b)"
    let n = 1_000_000
    let w = 100_000
    var res = [R]()

    var a = M31(v: 0x12345678 % M31.P)
    let b = M31(v: 0x07654321 % M31.P)

    res.append(R(field: f, op: "mul", ns: measure(n, warmup: w) { a = m31Mul(a, b); _fieldBenchSink32 = a.v }))
    a = M31(v: 0x12345678 % M31.P)
    res.append(R(field: f, op: "sqr", ns: measure(n, warmup: w) { a = m31Sqr(a); _fieldBenchSink32 = a.v }))
    a = M31(v: 0x12345678 % M31.P)
    res.append(R(field: f, op: "add", ns: measure(n, warmup: w) { a = m31Add(a, b); _fieldBenchSink32 = a.v }))
    a = M31(v: 0x12345678 % M31.P)
    res.append(R(field: f, op: "sub", ns: measure(n, warmup: w) { a = m31Sub(a, b); _fieldBenchSink32 = a.v }))

    return res
}

// MARK: - ASM vs C comparison

private func benchAsmVsC() -> (asm: Double, c: Double) {
    let n = 1_000_000
    let ap = UnsafeMutablePointer<UInt64>.allocate(capacity: 4)
    let bp = UnsafeMutablePointer<UInt64>.allocate(capacity: 4)
    let rp = UnsafeMutablePointer<UInt64>.allocate(capacity: 4)
    defer { ap.deallocate(); bp.deallocate(); rp.deallocate() }

    let a0: [UInt64] = [0xac96341c4ffffffb, 0x36fc76959f60cd29,
                        0x666ea36f7879462e, 0x0e0a77c19a07df2f]
    let b0: [UInt64] = [0x1bb8e645ae216da7, 0x53fe3ab1e35c59e3,
                        0x8c49833d53bb8085, 0x0216d0b17f4e44a5]

    for i in 0..<4 { ap[i] = a0[i]; bp[i] = b0[i] }
    let asmNs = measure(n, warmup: 100_000) {
        mont_mul_asm(rp, ap, bp)
        ap[0]=rp[0]; ap[1]=rp[1]; ap[2]=rp[2]; ap[3]=rp[3]
    }

    for i in 0..<4 { ap[i] = a0[i] }
    let cNs = measure(n, warmup: 100_000) {
        mont_mul_c(rp, ap, bp)
        ap[0]=rp[0]; ap[1]=rp[1]; ap[2]=rp[2]; ap[3]=rp[3]
    }

    return (asmNs, cNs)
}

// MARK: - Summary table

private func pad(_ s: String, _ w: Int) -> String {
    s.padding(toLength: w, withPad: " ", startingAt: 0)
}

private func printSummary(_ all: [R]) {
    let sep = String(repeating: "=", count: 88)
    let line = String(repeating: "-", count: 88)
    fputs("\n\(sep)\n", stderr)
    fputs("  FIELD OPERATIONS SUMMARY TABLE\n", stderr)
    fputs("\(sep)\n", stderr)
    fputs("  \(pad("Field", 22)) \(pad("Operation", 12)) \(pad("ns/op", 10)) \(pad("M ops/s", 12)) \(pad("cyc/op*", 10))\n", stderr)
    fputs("\(line)\n", stderr)

    var lastField = ""
    for r in all {
        if r.field != lastField {
            if !lastField.isEmpty { fputs("\n", stderr) }
            lastField = r.field
        }
        let cyc = r.ns * 3.5
        fputs("  \(pad(r.field, 22)) \(pad(r.op, 12)) \(String(format: "%10.1f %12.1f %10.1f", r.ns, r.mops, cyc))\n", stderr)
    }
    fputs("\(line)\n", stderr)
    fputs("  * Estimated cycles at 3.5 GHz (M-series P-core)\n", stderr)
}

private func floorRow(_ field: String, _ limbs: String, _ muls: String, _ desc: String) {
    fputs("  \(pad(field, 22)) \(pad(limbs, 6)) \(pad(muls, 8)) \(desc)\n", stderr)
}

private func printFloors() {
    let sep = String(repeating: "=", count: 88)
    let line = String(repeating: "-", count: 88)
    fputs("\n\(sep)\n", stderr)
    fputs("  THEORETICAL FLOORS (minimum cycles for multiply chain)\n", stderr)
    fputs("\(sep)\n", stderr)
    floorRow("Field", "Limbs", "mul ops", "Min cycles (mul chain)")
    fputs("\(line)\n", stderr)

    floorRow("BN254 Fr/Fp (256b)",  "4",  "16", "~40-50 cyc (16 mul@4c + carry)")
    floorRow("BLS12-381 Fp (384b)", "6",  "36", "~100-120 cyc (36 mul@4c + carry)")
    floorRow("BLS12-381/377 Fr",    "4",  "16", "~40-50 cyc (4-limb CIOS)")
    floorRow("Ed25519 Fp (255b)",   "4",  "16", "~35-45 cyc (Solinas fast reduce)")
    floorRow("Goldilocks (64b)",    "1",  "1",  "~8-10 cyc (1 mul + Solinas)")
    floorRow("BabyBear (31b)",      "1",  "1",  "~4-6 cyc (32b mul + reduce)")
    floorRow("Mersenne31 (31b)",    "1",  "1",  "~4-6 cyc (32b mul + Mersenne)")
    floorRow("Add/Sub (any 256b)",  "-",  "0",  "~4-6 cyc (4-limb add + csel)")

    fputs("\(line)\n", stderr)
    fputs("  M-series Firestorm: mul/umulh 4-cycle latency, 1/cycle throughput\n", stderr)
    fputs("  Practical overhead: function call, load/store, branch, carry prop\n", stderr)
}

// MARK: - Public entry point

public func runFieldOpsBench() {
    let sep = String(repeating: "=", count: 72)
    fputs("\n\(sep)\n", stderr)
    fputs("  COMPREHENSIVE FIELD OPERATIONS BENCHMARK\n", stderr)
    fputs("  1M iterations per operation (10K for inverse)\n", stderr)
    fputs("  Serial dependency chain to prevent reordering\n", stderr)
    fputs("\(sep)\n", stderr)

    var all = [R]()

    // Test values for 4-limb fields (small enough to be valid in all fields)
    let a4: [UInt64] = [0x00000000CAFEBABE, 0x0000000000000001,
                        0x0000000000000001, 0x0000000000000001]
    let b4: [UInt64] = [0x00000000DEADBEEF, 0x0000000000000002,
                        0x0000000000000003, 0x0000000000000001]

    fputs("\n--- BN254 Fr (C CIOS Montgomery, 4-limb) ---\n", stderr)
    let r1 = bench4(field: "BN254 Fr (256b)", aInit: a4, bInit: b4,
                    mul: bn254_fr_mul, sqr: bn254_fr_sqr,
                    add: bn254_fr_add, sub: bn254_fr_sub,
                    inv: bn254_fr_inverse)
    r1.forEach { emit($0) }; all += r1

    fputs("\n--- BN254 Fp (C CIOS, sqr+inv) ---\n", stderr)
    let r2 = bench4(field: "BN254 Fp (256b)", aInit: a4, bInit: b4,
                    sqr: bn254_fp_sqr, inv: bn254_fp_inv)
    r2.forEach { emit($0) }; all += r2

    fputs("\n--- BLS12-381 Fr (C CIOS Montgomery, 4-limb) ---\n", stderr)
    let r3 = bench4(field: "BLS12-381 Fr (255b)", aInit: a4, bInit: b4,
                    mul: bls12_381_fr_mul, sqr: bls12_381_fr_sqr,
                    add: bls12_381_fr_add, sub: bls12_381_fr_sub)
    r3.forEach { emit($0) }; all += r3

    // 6-limb test values
    let a6: [UInt64] = [0x00000000CAFEBABE, 0x0000000000000001,
                        0x0000000000000001, 0x0000000000000001,
                        0x0000000000000001, 0x0000000000000001]
    let b6: [UInt64] = [0x00000000DEADBEEF, 0x0000000000000002,
                        0x0000000000000003, 0x0000000000000001,
                        0x0000000000000002, 0x0000000000000001]

    fputs("\n--- BLS12-381 Fp (C CIOS Montgomery, 6-limb) ---\n", stderr)
    let r4 = bench6(field: "BLS12-381 Fp (384b)", aInit: a6, bInit: b6,
                    mul: bls12_381_fp_mul, sqr: bls12_381_fp_sqr,
                    add: bls12_381_fp_add, sub: bls12_381_fp_sub,
                    inv: bls12_381_fp_inv_ext)
    r4.forEach { emit($0) }; all += r4

    // BLS12-377 Fq (6-limb base field)
    fputs("\n--- BLS12-377 Fq (C CIOS Montgomery, 6-limb) ---\n", stderr)
    let r4b = bench6(field: "BLS12-377 Fq (377b)", aInit: a6, bInit: b6,
                     mul: bls12_377_fq_mul, sqr: bls12_377_fq_sqr,
                     add: bls12_377_fq_add, sub: bls12_377_fq_sub,
                     inv: bls12_377_fq_inverse)
    r4b.forEach { emit($0) }; all += r4b

    fputs("\n--- BLS12-377 Fr (C CIOS Montgomery, 4-limb) ---\n", stderr)
    let r5 = bench4(field: "BLS12-377 Fr (253b)", aInit: a4, bInit: b4,
                    mul: bls12_377_fr_mul, sqr: bls12_377_fr_sqr,
                    add: bls12_377_fr_add, sub: bls12_377_fr_sub)
    r5.forEach { emit($0) }; all += r5

    fputs("\n--- Stark252 (C CIOS Montgomery, 4-limb) ---\n", stderr)
    let r6 = bench4(field: "Stark252 (252b)", aInit: a4, bInit: b4,
                    mul: stark252_fp_mul, sqr: stark252_fp_sqr,
                    add: stark252_fp_add, sub: stark252_fp_sub)
    r6.forEach { emit($0) }; all += r6

    fputs("\n--- Ed25519 Fp (C Solinas, 4-limb) ---\n", stderr)
    let r7 = bench4(field: "Ed25519 Fp (255b)", aInit: a4, bInit: b4,
                    mul: ed25519_fp_mul, sqr: ed25519_fp_sqr,
                    add: ed25519_fp_add, sub: ed25519_fp_sub,
                    inv: ed25519_fp_inverse)
    r7.forEach { emit($0) }; all += r7

    fputs("\n--- Pallas Fp (C CIOS Montgomery, 4-limb) ---\n", stderr)
    let r8 = bench4(field: "Pallas Fp (255b)", aInit: a4, bInit: b4,
                    mul: pallas_fp_mul, sqr: pallas_fp_sqr,
                    add: pallas_fp_add, sub: pallas_fp_sub)
    r8.forEach { emit($0) }; all += r8

    fputs("\n--- Vesta Fp (C CIOS Montgomery, 4-limb) ---\n", stderr)
    let r9 = bench4(field: "Vesta Fp (255b)", aInit: a4, bInit: b4,
                    mul: vesta_fp_mul, sqr: vesta_fp_sqr,
                    add: vesta_fp_add, sub: vesta_fp_sub)
    r9.forEach { emit($0) }; all += r9

    fputs("\n--- Goldilocks (Swift, 64-bit Solinas) ---\n", stderr)
    let r10 = benchGoldilocks(); r10.forEach { emit($0) }; all += r10

    fputs("\n--- BabyBear (Swift, 31-bit Montgomery) ---\n", stderr)
    let r11 = benchBabyBear(); r11.forEach { emit($0) }; all += r11

    fputs("\n--- Mersenne31 (Swift, 31-bit Mersenne) ---\n", stderr)
    let r12 = benchMersenne31(); r12.forEach { emit($0) }; all += r12

    fputs("\n--- BN254 Fr: ARM64 ASM vs C CIOS Montgomery Mul ---\n", stderr)
    let (asmMul, cMul) = benchAsmVsC()
    fputs(String(format: "  ASM mont_mul: %7.1f ns/op  (%6.1f M ops/s)\n",
                 asmMul, 1e3 / asmMul), stderr)
    fputs(String(format: "  C   mont_mul: %7.1f ns/op  (%6.1f M ops/s)\n",
                 cMul, 1e3 / cMul), stderr)
    let diff = (cMul - asmMul) / cMul * 100
    fputs(String(format: "  ASM advantage: %+.1f%%\n", diff), stderr)

    printSummary(all)
    printFloors()

    fputs("\nField operations benchmark complete.\n", stderr)
}
