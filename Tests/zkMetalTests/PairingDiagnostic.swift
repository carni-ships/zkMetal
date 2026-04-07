// BN254 pairing bilinearity test
import zkMetal

func runPairingDiagnostic() {
    let g1a = bn254G1Generator()
    let g2a = bn254G2Generator()

    let negG1a = pointNegateAffine(g1a)
    let g1proj = pointFromAffine(g1a)
    let twoG1 = pointDouble(g1proj)
    guard let twoG1a = pointToAffine(twoG1) else {
        print("[pairing] FAIL: could not convert 2G1 to affine")
        return
    }

    // C path: fully in C — e(2G1, G2) * e(-G1, G2) * e(-G1, G2) should = 1
    let ok = cBN254PairingCheck([(twoG1a, g2a), (negG1a, g2a), (negG1a, g2a)])
    print("[pairing] bilinearity check: \(ok ? "PASS" : "FAIL")")
}
