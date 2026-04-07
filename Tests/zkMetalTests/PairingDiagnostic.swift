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

    // BLS12-381 pairing bilinearity test: e(G1, G2) * e(-G1, G2) = 1
    let g1_381 = bls12381G1Generator()
    let g2_381 = bls12381G2SimplePoint()
    let negG1_381 = g1_381NegateAffine(g1_381)

    let ok381 = bls12381PairingCheck([(g1_381, g2_381), (negG1_381, g2_381)])
    print("[pairing] BLS12-381 bilinearity check: \(ok381 ? "PASS" : "FAIL")")
}
