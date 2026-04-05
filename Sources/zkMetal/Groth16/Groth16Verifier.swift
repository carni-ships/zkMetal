// Groth16 Verifier -- BN254 pairing-based verification
// Uses C-accelerated BN254 pairing via __uint128_t CIOS Montgomery arithmetic
import Foundation
import NeonFieldOps

public class Groth16Verifier {
    public init() {}
    public func verify(proof: Groth16Proof, vk: Groth16VerificationKey, publicInputs: [Fr]) -> Bool {
        precondition(publicInputs.count + 1 == vk.ic.count)
        var vkX = vk.ic[0]
        for i in 0..<publicInputs.count {
            if !publicInputs[i].isZero { vkX = pointAdd(vkX, pointScalarMul(vk.ic[i+1], publicInputs[i])) }
        }
        guard let pA = pointToAffine(proof.a), let pC = pointToAffine(proof.c),
              let al = pointToAffine(vk.alpha_g1), let vx = pointToAffine(vkX) else { return false }
        guard let pB = g2ToAffine(proof.b), let be = g2ToAffine(vk.beta_g2),
              let ga = g2ToAffine(vk.gamma_g2), let de = g2ToAffine(vk.delta_g2) else { return false }
        return bn254PairingCheck([(pointNegateAffine(pA), pB), (al, be), (vx, ga), (pC, de)])
    }
}
