// Groth16 Trusted Setup (toy implementation for benchmarking)
import Foundation

public class Groth16Setup {
    public init() {}

    public func setup(r1cs: R1CSInstance) -> (Groth16ProvingKey, Groth16VerificationKey) {
        let m = r1cs.numVars
        let nPub = r1cs.numPublic

        // Domain size: smallest power of 2 >= numConstraints
        var domainN = 1; var logN = 0
        while domainN < r1cs.numConstraints { domainN <<= 1; logN += 1 }

        // Random toxic waste
        let tau = groth16RandomFr()
        let alpha = groth16RandomFr()
        let beta = groth16RandomFr()
        let gamma = groth16RandomFr()
        let delta = groth16RandomFr()

        let gammaInv = frInverse(gamma)
        let deltaInv = frInverse(delta)

        let g1 = pointFromAffine(bn254G1Generator())
        let g2 = g2FromAffine(bn254G2Generator())

        let alpha_g1 = pointScalarMul(g1, alpha)
        let beta_g1 = pointScalarMul(g1, beta)
        let beta_g2 = g2ScalarMul(g2, frToInt(beta))
        let delta_g1 = pointScalarMul(g1, delta)
        let delta_g2 = g2ScalarMul(g2, frToInt(delta))
        let gamma_g2 = g2ScalarMul(g2, frToInt(gamma))

        // Powers of tau for h_query
        var tauPow = [Fr](repeating: .one, count: domainN + 1)
        for i in 1..<tauPow.count { tauPow[i] = frMul(tauPow[i-1], tau) }

        // Compute omega (primitive domainN-th root of unity) and its powers
        let omega = frRootOfUnity(logN: logN)
        var omegaPow = [Fr](repeating: .one, count: domainN)
        for i in 1..<domainN { omegaPow[i] = frMul(omegaPow[i-1], omega) }

        // Compute Lagrange basis values at tau:
        // L_i(tau) = Z(tau) / (domainN * omega^i * (tau - omega^i))
        // where Z(tau) = tau^domainN - 1
        let zTau = frSub(tauPow[domainN], .one)
        let nFr = frFromInt(UInt64(domainN))
        let nInv = frInverse(nFr)
        let zOverN = frMul(zTau, nInv)  // Z(tau) / domainN

        var lagrangeAtTau = [Fr](repeating: .zero, count: domainN)
        for i in 0..<domainN {
            let diff = frSub(tau, omegaPow[i])  // tau - omega^i
            if diff.isZero {
                // tau happens to equal omega^i (vanishingly unlikely with random tau)
                lagrangeAtTau[i] = .one
            } else {
                // L_i(tau) = omega^i * (tau^n - 1) / (n * (tau - omega^i))
                lagrangeAtTau[i] = frMul(frMul(zOverN, omegaPow[i]), frInverse(diff))
            }
        }

        // Evaluate u_j(tau), v_j(tau), w_j(tau) using Lagrange basis
        // u_j(tau) = sum_i A[i,j] * L_i(tau)
        var uAtTau = [Fr](repeating: .zero, count: m)
        var vAtTau = [Fr](repeating: .zero, count: m)
        var wAtTau = [Fr](repeating: .zero, count: m)
        for e in r1cs.aEntries { uAtTau[e.col] = frAdd(uAtTau[e.col], frMul(e.val, lagrangeAtTau[e.row])) }
        for e in r1cs.bEntries { vAtTau[e.col] = frAdd(vAtTau[e.col], frMul(e.val, lagrangeAtTau[e.row])) }
        for e in r1cs.cEntries { wAtTau[e.col] = frAdd(wAtTau[e.col], frMul(e.val, lagrangeAtTau[e.row])) }

        // a_query[j] = [u_j(tau)]_1
        var a_query = [PointProjective]()
        for j in 0..<m { a_query.append(pointScalarMul(g1, uAtTau[j])) }

        // b_g1_query[j] = [v_j(tau)]_1, b_g2_query[j] = [v_j(tau)]_2
        var b_g1_query = [PointProjective]()
        var b_g2_query = [G2ProjectivePoint]()
        for j in 0..<m {
            b_g1_query.append(pointScalarMul(g1, vAtTau[j]))
            b_g2_query.append(g2ScalarMul(g2, frToInt(vAtTau[j])))
        }

        // h_query[i] = [tau^i * Z(tau) / delta]_1 for i = 0..domainN-1
        // Z(tau) = tau^domainN - 1 (vanishing polynomial on NTT domain)
        let zTauOverDelta = frMul(zTau, deltaInv)
        var h_query = [PointProjective]()
        for i in 0..<domainN { h_query.append(pointScalarMul(g1, frMul(tauPow[i], zTauOverDelta))) }

        // ic[j] for j = 0..nPub: (beta*u_j + alpha*v_j + w_j) / gamma
        var ic = [PointProjective]()
        for j in 0...(nPub) {
            let coeff = frMul(frAdd(frAdd(frMul(beta, uAtTau[j]), frMul(alpha, vAtTau[j])), wAtTau[j]), gammaInv)
            ic.append(pointScalarMul(g1, coeff))
        }

        // l_query[j] for witness variables: (beta*u_j + alpha*v_j + w_j) / delta
        var l_query = [PointProjective]()
        for j in (nPub+1)..<m {
            let coeff = frMul(frAdd(frAdd(frMul(beta, uAtTau[j]), frMul(alpha, vAtTau[j])), wAtTau[j]), deltaInv)
            l_query.append(pointScalarMul(g1, coeff))
        }

        let pk = Groth16ProvingKey(
            alpha_g1: alpha_g1, beta_g1: beta_g1, beta_g2: beta_g2,
            delta_g1: delta_g1, delta_g2: delta_g2,
            ic: ic, a_query: a_query, b_g1_query: b_g1_query,
            b_g2_query: b_g2_query, h_query: h_query, l_query: l_query)

        let vk = Groth16VerificationKey(
            alpha_g1: alpha_g1, beta_g2: beta_g2,
            gamma_g2: gamma_g2, delta_g2: delta_g2, ic: ic)

        return (pk, vk)
    }
}

// MARK: - Helpers

public func groth16RandomFr() -> Fr {
    var limbs = [UInt32](repeating: 0, count: 8)
    for i in 0..<8 { limbs[i] = UInt32.random(in: 0...UInt32.max) }
    return frFromLimbs(limbs)
}

// MARK: - Example Circuit: x^3 + x + 5 = y

/// Build R1CS for x^3 + x + 5 = y
/// Variables: [1, x, y, v1, v2] where v1=x*x, v2=v1*x=x^3
/// Constraints:
///   v1 = x * x          -> A=[0,1,0,0,0], B=[0,1,0,0,0], C=[0,0,0,1,0]
///   v2 = v1 * x         -> A=[0,0,0,1,0], B=[0,1,0,0,0], C=[0,0,0,0,1]
///   y = v2 + x + 5      -> A=[5,1,0,0,1], B=[1,0,0,0,0], C=[0,0,1,0,0]
public func buildExampleCircuit() -> R1CSInstance {
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    // Constraint 0: v1 = x * x
    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 1, val: one))
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    // Constraint 1: v2 = v1 * x
    aE.append(R1CSEntry(row: 1, col: 3, val: one))
    bE.append(R1CSEntry(row: 1, col: 1, val: one))
    cE.append(R1CSEntry(row: 1, col: 4, val: one))

    // Constraint 2: y = v2 + x + 5  =>  (5 + x + v2) * 1 = y
    aE.append(R1CSEntry(row: 2, col: 0, val: frFromInt(5)))
    aE.append(R1CSEntry(row: 2, col: 1, val: one))
    aE.append(R1CSEntry(row: 2, col: 4, val: one))
    bE.append(R1CSEntry(row: 2, col: 0, val: one))
    cE.append(R1CSEntry(row: 2, col: 2, val: one))

    return R1CSInstance(numConstraints: 3, numVars: 5, numPublic: 2,
                        aEntries: aE, bEntries: bE, cEntries: cE)
}

/// Compute witness for x^3 + x + 5 = y. Returns (publicInputs=[x, y], witness=[v1, v2])
public func computeExampleWitness(x: UInt64) -> ([Fr], [Fr]) {
    let xFr = frFromInt(x)
    let v1 = frMul(xFr, xFr)
    let v2 = frMul(v1, xFr)
    let y = frAdd(frAdd(v2, xFr), frFromInt(5))
    return ([xFr, y], [v1, v2])
}

/// Build a bench circuit with `numConstraints` multiplication gates: a chain v_i = v_{i-1} * x
public func buildBenchCircuit(numConstraints n: Int) -> (R1CSInstance, [Fr], [Fr]) {
    // Variables: [1, x, y, w0, w1, ..., w_{n-2}]
    // numVars = 3 + (n-1) = n+2
    let numVars = n + 2
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    // Constraint 0: w0 = x * x
    aE.append(R1CSEntry(row: 0, col: 1, val: .one))
    bE.append(R1CSEntry(row: 0, col: 1, val: .one))
    cE.append(R1CSEntry(row: 0, col: 3, val: .one))

    // Constraints 1..n-2: w_i = w_{i-1} * x
    for i in 1..<(n-1) {
        aE.append(R1CSEntry(row: i, col: 2 + i, val: .one))
        bE.append(R1CSEntry(row: i, col: 1, val: .one))
        cE.append(R1CSEntry(row: i, col: 3 + i, val: .one))
    }

    // Last constraint: y = w_{n-2} + 1  =>  (1 + w_{n-2}) * 1 = y
    let lastWit = n == 1 ? 1 : (numVars - 1)  // col of last w
    aE.append(R1CSEntry(row: n-1, col: 0, val: .one))
    aE.append(R1CSEntry(row: n-1, col: lastWit, val: .one))
    bE.append(R1CSEntry(row: n-1, col: 0, val: .one))
    cE.append(R1CSEntry(row: n-1, col: 2, val: .one))

    // Compute witness for x=3
    let x = frFromInt(3)
    var ws = [Fr]()
    var cur = frMul(x, x)  // w0 = x^2
    ws.append(cur)
    for _ in 1..<(n-1) { cur = frMul(cur, x); ws.append(cur) }
    let y = frAdd(cur, .one)

    let r1cs = R1CSInstance(numConstraints: n, numVars: numVars, numPublic: 2,
                             aEntries: aE, bEntries: bE, cEntries: cE)
    return (r1cs, [x, y], ws)
}
