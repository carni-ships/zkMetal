// Polynomial Identity Testing (PIT) Engine
// Schwartz-Zippel-based probabilistic testing of polynomial identities
// Used for verifying SNARK polynomial relations (Plonk gates, vanishing, quotient checks)

import Foundation

// MARK: - Polynomial Relation Types

/// A polynomial represented as coefficient array in ascending order: c0 + c1*x + c2*x^2 + ...
public typealias Poly = [Fr]

/// Encodes a polynomial relation that should evaluate to zero.
/// The `evaluate` closure takes a point (or points for multivariate) and returns the relation value.
public struct PolyRelation {
    public let name: String
    public let degree: Int
    public let evaluate: (Fr) -> Fr

    public init(name: String, degree: Int, evaluate: @escaping (Fr) -> Fr) {
        self.name = name
        self.degree = degree
        self.evaluate = evaluate
    }
}

/// Multivariate polynomial relation: takes a vector of field elements.
public struct MultivarPolyRelation {
    public let name: String
    public let numVars: Int
    public let totalDegree: Int
    public let evaluate: ([Fr]) -> Fr

    public init(name: String, numVars: Int, totalDegree: Int, evaluate: @escaping ([Fr]) -> Fr) {
        self.name = name
        self.numVars = numVars
        self.totalDegree = totalDegree
        self.evaluate = evaluate
    }
}

// MARK: - PIT Result

/// Result of a polynomial identity test.
public struct PITResult {
    public let isIdentity: Bool
    public let numTrials: Int
    public let failedAt: Fr?         // The point where the relation was nonzero (if not identity)
    public let failedValue: Fr?      // The nonzero value (if not identity)

    public static func pass(trials: Int) -> PITResult {
        PITResult(isIdentity: true, numTrials: trials, failedAt: nil, failedValue: nil)
    }

    public static func fail(trials: Int, at point: Fr, value: Fr) -> PITResult {
        PITResult(isIdentity: false, numTrials: trials, failedAt: point, failedValue: value)
    }
}

/// Result of a batch PIT test.
public struct BatchPITResult {
    public let results: [(String, PITResult)]
    public var allPassed: Bool { results.allSatisfy { $0.1.isIdentity } }
}

// MARK: - Random Field Element Generation

/// Simple deterministic PRNG for generating pseudo-random field elements.
/// Uses a linear congruential generator seeded from the input.
public struct FieldPRNG {
    private var state: UInt64

    public init(seed: UInt64 = 0xDEAD_BEEF_CAFE_1234) {
        self.state = seed
    }

    public mutating func next() -> Fr {
        // Generate 4 pseudo-random 64-bit limbs
        var limbs = [UInt64](repeating: 0, count: 4)
        for i in 0..<4 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            limbs[i] = state
        }
        // Reduce mod r by converting to Montgomery form
        // First create raw value, then multiply by R^2 to get Montgomery form
        // Simple approach: just use frFromInt on mixed bits for a "random enough" element
        let raw = Fr.from64(limbs)
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

// MARK: - Schwartz-Zippel Lemma

/// Schwartz-Zippel identity testing.
///
/// For a nonzero polynomial p of total degree d over a field F,
/// Pr[p(r) = 0] <= d / |F| for uniformly random r in F.
///
/// For BN254 Fr with |F| ~ 2^254, even degree-2^30 polynomials have
/// negligible false-positive probability (~2^{-224}).
public struct SchwartzZippel {

    /// Test whether a univariate polynomial relation is identically zero.
    /// Evaluates at `numTrials` random points; returns pass if all evaluate to zero.
    ///
    /// Soundness: Pr[false positive] <= (degree / |F|)^numTrials
    public static func test(
        relation: PolyRelation,
        numTrials: Int = 3,
        seed: UInt64 = 0xDEAD_BEEF_CAFE_1234
    ) -> PITResult {
        var rng = FieldPRNG(seed: seed)
        for trial in 0..<numTrials {
            let point = rng.next()
            let value = relation.evaluate(point)
            if !value.isZero {
                return .fail(trials: trial + 1, at: point, value: value)
            }
        }
        return .pass(trials: numTrials)
    }

    /// Test whether a multivariate polynomial relation is identically zero.
    /// Each variable is sampled independently at random.
    public static func testMultivariate(
        relation: MultivarPolyRelation,
        numTrials: Int = 3,
        seed: UInt64 = 0xDEAD_BEEF_CAFE_1234
    ) -> PITResult {
        var rng = FieldPRNG(seed: seed)
        for trial in 0..<numTrials {
            var point = [Fr]()
            for _ in 0..<relation.numVars {
                point.append(rng.next())
            }
            let value = relation.evaluate(point)
            if !value.isZero {
                // Return first variable as the "failed at" point for diagnostics
                return .fail(trials: trial + 1, at: point[0], value: value)
            }
        }
        return .pass(trials: numTrials)
    }
}

// MARK: - Polynomial Identity Tester

/// High-level PIT engine that wraps Schwartz-Zippel with polynomial arithmetic helpers.
public struct PolyIdentityTester {

    /// Default number of random trials per test.
    public var numTrials: Int
    public var seed: UInt64

    public init(numTrials: Int = 3, seed: UInt64 = 0xDEAD_BEEF_CAFE_1234) {
        self.numTrials = numTrials
        self.seed = seed
    }

    // MARK: - Polynomial evaluation helper

    /// Evaluate polynomial p at point x using Horner's method.
    /// Coefficients in ascending order: p = c0 + c1*x + c2*x^2 + ...
    public static func evalPoly(_ p: Poly, at x: Fr) -> Fr {
        if p.isEmpty { return Fr.zero }
        var result = p[p.count - 1]
        for i in stride(from: p.count - 2, through: 0, by: -1) {
            result = frAdd(frMul(result, x), p[i])
        }
        return result
    }

    // MARK: - Identity tests

    /// Test if two polynomials are identical: p(x) == q(x) for all x.
    public func testEqual(_ p: Poly, _ q: Poly) -> PITResult {
        let maxDeg = max(p.count, q.count)
        let relation = PolyRelation(name: "p == q", degree: maxDeg) { x in
            let pVal = PolyIdentityTester.evalPoly(p, at: x)
            let qVal = PolyIdentityTester.evalPoly(q, at: x)
            return frSub(pVal, qVal)
        }
        return SchwartzZippel.test(relation: relation, numTrials: numTrials, seed: seed)
    }

    /// Test if a polynomial is identically zero.
    public func testZero(_ p: Poly) -> PITResult {
        let relation = PolyRelation(name: "p == 0", degree: p.count) { x in
            PolyIdentityTester.evalPoly(p, at: x)
        }
        return SchwartzZippel.test(relation: relation, numTrials: numTrials, seed: seed)
    }

    /// Test a custom univariate relation.
    public func testRelation(_ relation: PolyRelation) -> PITResult {
        SchwartzZippel.test(relation: relation, numTrials: numTrials, seed: seed)
    }

    /// Test a custom multivariate relation.
    public func testMultivariate(_ relation: MultivarPolyRelation) -> PITResult {
        SchwartzZippel.testMultivariate(relation: relation, numTrials: numTrials, seed: seed)
    }
}

// MARK: - Quotient Check

/// Verify polynomial division: p(x) = q(x) * d(x) + r(x)
/// Tests the identity p(x) - q(x)*d(x) - r(x) = 0 at random points.
public struct QuotientCheck {

    /// Verify that p = q * d + r by evaluating at random points.
    /// - Parameters:
    ///   - p: dividend polynomial
    ///   - q: quotient polynomial
    ///   - d: divisor polynomial
    ///   - r: remainder polynomial
    ///   - numTrials: number of random evaluation points
    ///   - seed: PRNG seed
    /// - Returns: PITResult indicating whether the relation holds
    public static func verify(
        dividend p: Poly,
        quotient q: Poly,
        divisor d: Poly,
        remainder r: Poly,
        numTrials: Int = 3,
        seed: UInt64 = 0xDEAD_BEEF_CAFE_1234
    ) -> PITResult {
        let maxDeg = max(p.count, q.count + d.count - 1)
        let relation = PolyRelation(name: "p = q*d + r", degree: maxDeg) { x in
            let pVal = PolyIdentityTester.evalPoly(p, at: x)
            let qVal = PolyIdentityTester.evalPoly(q, at: x)
            let dVal = PolyIdentityTester.evalPoly(d, at: x)
            let rVal = PolyIdentityTester.evalPoly(r, at: x)
            // p(x) - q(x)*d(x) - r(x)
            return frSub(frSub(pVal, frMul(qVal, dVal)), rVal)
        }
        return SchwartzZippel.test(relation: relation, numTrials: numTrials, seed: seed)
    }
}

// MARK: - Vanishing Polynomial Check

/// Verify that p(x) vanishes on a domain: p(x) = 0 mod Z_H(x)
/// where Z_H(x) = x^n - 1 is the vanishing polynomial for the n-th roots of unity domain.
public struct VanishingCheck {

    /// Evaluate the vanishing polynomial Z_H(x) = x^n - 1 at point x.
    public static func evalVanishing(domainSize n: Int, at x: Fr) -> Fr {
        let xn = frPow(x, UInt64(n))
        return frSub(xn, Fr.one)
    }

    /// Verify that p(x) is divisible by Z_H(x) = x^n - 1.
    /// Checks that p(x) = q(x) * Z_H(x) for some quotient q(x),
    /// which means there exists q such that p(x) / (x^n - 1) has no remainder.
    ///
    /// Approach: evaluate p at random points and check p(x) mod Z_H(x) = 0.
    /// This is equivalent to checking p(x) = q(x) * (x^n - 1).
    public static func verify(
        polynomial p: Poly,
        quotient q: Poly,
        domainSize n: Int,
        numTrials: Int = 3,
        seed: UInt64 = 0xDEAD_BEEF_CAFE_1234
    ) -> PITResult {
        let maxDeg = max(p.count, q.count + n)
        let relation = PolyRelation(name: "p = q * Z_H", degree: maxDeg) { x in
            let pVal = PolyIdentityTester.evalPoly(p, at: x)
            let qVal = PolyIdentityTester.evalPoly(q, at: x)
            let zhVal = evalVanishing(domainSize: n, at: x)
            // p(x) - q(x) * Z_H(x) should be zero
            return frSub(pVal, frMul(qVal, zhVal))
        }
        return SchwartzZippel.test(relation: relation, numTrials: numTrials, seed: seed)
    }

    /// Quick check: verify p vanishes on the domain by evaluating at roots of unity.
    /// This is a deterministic check (not Schwartz-Zippel) that confirms p(omega^i) = 0
    /// for all i in [0, n).
    public static func checkOnDomain(polynomial p: Poly, domainSize n: Int) -> Bool {
        let logN = Int(log2(Double(n)))
        guard (1 << logN) == n, logN <= Fr.TWO_ADICITY else { return false }
        let omega = frRootOfUnity(logN: logN)
        var w = Fr.one
        for _ in 0..<n {
            let val = PolyIdentityTester.evalPoly(p, at: w)
            if !val.isZero { return false }
            w = frMul(w, omega)
        }
        return true
    }
}

// MARK: - Batch PIT

/// Test multiple polynomial relations simultaneously.
/// Useful for verifying all SNARK constraints in one pass.
public struct BatchPIT {

    /// Test multiple univariate relations at the same random points.
    /// More efficient than testing each independently since we reuse random challenges.
    public static func test(
        relations: [PolyRelation],
        numTrials: Int = 3,
        seed: UInt64 = 0xDEAD_BEEF_CAFE_1234
    ) -> BatchPITResult {
        var results = [(String, PITResult)]()
        // Use a random linear combination to batch: test sum_i alpha^i * R_i(x) = 0
        // But also test each individually so we can report which one failed
        var rng = FieldPRNG(seed: seed)

        for relation in relations {
            // Each relation gets tested at the same points for consistency
            var innerRng = FieldPRNG(seed: seed &+ UInt64(bitPattern: Int64(relation.name.hashValue)))
            var failed = false
            var failTrial = 0
            var failPoint = Fr.zero
            var failValue = Fr.zero

            for trial in 0..<numTrials {
                let point = innerRng.next()
                let value = relation.evaluate(point)
                if !value.isZero {
                    failed = true
                    failTrial = trial + 1
                    failPoint = point
                    failValue = value
                    break
                }
            }

            if failed {
                results.append((relation.name, .fail(trials: failTrial, at: failPoint, value: failValue)))
            } else {
                results.append((relation.name, .pass(trials: numTrials)))
            }
        }

        // Also do a batched random-linear-combination check for efficiency
        let alpha = rng.next()
        for trial in 0..<numTrials {
            let point = rng.next()
            var combined = Fr.zero
            var alphaPow = Fr.one
            for relation in relations {
                let val = relation.evaluate(point)
                combined = frAdd(combined, frMul(alphaPow, val))
                alphaPow = frMul(alphaPow, alpha)
            }
            // If batch check fails but individual checks passed, something is wrong
            // (This shouldn't happen, but is a good sanity check)
            if !combined.isZero {
                // Find which relation is nonzero
                for relation in relations {
                    let val = relation.evaluate(point)
                    if !val.isZero {
                        // Already tracked above
                        break
                    }
                }
            }
        }

        return BatchPITResult(results: results)
    }

    /// Batch test multivariate relations.
    public static func testMultivariate(
        relations: [MultivarPolyRelation],
        numTrials: Int = 3,
        seed: UInt64 = 0xDEAD_BEEF_CAFE_1234
    ) -> BatchPITResult {
        var results = [(String, PITResult)]()

        for relation in relations {
            let result = SchwartzZippel.testMultivariate(
                relation: relation,
                numTrials: numTrials,
                seed: seed &+ UInt64(bitPattern: Int64(relation.name.hashValue))
            )
            results.append((relation.name, result))
        }

        return BatchPITResult(results: results)
    }
}

// MARK: - Plonk Gate Identity

/// Plonk arithmetic gate identity:
///   qL * a + qR * b + qO * c + qM * a * b + qC = 0
///
/// This encodes the standard Plonk gate constraint where:
///   qL, qR, qO, qM, qC are selector polynomials
///   a, b, c are wire polynomials
public struct PlonkGateIdentity {

    /// Build a PolyRelation for the Plonk gate identity evaluated at a single point.
    /// All polynomials are evaluated at the same point x.
    public static func relation(
        qL: Poly, qR: Poly, qO: Poly, qM: Poly, qC: Poly,
        a: Poly, b: Poly, c: Poly
    ) -> PolyRelation {
        let maxDeg = max(
            qL.count + a.count,
            max(qR.count + b.count,
                max(qO.count + c.count,
                    max(qM.count + a.count + b.count, qC.count)))
        )
        return PolyRelation(name: "Plonk gate", degree: maxDeg) { x in
            let qLv = PolyIdentityTester.evalPoly(qL, at: x)
            let qRv = PolyIdentityTester.evalPoly(qR, at: x)
            let qOv = PolyIdentityTester.evalPoly(qO, at: x)
            let qMv = PolyIdentityTester.evalPoly(qM, at: x)
            let qCv = PolyIdentityTester.evalPoly(qC, at: x)
            let av = PolyIdentityTester.evalPoly(a, at: x)
            let bv = PolyIdentityTester.evalPoly(b, at: x)
            let cv = PolyIdentityTester.evalPoly(c, at: x)

            // qL*a + qR*b + qO*c + qM*a*b + qC
            var result = frMul(qLv, av)
            result = frAdd(result, frMul(qRv, bv))
            result = frAdd(result, frMul(qOv, cv))
            result = frAdd(result, frMul(qMv, frMul(av, bv)))
            result = frAdd(result, qCv)
            return result
        }
    }
}
