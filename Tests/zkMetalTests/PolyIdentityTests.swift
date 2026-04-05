import zkMetal

public func runPolyIdentityTests() {
    suite("PolyIdentityTest")

    // --- Basic identity: (x+1)(x-1) = x^2 - 1 ---
    do {
        let pit = PolyIdentityTester(numTrials: 5)

        // (x+1) = 1 + x  -> coeffs [1, 1]
        // (x-1) = -1 + x -> coeffs [-1, 1]
        // x^2 - 1         -> coeffs [-1, 0, 1]
        let one = Fr.one
        let negOne = frNeg(Fr.one)

        let xPlus1: Poly = [one, one]            // 1 + x
        let xMinus1: Poly = [negOne, one]         // -1 + x
        let x2Minus1: Poly = [negOne, Fr.zero, one] // -1 + x^2

        // Test: (x+1)*(x-1) - (x^2-1) = 0
        let relation = PolyRelation(name: "(x+1)(x-1) = x^2-1", degree: 2) { x in
            let lhs = frMul(
                PolyIdentityTester.evalPoly(xPlus1, at: x),
                PolyIdentityTester.evalPoly(xMinus1, at: x)
            )
            let rhs = PolyIdentityTester.evalPoly(x2Minus1, at: x)
            return frSub(lhs, rhs)
        }
        let result = pit.testRelation(relation)
        expect(result.isIdentity, "(x+1)(x-1) = x^2-1")
    }

    // --- Non-identity detected: x^2 + 1 != x^2 - 1 ---
    do {
        let pit = PolyIdentityTester(numTrials: 5)

        let one = Fr.one
        let negOne = frNeg(Fr.one)
        let x2Plus1: Poly = [one, Fr.zero, one]     // 1 + x^2
        let x2Minus1: Poly = [negOne, Fr.zero, one]  // -1 + x^2

        let result = pit.testEqual(x2Plus1, x2Minus1)
        expect(!result.isIdentity, "x^2+1 != x^2-1 detected")
        expect(result.failedValue != nil, "Non-identity has counterexample")
    }

    // --- Zero polynomial identity ---
    do {
        let pit = PolyIdentityTester(numTrials: 5)
        let zero: Poly = [Fr.zero, Fr.zero, Fr.zero]
        let result = pit.testZero(zero)
        expect(result.isIdentity, "Zero polynomial is zero")
    }

    // --- Quotient check: polynomial long division ---
    // p(x) = x^3 + 2x^2 + 3x + 4
    // d(x) = x + 1
    // p(x) / d(x) = q(x) with remainder r
    // Verify: p = q * d + r
    do {
        let one = Fr.one
        let two = frFromInt(2)
        let three = frFromInt(3)
        let four = frFromInt(4)

        // p(x) = 4 + 3x + 2x^2 + x^3
        let p: Poly = [four, three, two, one]
        // d(x) = 1 + x
        let d: Poly = [one, one]

        // Manual long division: x^3 + 2x^2 + 3x + 4 divided by (x + 1)
        // q(x) = x^2 + x + 2, r = 2
        let q: Poly = [two, one, one]
        let r: Poly = [two]

        let result = QuotientCheck.verify(dividend: p, quotient: q, divisor: d, remainder: r)
        expect(result.isIdentity, "Quotient check: x^3+2x^2+3x+4 = (x^2+x+2)(x+1) + 2")

        // Wrong remainder should fail
        let wrongR: Poly = [three]
        let resultBad = QuotientCheck.verify(dividend: p, quotient: q, divisor: d, remainder: wrongR)
        expect(!resultBad.isIdentity, "Wrong remainder detected")
    }

    // --- Vanishing polynomial check ---
    // Build a polynomial that vanishes on 4th roots of unity (domain size 4)
    // Z_H(x) = x^4 - 1
    // If p(x) = (x^4 - 1) * q(x), then p vanishes on the domain
    do {
        let domainSize = 4
        let one = Fr.one
        let negOne = frNeg(one)

        // q(x) = 1 + x  (arbitrary quotient)
        let q: Poly = [one, one]

        // p(x) = q(x) * Z_H(x) = (1 + x)(x^4 - 1)
        // = x^4 - 1 + x^5 - x
        // = -1 - x + 0x^2 + 0x^3 + x^4 + x^5
        let p: Poly = [negOne, negOne, Fr.zero, Fr.zero, one, one]

        let result = VanishingCheck.verify(
            polynomial: p, quotient: q, domainSize: domainSize
        )
        expect(result.isIdentity, "Vanishing check: p = q * Z_H")

        // Also verify p actually vanishes on the domain roots
        let vanishes = VanishingCheck.checkOnDomain(polynomial: p, domainSize: domainSize)
        expect(vanishes, "p vanishes on 4th roots of unity")

        // A polynomial that does NOT vanish on the domain
        let pBad: Poly = [one, one] // 1 + x (does not vanish at x=1)
        let notVanishes = VanishingCheck.checkOnDomain(polynomial: pBad, domainSize: domainSize)
        expect(!notVanishes, "1+x does not vanish on domain")
    }

    // --- Batch identity testing ---
    do {
        let one = Fr.one
        let negOne = frNeg(one)
        let two = frFromInt(2)

        // Relation 1: (x+1)^2 = x^2 + 2x + 1  (TRUE)
        let r1 = PolyRelation(name: "(x+1)^2 = x^2+2x+1", degree: 2) { x in
            let xp1 = frAdd(x, one)
            let lhs = frMul(xp1, xp1)
            // x^2 + 2x + 1
            let rhs = frAdd(frAdd(frMul(x, x), frMul(two, x)), one)
            return frSub(lhs, rhs)
        }

        // Relation 2: x * (-x) = -x^2  (TRUE)
        let r2 = PolyRelation(name: "x*(-x) = -x^2", degree: 2) { x in
            let lhs = frMul(x, frNeg(x))
            let rhs = frNeg(frMul(x, x))
            return frSub(lhs, rhs)
        }

        // Relation 3: x + 1 = x (FALSE)
        let r3 = PolyRelation(name: "x+1 = x (false)", degree: 1) { x in
            frSub(frAdd(x, one), x)
        }

        let batchResult = BatchPIT.test(relations: [r1, r2, r3], numTrials: 3)
        expect(!batchResult.allPassed, "Batch: not all pass (one is false)")
        expect(batchResult.results[0].1.isIdentity, "Batch: (x+1)^2 passes")
        expect(batchResult.results[1].1.isIdentity, "Batch: x*(-x) passes")
        expect(!batchResult.results[2].1.isIdentity, "Batch: x+1=x fails")
    }

    // --- Multivariate Schwartz-Zippel ---
    // Test: f(x,y) = x*y - y*x = 0 (commutativity)
    do {
        let commRelation = MultivarPolyRelation(
            name: "xy - yx = 0",
            numVars: 2,
            totalDegree: 2
        ) { vars in
            let x = vars[0], y = vars[1]
            return frSub(frMul(x, y), frMul(y, x))
        }
        let result = SchwartzZippel.testMultivariate(relation: commRelation, numTrials: 5)
        expect(result.isIdentity, "Multivariate: xy = yx")

        // Non-identity: f(x,y) = x^2 + y^2 - 1 (not identically zero)
        let circleRelation = MultivarPolyRelation(
            name: "x^2+y^2-1 (not zero)",
            numVars: 2,
            totalDegree: 2
        ) { vars in
            let x = vars[0], y = vars[1]
            return frSub(frAdd(frMul(x, x), frMul(y, y)), Fr.one)
        }
        let result2 = SchwartzZippel.testMultivariate(relation: circleRelation, numTrials: 5)
        expect(!result2.isIdentity, "Multivariate: x^2+y^2-1 is not zero")
    }

    // --- Multivariate: sumcheck-style identity ---
    // Test: f(x,y,z) = (x+y+z)^2 - x^2 - y^2 - z^2 - 2xy - 2xz - 2yz = 0
    do {
        let two = frFromInt(2)
        let expandRelation = MultivarPolyRelation(
            name: "expansion identity",
            numVars: 3,
            totalDegree: 2
        ) { vars in
            let x = vars[0], y = vars[1], z = vars[2]
            // (x+y+z)^2
            let sum = frAdd(frAdd(x, y), z)
            let lhs = frMul(sum, sum)
            // x^2 + y^2 + z^2 + 2xy + 2xz + 2yz
            var rhs = frAdd(frMul(x, x), frMul(y, y))
            rhs = frAdd(rhs, frMul(z, z))
            rhs = frAdd(rhs, frMul(two, frMul(x, y)))
            rhs = frAdd(rhs, frMul(two, frMul(x, z)))
            rhs = frAdd(rhs, frMul(two, frMul(y, z)))
            return frSub(lhs, rhs)
        }
        let result = SchwartzZippel.testMultivariate(relation: expandRelation, numTrials: 5)
        expect(result.isIdentity, "Multivariate: (x+y+z)^2 expansion")
    }

    // --- Plonk gate identity ---
    // Encode a simple addition gate: a + b - c = 0
    // qL=1, qR=1, qO=-1, qM=0, qC=0
    // Wire polynomials: a(x) = 3+2x, b(x) = 1+x, c(x) = a(x)+b(x) = 4+3x
    do {
        let one = Fr.one
        let negOne = frNeg(one)
        let two = frFromInt(2)
        let three = frFromInt(3)
        let four = frFromInt(4)

        // Selector polynomials (constants for a single gate type)
        let qL: Poly = [one]
        let qR: Poly = [one]
        let qO: Poly = [negOne]
        let qM: Poly = [Fr.zero]
        let qC: Poly = [Fr.zero]

        // Wire polynomials
        let a: Poly = [three, two]    // 3 + 2x
        let b: Poly = [one, one]      // 1 + x
        let c: Poly = [four, three]   // 4 + 3x = a + b

        let relation = PlonkGateIdentity.relation(
            qL: qL, qR: qR, qO: qO, qM: qM, qC: qC,
            a: a, b: b, c: c
        )
        let pit = PolyIdentityTester(numTrials: 5)
        let result = pit.testRelation(relation)
        expect(result.isIdentity, "Plonk gate: a + b - c = 0")
    }

    // --- Plonk multiplication gate: a * b - c = 0 ---
    // qL=0, qR=0, qO=-1, qM=1, qC=0
    do {
        let one = Fr.one
        let negOne = frNeg(one)
        let two = frFromInt(2)
        let three = frFromInt(3)

        let qL: Poly = [Fr.zero]
        let qR: Poly = [Fr.zero]
        let qO: Poly = [negOne]
        let qM: Poly = [one]
        let qC: Poly = [Fr.zero]

        // a(x) = 2 + x, b(x) = 3 + x
        // c(x) = a*b = (2+x)(3+x) = 6 + 5x + x^2
        let a: Poly = [two, one]
        let b: Poly = [three, one]
        let five = frFromInt(5)
        let six = frFromInt(6)
        let c: Poly = [six, five, one]  // 6 + 5x + x^2

        let relation = PlonkGateIdentity.relation(
            qL: qL, qR: qR, qO: qO, qM: qM, qC: qC,
            a: a, b: b, c: c
        )
        let pit = PolyIdentityTester(numTrials: 5)
        let result = pit.testRelation(relation)
        expect(result.isIdentity, "Plonk gate: a * b - c = 0")
    }

    // --- Plonk gate with wrong wire should fail ---
    do {
        let one = Fr.one
        let negOne = frNeg(one)
        let two = frFromInt(2)
        let three = frFromInt(3)

        let qL: Poly = [one]
        let qR: Poly = [one]
        let qO: Poly = [negOne]
        let qM: Poly = [Fr.zero]
        let qC: Poly = [Fr.zero]

        let a: Poly = [three, two]
        let b: Poly = [one, one]
        let cWrong: Poly = [one, one]  // Wrong: should be [4, 3]

        let relation = PlonkGateIdentity.relation(
            qL: qL, qR: qR, qO: qO, qM: qM, qC: qC,
            a: a, b: b, c: cWrong
        )
        let pit = PolyIdentityTester(numTrials: 5)
        let result = pit.testRelation(relation)
        expect(!result.isIdentity, "Plonk gate: wrong wire detected")
    }

    // --- Plonk constant gate: qC = c (qL=qR=qM=0, qO=-1) ---
    do {
        let negOne = frNeg(Fr.one)
        let seven = frFromInt(7)

        let qL: Poly = [Fr.zero]
        let qR: Poly = [Fr.zero]
        let qO: Poly = [negOne]
        let qM: Poly = [Fr.zero]
        let qC: Poly = [seven]
        let a: Poly = [Fr.zero]
        let b: Poly = [Fr.zero]
        let c: Poly = [seven]  // c = 7

        let relation = PlonkGateIdentity.relation(
            qL: qL, qR: qR, qO: qO, qM: qM, qC: qC,
            a: a, b: b, c: c
        )
        let pit = PolyIdentityTester(numTrials: 5)
        let result = pit.testRelation(relation)
        expect(result.isIdentity, "Plonk gate: constant 7 = c")
    }
}
