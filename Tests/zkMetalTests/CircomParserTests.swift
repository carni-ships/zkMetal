// Tests for Circom R1CS and witness (.wtns) binary parsers
import zkMetal
import Foundation

// MARK: - Test Helpers: Build binary blobs

/// Build the BN254 Fr prime as 32 LE bytes.
private func bn254FrPrimeBytes() -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: 32)
    let limbs: [UInt64] = Fr.P
    for i in 0..<4 {
        let v = limbs[i]
        for j in 0..<8 {
            bytes[i * 8 + j] = UInt8((v >> (j * 8)) & 0xFF)
        }
    }
    return bytes
}

/// Encode a UInt32 as 4 LE bytes.
private func le32(_ v: UInt32) -> [UInt8] {
    return [UInt8(v & 0xFF), UInt8((v >> 8) & 0xFF),
            UInt8((v >> 16) & 0xFF), UInt8((v >> 24) & 0xFF)]
}

/// Encode a UInt64 as 8 LE bytes.
private func le64(_ v: UInt64) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: 8)
    for i in 0..<8 { bytes[i] = UInt8((v >> (i * 8)) & 0xFF) }
    return bytes
}

/// Encode a field element (standard form UInt64 value) as 32 LE bytes.
private func fieldBytes(_ val: UInt64) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: 32)
    for i in 0..<8 { bytes[i] = UInt8((val >> (i * 8)) & 0xFF) }
    return bytes
}

/// Build a section: type (UInt32) + size (UInt64) + data.
private func section(_ type: UInt32, _ data: [UInt8]) -> [UInt8] {
    return le32(type) + le64(UInt64(data.count)) + data
}

/// Build a sparse vector: nTerms (UInt32) + [wireId (UInt32) + coeff (32 bytes)]...
private func sparseVec(_ terms: [(wireId: UInt32, coeff: UInt64)]) -> [UInt8] {
    var bytes = le32(UInt32(terms.count))
    for (wid, c) in terms {
        bytes += le32(wid)
        bytes += fieldBytes(c)
    }
    return bytes
}

/// Build a minimal R1CS binary for the circuit: v = x * x (single multiply gate)
/// Variables: [one(0), x(1), v(2)]
/// Constraint: A=[x], B=[x], C=[v]  i.e. x * x = v
/// nOutputs=1 (v is output), nPubInputs=0, nPrivInputs=1 (x is private)
/// nWires=3, nConstraints=1
private func buildSimpleR1CSBinary() -> Data {
    let prime = bn254FrPrimeBytes()

    // Header section (type 1)
    var header = [UInt8]()
    header += le32(32)          // fieldSize
    header += prime             // prime (32 bytes)
    header += le32(3)           // nWires
    header += le32(1)           // nOutputs
    header += le32(0)           // nPubInputs
    header += le32(1)           // nPrivInputs
    header += le64(3)           // nLabels
    header += le32(1)           // nConstraints

    // Constraint section (type 2): x * x = v
    // Wire layout: [one(0), v(1)=output, x(2)=private]
    var constraints = [UInt8]()
    constraints += sparseVec([(wireId: 2, coeff: 1)])   // A: wire 2 (x), coeff 1
    constraints += sparseVec([(wireId: 2, coeff: 1)])   // B: wire 2 (x), coeff 1
    constraints += sparseVec([(wireId: 1, coeff: 1)])   // C: wire 1 (v), coeff 1

    // Wire-to-label section (type 3)
    var labels = [UInt8]()
    labels += le64(0)   // wire 0 -> label 0
    labels += le64(1)   // wire 1 -> label 1
    labels += le64(2)   // wire 2 -> label 2

    // Assemble file
    var file = [UInt8]()
    file += [0x72, 0x31, 0x63, 0x73]   // magic "r1cs"
    file += le32(1)                      // version
    file += le32(3)                      // 3 sections
    file += section(1, header)
    file += section(2, constraints)
    file += section(3, labels)

    return Data(file)
}

/// Build an R1CS binary for: y = x^2 + x + 5
/// Variables: [one(0), y(1), x(2), v(3)]
///   wire 0 = "one", wire 1 = output y, wire 2 = private x, wire 3 = intermediate v = x*x
/// Constraints:
///   (0) v = x * x           -> A=[x], B=[x], C=[v]
///   (1) y = v + x + 5*one   -> A=[5*one + x + v], B=[one], C=[y]
/// nOutputs=1, nPubInputs=0, nPrivInputs=1, nWires=4, nConstraints=2
private func buildQuadraticR1CSBinary() -> Data {
    let prime = bn254FrPrimeBytes()

    var header = [UInt8]()
    header += le32(32)
    header += prime
    header += le32(4)   // nWires
    header += le32(1)   // nOutputs (y)
    header += le32(0)   // nPubInputs
    header += le32(1)   // nPrivInputs (x)
    header += le64(4)   // nLabels
    header += le32(2)   // nConstraints

    // Constraint 0: x * x = v
    var c0 = [UInt8]()
    c0 += sparseVec([(wireId: 2, coeff: 1)])    // A: x
    c0 += sparseVec([(wireId: 2, coeff: 1)])    // B: x
    c0 += sparseVec([(wireId: 3, coeff: 1)])    // C: v

    // Constraint 1: (5 + x + v) * 1 = y
    var c1 = [UInt8]()
    c1 += sparseVec([(wireId: 0, coeff: 5), (wireId: 2, coeff: 1), (wireId: 3, coeff: 1)])  // A
    c1 += sparseVec([(wireId: 0, coeff: 1)])    // B: one
    c1 += sparseVec([(wireId: 1, coeff: 1)])    // C: y

    var constraints = c0 + c1

    var file = [UInt8]()
    file += [0x72, 0x31, 0x63, 0x73]
    file += le32(1)
    file += le32(2)     // 2 sections (no labels)
    file += section(1, header)
    file += section(2, constraints)

    return Data(file)
}

/// Build a .wtns binary for witness values [one, y, x, v] where x=3, v=9, y=17 (x^2+x+5)
private func buildQuadraticWitnessBinary() -> Data {
    let prime = bn254FrPrimeBytes()

    // Header section
    var header = [UInt8]()
    header += le32(32)      // fieldSize
    header += prime          // prime
    header += le32(4)        // nWitness

    // Witness values: [1, 17, 3, 9]  (one, y, x, v=x*x)
    var witness = [UInt8]()
    witness += fieldBytes(1)    // wire 0 = one
    witness += fieldBytes(17)   // wire 1 = y = 3^2 + 3 + 5 = 17
    witness += fieldBytes(3)    // wire 2 = x = 3
    witness += fieldBytes(9)    // wire 3 = v = x*x = 9

    var file = [UInt8]()
    file += [0x77, 0x74, 0x6E, 0x73]   // magic "wtns"
    file += le32(2)                      // version
    file += le32(2)                      // 2 sections
    file += section(1, header)
    file += section(2, witness)

    return Data(file)
}

// MARK: - Tests

func runCircomParserTests() {
    suite("Circom R1CS Parser")

    // Test 1: Parse simple R1CS (single multiply gate)
    do {
        let data = buildSimpleR1CSBinary()
        let file = try R1CSParser.parse(data)
        expect(file.version == 1, "R1CS version is 1")
        expectEqual(Int(file.header.nWires), 3, "nWires = 3")
        expectEqual(Int(file.header.nOutputs), 1, "nOutputs = 1")
        expectEqual(Int(file.header.nPubInputs), 0, "nPubInputs = 0")
        expectEqual(Int(file.header.nPrivInputs), 1, "nPrivInputs = 1")
        expectEqual(Int(file.header.nConstraints), 1, "nConstraints = 1")
        expectEqual(Int(file.header.fieldSize), 32, "fieldSize = 32")
        expectEqual(file.constraints.count, 1, "1 constraint parsed")

        // Check wire-to-label mapping
        expect(file.wireToLabel != nil, "wire-to-label section present")
        if let labels = file.wireToLabel {
            expectEqual(labels.count, 3, "3 labels")
            expectEqual(labels[0], 0, "label[0] = 0")
            expectEqual(labels[2], 2, "label[2] = 2")
        }

        // Check constraint structure: A has 1 term (wire 2=x), B has 1 term (wire 2=x), C has 1 term (wire 1=v)
        let c = file.constraints[0]
        expectEqual(c.a.terms.count, 1, "A has 1 term")
        expectEqual(Int(c.a.terms[0].wireId), 2, "A term wire = 2 (x)")
        expectEqual(c.b.terms.count, 1, "B has 1 term")
        expectEqual(Int(c.b.terms[0].wireId), 2, "B term wire = 2 (x)")
        expectEqual(c.c.terms.count, 1, "C has 1 term")
        expectEqual(Int(c.c.terms[0].wireId), 1, "C term wire = 1 (v)")

        // Verify coefficient is 1 in Montgomery form
        let coeffOne = c.a.terms[0].coeff
        expect(frEq(coeffOne, Fr.one), "coefficient = 1 (Montgomery)")
    } catch {
        expect(false, "Simple R1CS parse failed: \(error)")
    }

    // Test 2: Convert to R1CSInstance and verify satisfaction
    do {
        let data = buildSimpleR1CSBinary()
        let file = try R1CSParser.parse(data)
        let r1cs = R1CSParser.toR1CSInstance(file)

        expectEqual(r1cs.numConstraints, 1, "R1CSInstance numConstraints = 1")
        expectEqual(r1cs.numVars, 3, "R1CSInstance numVars = 3")
        expectEqual(r1cs.numPublic, 1, "R1CSInstance numPublic = 1 (nOutputs)")

        // Build z = [one, v=9, x=3] — wire layout: [one(0), v(1)=output, x(2)=private]
        let x = frFromInt(3)
        let v = frMul(x, x)  // v = 9
        let z = [Fr.one, v, x]  // [one, output, private]
        expect(r1cs.isSatisfied(z: z), "R1CS satisfied for x=3, v=9")

        // Check that wrong witness fails
        let zBad = [Fr.one, frFromInt(10), x]  // v=10 != x*x=9
        expect(!r1cs.isSatisfied(z: zBad), "R1CS fails for wrong witness")
    } catch {
        expect(false, "R1CS to R1CSInstance failed: \(error)")
    }

    // Test 3: Quadratic circuit R1CS parse + satisfaction
    do {
        let data = buildQuadraticR1CSBinary()
        let file = try R1CSParser.parse(data)

        expectEqual(Int(file.header.nConstraints), 2, "quadratic: 2 constraints")
        expectEqual(Int(file.header.nWires), 4, "quadratic: 4 wires")
        expect(file.wireToLabel == nil, "quadratic: no label section")

        let r1cs = R1CSParser.toR1CSInstance(file)
        expectEqual(r1cs.numPublic, 1, "quadratic: numPublic = 1")

        // z = [one, y, x, v] where x=3, v=9, y=17
        let x = frFromInt(3)
        let v = frMul(x, x)
        let y = frAdd(frAdd(v, x), frFromInt(5))
        let z = [Fr.one, y, x, v]
        expect(r1cs.isSatisfied(z: z), "quadratic R1CS satisfied for x=3")

        // Also test x=7: v=49, y=49+7+5=61
        let x2 = frFromInt(7)
        let v2 = frMul(x2, x2)
        let y2 = frAdd(frAdd(v2, x2), frFromInt(5))
        let z2 = [Fr.one, y2, x2, v2]
        expect(r1cs.isSatisfied(z: z2), "quadratic R1CS satisfied for x=7")
    } catch {
        expect(false, "Quadratic R1CS parse failed: \(error)")
    }

    // Test 4: Invalid magic bytes
    do {
        var badData = buildSimpleR1CSBinary()
        badData[0] = 0x00  // corrupt magic
        _ = try R1CSParser.parse(badData)
        expect(false, "should have thrown for invalid magic")
    } catch let e as R1CSParserError {
        expect(e.description.contains("invalid magic"), "invalid magic error: \(e)")
    } catch {
        expect(false, "unexpected error type: \(error)")
    }

    // Test 5: Unsupported version
    do {
        var badData = [UInt8]()
        badData += [0x72, 0x31, 0x63, 0x73]  // magic
        badData += le32(99)                    // bad version
        badData += le32(0)                     // 0 sections
        _ = try R1CSParser.parse(Data(badData))
        expect(false, "should have thrown for unsupported version")
    } catch let e as R1CSParserError {
        expect(e.description.contains("unsupported version"), "version error: \(e)")
    } catch {
        expect(false, "unexpected error type: \(error)")
    }

    suite("Circom Witness Parser")

    // Test 6: Parse witness file
    do {
        let data = buildQuadraticWitnessBinary()
        let file = try WitnessParser.parse(data)

        expectEqual(file.version, 2, "WTNS version = 2")
        expectEqual(Int(file.header.nWitness), 4, "nWitness = 4")
        expectEqual(Int(file.header.fieldSize), 32, "fieldSize = 32")
        expectEqual(file.values.count, 4, "4 witness values")

        // Check values
        expect(frEq(file.values[0], Fr.one), "wire 0 = one")
        expect(frEq(file.values[1], frFromInt(17)), "wire 1 = y = 17")
        expect(frEq(file.values[2], frFromInt(3)), "wire 2 = x = 3")
        expect(frEq(file.values[3], frFromInt(9)), "wire 3 = v = 9")
    } catch {
        expect(false, "Witness parse failed: \(error)")
    }

    // Test 7: Witness + R1CS round-trip satisfaction check
    do {
        let r1csData = buildQuadraticR1CSBinary()
        let wtnsData = buildQuadraticWitnessBinary()

        let r1csFile = try R1CSParser.parse(r1csData)
        let wtnsFile = try WitnessParser.parse(wtnsData)
        let r1cs = R1CSParser.toR1CSInstance(r1csFile)

        let z = WitnessParser.witnessVector(wtnsFile)
        expect(r1cs.isSatisfied(z: z), "R1CS satisfied with parsed witness")

        // Extract public inputs and private witness
        let pubInputs = WitnessParser.publicInputs(wtnsFile, r1cs: r1csFile)
        let privWitness = WitnessParser.privateWitness(wtnsFile, r1cs: r1csFile)
        expectEqual(pubInputs.count, 1, "1 public input (y)")
        expect(frEq(pubInputs[0], frFromInt(17)), "public input y = 17")
        expectEqual(privWitness.count, 2, "2 private values (x, v)")
        expect(frEq(privWitness[0], frFromInt(3)), "private x = 3")
        expect(frEq(privWitness[1], frFromInt(9)), "private v = 9")
    } catch {
        expect(false, "R1CS+Witness round-trip failed: \(error)")
    }

    // Test 8: Invalid witness magic
    do {
        var badData = buildQuadraticWitnessBinary()
        badData[0] = 0x00
        _ = try WitnessParser.parse(badData)
        expect(false, "should have thrown for invalid magic")
    } catch let e as WitnessParserError {
        expect(e.description.contains("invalid magic"), "witness invalid magic: \(e)")
    } catch {
        expect(false, "unexpected error type: \(error)")
    }

    // Test 9: Field element edge case - zero
    do {
        // Verify that fieldBytes(0) parses to Fr.zero
        let zeroBytes = fieldBytes(0)
        let data = Data(zeroBytes)
        // Use the parser's field conversion indirectly through a witness file
        let prime = bn254FrPrimeBytes()
        var header = [UInt8]()
        header += le32(32)
        header += prime
        header += le32(1)

        var witness = [UInt8]()
        witness += fieldBytes(0)  // zero

        var file = [UInt8]()
        file += [0x77, 0x74, 0x6E, 0x73]
        file += le32(2)
        file += le32(2)
        file += section(1, header)
        file += section(2, witness)

        let parsed = try WitnessParser.parse(Data(file))
        expect(parsed.values[0].isZero, "field element 0 parses to Fr.zero")
    } catch {
        expect(false, "Zero field element test failed: \(error)")
    }
}
