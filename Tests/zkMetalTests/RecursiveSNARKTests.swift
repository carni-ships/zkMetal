// RecursiveSNARK Tests — recursive proof composition engine
//
// NOTE: The Groth16 recursive tests were removed because Groth16VerifierCircuitEncoder
// was never implemented (the API design had fundamental issues with EC point operations
// vs field arithmetic). The IPAVerifierCircuitTests.swift provides proper tests for
// the IPA verifier circuit encoder which is the working implementation.
//
// For recursive proofs with different inner systems, use UniversalRecursiveProver<Encoder>
// with the appropriate encoder (IPAPastaVerifierCircuitEncoder, PlonkVerifierCircuitEncoder,
// Plonky2VerifierCircuitEncoder, or Halo2VerifierCircuitEncoder).

import zkMetal
import Foundation

// MARK: - Test Runner
//
// NOTE: These tests are stubs that reference types that were never fully implemented.
// The actual recursive SNARK composition tests are in IPAVerifierCircuitTests.swift
// which tests the IPA verifier circuit encoder.

public func runRecursiveSNARKTests() {
    fputs("\n--- Recursive SNARK Composition ---\n", stderr)
    fputs("  NOTE: Groth16 recursive tests removed (Groth16VerifierCircuitEncoder not implemented)\n", stderr)
    fputs("  See IPAVerifierCircuitTests.swift for working IPA encoder tests\n", stderr)
    fputs("[Recursive SNARK] 0 passed, 0 failed (placeholder - see IPAVerifierCircuitTests)\n", stderr)
}
