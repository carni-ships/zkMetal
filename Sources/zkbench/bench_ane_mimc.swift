// bench_ane_mimc.swift — MiMC ANE benchmarks
//
// Benchmarks MiMC hash using x^7 S-box (BN254 Fr).
//
// Note: ANE MiMC returns -1 (not implemented), and the scalar
// fallback is a placeholder that doesn't perform actual arithmetic.
// This benchmark documents the current state and establishes
// a framework for future ANE speedup comparisons.

import ANEOps
import Foundation

public func runANEMiMCBench() {
    print("=== ANE MiMC Benchmark ===")
    print("Note: ANE MiMC returns -1 (not available).")
    print("      Scalar fallback is a placeholder (no actual arithmetic).\n")

    // ============================================================
    // Check ANE availability
    // ============================================================
    let aneAvailable = ane_mimc_available()
    print("--- ANE Availability ---")
    print("  ane_mimc_available(): \(aneAvailable ? "true" : "false")")

    // ============================================================
    // Create ANE state (will fail since ANE not available)
    // ============================================================
    print("\n--- ANE State Creation ---")
    let state = ane_mimc_create(91)  // 91 rounds
    if state == nil {
        print("  ane_mimc_create(91): NULL (expected - ANE not available)")
    } else {
        print("  ane_mimc_create(91): non-NULL (unexpected)")
    }

    // ============================================================
    // MiMC hash attempt
    // ============================================================
    print("\n--- MiMC Hash ---")
    print("  MiMC hash: ANE not available, scalar fallback is placeholder.")
    print("  Real implementation requires BN254 Fr Montgomery multiplication.")
    print("  Framework ready for future ANE implementation.")

    // ============================================================
    // Cleanup (no-op since state is NULL)
    // ============================================================
    ane_mimc_destroy(state)

    // ============================================================
    // Framework note
    // ============================================================
    print("\n--- Framework Status ---")
    print("  MiMC ANE framework is in place but computation is stubbed.")
    print("  Actual BN254 Fr arithmetic (Montgomery mul) needs to be added")
    print("  to ane_mimc.mm mimc_x7_scalar() for real benchmarks.")
    print("  Once implemented, this benchmark will measure:")
    print("    - Single element x^7: 3 Montgomery muls (x^2, x^4, x^7)")
    print("    - Full MiMC hash: 91 rounds of x^7 + key addition")
    print("    - Batch throughput for parallel hashing")

    print("\nMiMC ANE benchmark complete.")
}
