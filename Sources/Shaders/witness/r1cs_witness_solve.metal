// GPU kernel for R1CS witness solving
//
// Given packed constraint data and a partial witness, solves for unknown
// variables in parallel. Each thread handles one constraint that has been
// pre-analyzed on CPU to have exactly one unknown variable.
//
// Solve modes:
//   0: unknown in C => target = (A*B - knownC) / coeff
//   1: unknown in A => target = (C - knownA*B) / (coeff*B)
//   2: unknown in B => target = (C - A*knownB) / (coeff*A)
//
// The CPU pre-computes: knownA, knownB, knownC (partial LC evaluations),
// targetCoeff, and the solve mode. The GPU performs the final field
// arithmetic (multiply, subtract, batch-precomputed inverse multiply).

#include <metal_stdlib>
using namespace metal;

// Fr type and operations are included inline at compile time from bn254_fr.metal

// Packed solve info for one constraint:
//   [0..7]   = knownA (Fr, 8 limbs)
//   [8..15]  = knownB (Fr, 8 limbs)
//   [16..23] = knownC (Fr, 8 limbs)
//   [24..31] = targetCoeff (Fr, 8 limbs)
//   [32..39] = precomputedInverse (Fr, 8 limbs) -- batch inverse of denominator
//   [40]     = targetVar (uint)
//   [41]     = solveMode (uint: 0=C, 1=A, 2=B)

struct R1CSSolveInfo {
    Fr knownA;
    Fr knownB;
    Fr knownC;
    Fr targetCoeff;
    Fr precomputedInv;  // inv(denominator) from batch inversion on CPU
    uint targetVar;
    uint solveMode;
    uint _pad0;  // padding for alignment
    uint _pad1;
};

/// r1cs_witness_solve_bn254: Solve one constraint per thread using pre-computed inverses.
///
/// For each constraint, the CPU has already:
///   1. Evaluated the known parts of each LC (knownA, knownB, knownC)
///   2. Determined which LC contains the unknown (solveMode)
///   3. Computed the coefficient of the unknown in that LC (targetCoeff)
///   4. Computed the batch inverse of the denominator (precomputedInv)
///
/// The GPU just needs to do the final mul/sub/mul to get the answer.
kernel void r1cs_witness_solve_bn254(
    device Fr* witness                     [[buffer(0)]],  // witness vector (read/write)
    device const R1CSSolveInfo* infos      [[buffer(1)]],  // per-constraint solve info
    constant uint& num_constraints         [[buffer(2)]],  // number of constraints in this wave
    uint gid                               [[thread_position_in_grid]])
{
    if (gid >= num_constraints) return;

    R1CSSolveInfo info = infos[gid];

    Fr result;

    if (info.solveMode == 0) {
        // Unknown in C: target = (knownA * knownB - knownC) * inv(coeff)
        Fr product = fr_mul(info.knownA, info.knownB);
        Fr diff = fr_sub(product, info.knownC);
        result = fr_mul(diff, info.precomputedInv);
    } else if (info.solveMode == 1) {
        // Unknown in A: target = (knownC - knownA * knownB) * inv(knownB * coeff)
        Fr ab = fr_mul(info.knownA, info.knownB);
        Fr diff = fr_sub(info.knownC, ab);
        result = fr_mul(diff, info.precomputedInv);
    } else if (info.solveMode == 2) {
        // Unknown in B: target = (knownC - knownA * knownB) * inv(knownA * coeff)
        Fr ab = fr_mul(info.knownA, info.knownB);
        Fr diff = fr_sub(info.knownC, ab);
        result = fr_mul(diff, info.precomputedInv);
    } else {
        return;  // mode 3 = unsolvable, handled on CPU
    }

    witness[info.targetVar] = result;
}

/// r1cs_sparse_matvec_bn254: Compute sparse matrix-vector product for one row.
///
/// For each row i, computes result[i] = sum_j (val[j] * vec[col[j]])
/// where the entries for row i are at indices rowStart[i]..<rowStart[i+1].
///
/// This is used to evaluate A*z, B*z, C*z in parallel across rows.
kernel void r1cs_sparse_matvec_bn254(
    device Fr* result                      [[buffer(0)]],  // output: one Fr per row
    device const Fr* vec                   [[buffer(1)]],  // input vector (z)
    device const uint* cols                [[buffer(2)]],  // column indices (flat)
    device const Fr* vals                  [[buffer(3)]],  // coefficient values (flat)
    device const uint* row_starts         [[buffer(4)]],  // CSR row pointers
    constant uint& num_rows               [[buffer(5)]],  // number of rows
    uint gid                               [[thread_position_in_grid]])
{
    if (gid >= num_rows) return;

    uint start = row_starts[gid];
    uint end = row_starts[gid + 1];

    Fr acc = {{0, 0, 0, 0, 0, 0, 0, 0}};
    for (uint j = start; j < end; j++) {
        Fr v = fr_mul(vals[j], vec[cols[j]]);
        acc = fr_add(acc, v);
    }

    result[gid] = acc;
}
