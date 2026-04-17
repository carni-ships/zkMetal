# zkMetal Security Documentation

## Overview

This directory contains silicon-aware soundness analysis for zkMetal components, focusing on practical security issues that arise from real silicon implementation characteristics.

## Contents

### SiliconAwareSoundnessAnalysis.swift

Comprehensive security analysis document covering:

#### 1. Soundness Error Analysis
- **FRIAnalysis**: Concrete soundness error calculations for Binary FRI with folding factors 2/4/8
- **PedersenAnalysis**: Discrete logarithm security margins for Pedersen commitments
- **FoldingAnalysis**: Nova/Supernova folding scheme soundness analysis
- **MerkleAnalysis**: Merkle tree authentication path security

#### 2. Leakage Detection
- **FieldOperationLeakage**: Timing side channels in Montgomery multiplication, inversion, addition
- **PowerAnalysis**: GPU/CPU power analysis attack surface assessment
- **CacheAttackAnalysis**: Cache timing attacks on Merkle tree traversal
- **Poseidon2Analysis**: Poseidon hash side channel vulnerabilities

#### 3. Hardening Pass
- Priority-ordered list of operations requiring security hardening
- Constant-time modification guidelines
- Redundant check recommendations for critical operations

#### 4. Formal Verification Interface
- Component soundness proofs for BinaryFRI, Pedersen, NovaFolding, MerkleTree, CCS
- FRI/PCS verification connections
- Security assumptions documentation

## Key Findings

### Critical Issues (Priority 1)
- **Field Inversion (fpInverse)**: Variable-time exponentiation - HIGH RISK
  - Used in point operations and proof verification
  - Avoid in sensitive paths until fixed

### High Priority Issues (Priority 2-3)
- **GPU/CPU Hash Divergence**: Merkle tree verification may fail
  - Use instance method verifyProof() for GPU-built trees
- **Cross-Term Computation Timing**: Could leak witness data
  - Matrix-vector multiplication not constant-time

### Medium Priority Issues (Priority 4-6)
- **Pedersen Point Operations**: Non-constant-time comparison
- **GPU MSM Power Analysis**: Physical access required but still concerning
- **Fiat-Shamir Transcript**: Generally safe, monitor for changes

## Quick Reference: Security Margin Estimates

| Component | Theoretical | Practical | Notes |
|-----------|-------------|-----------|-------|
| Binary FRI (fold=2) | 128 bits | ~77 bits | With 10-bit implementation margin |
| Binary FRI (fold=8) | 128 bits | ~59 bits | Higher arity = lower margin |
| Pedersen (BN254) | 128 bits | ~120 bits | With GPU and size adjustments |
| Merkle (depth=20) | 128 bits | ~100 bits | With GPU access pattern margin |

## Usage

This is analysis documentation, not implementation code. The types defined serve as:

1. **Documentation**: Comprehensive security analysis for each component
2. **Reference**: Soundness formulas and security margin calculations
3. **Guidance**: Hardening recommendations and constant-time guidelines

## Related Documentation

- `/Sources/zkMetal/FRI/` - FRI implementations (analyzed here)
- `/Sources/zkMetal/Commitment/` - Commitment schemes (analyzed here)
- `/Sources/zkMetal/Folding/` - Folding schemes (analyzed here)
- `/BACKLOG/PLAN.md` - Project plan including security items

## Security Trade-offs

The analysis documents several intentional trade-offs:

1. **GPU Acceleration vs. Security**: GPU operations are faster but have measurable power signatures
2. **Performance vs. Constant-Time**: Variable-time operations may be faster but leak information
3. **Batch Operations vs. Individual Checks**: Batching improves performance but may expose correlation

These trade-offs are acceptable for typical deployments but should be reconsidered for high-security applications.

## Contributing

When modifying zkMetal components:

1. Review the security analysis for the component
2. Check if changes affect any documented assumptions
3. Update the analysis if new attack vectors are introduced
4. Test constant-time properties for modified operations

## Date

Analysis based on codebase dated: 2026-04-16
