# zkMetal Comprehensive Plan

**Last Updated**: 2026-04-16

## Executive Summary

Based on codebase analysis, recent commits, and subagent research, this document outlines the implementation plan for zkMetal.

## Priority Issues (From GitHub)

| Issue | Title | Priority | Effort | Status |
|-------|-------|----------|--------|--------|
| #3 | Implement Pedersen PCS Engine | High | Medium | **Complete** |
| #4 | Groth16VerifierCircuitEncoder Investigation | Medium | Large/Wontfix | Complete (Wontfix) |
| #5 | Binary-Native List-Decoding FRI | High | Very Large (10 wks) | In Progress (Phase 1) |

## Theoretical Extensions (From Backlog Ideas)

| # | Title | Impact | Effort | Status |
|---|-------|--------|--------|--------|
| 1 | Hybrid Tower-Prime Meta-Field | Very High | Very Large | Not Started |
| 2 | Binary-Native List-Decoding FRI | High | Very Large (10 wks) | Not Started |
| 3 | High-Arity Binary Folding + Native Recursion | High | Large | Not Started |
| 4 | Tower-Native Primitive Compiler | Very High | Very Large | Not Started |
| 5 | Silicon-Aware Soundness | Medium | Medium | Not Started |

## In-Progress Items

| Item | Status | Notes |
|------|--------|-------|
| GPU Additive FFT (GF2^8) | In Progress | ~11-14ms for 2^22, target ~0.5ms |
| GPU CSR Sparse Matvec | Complete | Integrated into GPUNovaFoldEngine |
| Binary-FRI Core Infrastructure | Phase 1 Complete | Types, fold engine, Johnson bound decoder |

## Implementation Order

### Phase 1: Quick Wins (1-2 weeks)

1. **Pedersen PCS Engine (Issue #3)**
   - Medium effort, high utility
   - Enables Pedersen-based polynomial commitment benchmarking
   - Uses existing Pedersen infrastructure
   - **Status: COMPLETE**

2. **Groth16 Wontfix Decision (Issue #4)**
   - Evaluate if Groth16 support is needed
   - If not, remove placeholder test stub
   - **Status: COMPLETE (Wontfix)**

### Phase 2: Core Infrastructure (2-4 weeks)

3. **Hybrid Meta-Field Type System (Backlog #1 - Phase 1)**
   - MetaFieldPair type
   - FieldSwitchGate protocol
   - Encoding relations

4. **Binary-FRI Core (Backlog #2 - Phase 1)**
   - BinaryFRIConfig ✓
   - BinaryFRITypes ✓
   - BinaryFRIFoldEngine ✓
   - BinaryJohnsonBoundDecoder ✓
   - **Status: Phase 1 COMPLETE**

### Phase 3: Advanced Features (4-8 weeks)

5. **High-Arity Binary Folding (Backlog #3)**
   - Extends Nova/Supernova for 2^k instances
   - O(log log N) recursion depth

6. **Johnson Bound Decoder (Backlog #2 - Phase 2)**
   - List decoding for binary AG codes
   - 30-50% proof size reduction potential

## Current Implementation Details

### Pedersen PCS Engine

**Location**: `Sources/zkMetal/Commitment/`

**Existing Infrastructure**:
- `PedersenEngine.swift` - core commitment
- `VectorPedersenCommitment.swift` - vector commitment
- `PedersenHash.swift` - hash function

**Required Addition**:
- `PedersenPCSEngine.swift` - commit/open/verify interface

**Interface to implement**:
```swift
public protocol PolynomialCommitmentScheme {
    associatedtype Field
    func commit(polynomial: [Field]) throws -> Commitment
    func open(polynomial: [Field], point: Field, witness: Field) throws -> Proof
    func verify(commitment: Commitment, point: Field, proof: Proof) throws -> Bool
}
```

### Binary-FRI Implementation

**Location**: `Sources/zkMetal/FRI/`

**New Files Needed**:
- `BinaryFRIConfig.swift`
- `BinaryFRITypes.swift`
- `BinaryFRIFoldEngine.swift`
- `BinaryFRIProverEngine.swift`
- `BinaryFRIVerifierEngine.swift`
- `BinaryJohnsonBoundDecoder.swift`
- `BinaryFRICoCurvilinear.swift`
- `BinaryProximityGap.swift`

**GPU Shaders Needed**:
- `Sources/Shaders/fri/binary_fri_fold.metal`
- `Sources/Shaders/fri/binary_co_curvilinear.metal`

## Dependencies

```
Phase 1 (Pedersen PCS)
    └── None (standalone)

Phase 2 (Meta-Field Type)
    └── Phase 1 complete
    └── Binary tower fields (exists)
    └── Prime fields (exists)

Phase 2 (Binary-FRI Core)
    └── Binary tower fields (exists)
    └── Additive FFT (exists)
    └── Merkle engine (exists)

Phase 3 (High-Arity Folding)
    └── Phase 2 Binary-FRI complete
    └── Nova/Supernova (exists)
```

## Benchmark Targets

| Primitive | Current | Target | Improvement |
|-----------|---------|--------|-------------|
| Pedersen PCS commit | N/A | TBD | New |
| Binary-FRI fold | N/A | TBD | N/A |
| Proof size (Binary-FRI) | N/A | -30-50% vs standard | 30-50% |

## Risk Assessment

| Item | Risk | Mitigation |
|------|------|------------|
| Pedersen PCS | Low | Uses existing infrastructure |
| Binary-FRI | High | 10-week timeline, complex theory |
| Meta-Field | Medium | Phased approach, early validation |
| High-Arity Folding | Medium | Builds on proven Nova/Supernova |

## Next Actions

1. **Immediate**: Start Pedersen PCS Engine implementation
2. **This week**: Decision on Groth16 (implement or wontfix)
3. **Next week**: Begin Binary-FRI core infrastructure
4. **This month**: Hybrid Meta-Field type system prototype
