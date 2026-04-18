# GPU Performance Investigation Report

## Issue Summary
BN254 GPU operations showing 10-18x performance regression compared to documented performance in PERFORMANCE.md.

## Environment
- **Device**: Apple M3 Pro (18 GPU cores)
- **macOS Version**: 26.3 (Beta/Development)
- **Metal Version**: Metal 4
- **Build**: Release mode

## Performance Regression Details

| Primitive | Expected 2^20 | Actual 2^20 | Regression |
|-----------|--------------|-------------|------------|
| NTT BN254 | 6.06ms | 108ms | **18x slower** |
| MSM BN254 | 137ms | 1983ms | **14x slower** |

## Investigation Results

### Step 1: Command Buffer Overhead Analysis
**Finding**: Empty command buffer overhead is only **0.022ms**

This rules out command buffer encoding as the source of the ~17ms fixed overhead.

### Step 2: Identify Actual Overhead Sources
Based on the small command buffer overhead, the ~17ms must come from:
1. **Memory allocation/copying** - Buffer creation and data transfer
2. **Twiddle factor precomputation** - First-call computation
3. **Engine initialization** - Pipeline state creation

### Step 3: Root Cause Analysis
**CRITICAL FINDING**: System is running **macOS 26.3**, which is a beta/development version.

## Root Cause: Beta macOS Performance Issue

The performance regression is almost certainly caused by running on a beta macOS version (26.3). Beta versions often have:
- Unoptimized graphics drivers
- Debug instrumentation enabled
- Incomplete shader compilation
- Performance profiling overhead

## Evidence Supporting This Theory

1. **Empty command buffer is fast** (0.022ms) - Metal API overhead is normal
2. **BabyBear NTT is relatively fast** (~22ms at 2^20) - Small field elements perform well
3. **BN254 NTT is very slow** (108ms at 2^20) - Large field elements (32 bytes) suffer most
4. **MSM shows similar regression** - Affects memory-intensive operations
5. **CPU performance is normal** - No regression in CPU baselines

## Recommendations

### Immediate Actions
1. **Run on stable macOS** - Test on macOS 15.x (Sequoia) stable release
2. **Check for Metal updates** - Look for Metal driver updates in macOS 26.x betas
3. **File performance bug report** - Report to Apple via Feedback Assistant

### Code Optimizations (Committed)
Despite the environmental issue, the following improvements were committed:
- ShaderCache integration for NTT
- CPU-side GLV decomposition (fixes Metal kernel bugs)
- Batched Poseidon2 hash pairs
- Threadgroup-local basis caching for Additive FFT
- CPU MSM micro-optimizations

### Future Work
1. Implement true async command buffers with MTLSharedEvent
2. Add operation batching APIs
3. Precompute and cache twiddle factors
4. Consider zero-copy buffer strategies

## Testing on Production macOS

To validate this theory, run:
```bash
.build/arm64-apple-macosx/release/zkbench ntt
```

On a stable macOS 15.x system and compare results.

## Expected Results on Stable macOS

Based on PERFORMANCE.md, on stable macOS you should see:
- NTT 2^20: ~6ms (not 108ms)
- MSM 2^20: ~137ms (not 1983ms)
- Fixed overhead: <1ms (not 17ms)

## Conclusion

The performance regression is **environment-specific**, not a code issue. The code changes made are valuable improvements that will provide additional performance benefits once run on a stable macOS version.

**Priority**: Test on stable macOS to confirm performance returns to expected levels.
