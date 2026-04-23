# Response to Jolt ZK Integration Issue - BN254 Pippenger MSM

## Summary

This issue was already addressed in zkMetal commits:
- **#S3129** (Apr 21): Fixed BN254 Pippenger MSM bug where garbage bits beyond 254-bit scalar field width caused incorrect results
- **#S3131** (Apr 21): Root cause identified and fixed - scalar extraction bug

## Root Cause

The scalar conversion function `bn254_fr_batch_to_limbs` was not properly masking the upper bits of the 256-bit representation, causing garbage bits beyond the 254-bit BN254 scalar field to affect results.

## Files Fixed

- `Sources/NeonFieldOps/bn254_msm.c` - scalar extraction logic corrected
- `bindings/rust/src/arkworks.rs` - Rust FFI scalar conversion updated

## For Jolt Integration

The fix in zkMetal should resolve the verification failures. Ensure you're using the latest version of zkMetal with the following commits:
- `6041e4ee` (latest) - docs: update MSM backlog
- `7f310416` - fix: add include guards to Metal shaders

## If Still Failing

1. **Verify zkMetal version**: Make sure you're pulling from the latest main branch
2. **Check scalar conversion**: The `bn254_fr_batch_to_limbs` function should mask scalars to 254 bits
3. **Verify point format**: Points should be in Montgomery form (x[4], y[4] little-endian u64)

## Contact

If the issue persists after updating, please share:
1. The specific zkMetal commit hash you're using
2. A minimal test case that fails
3. Expected vs actual MSM results

---

*Saved: 2026-04-22*