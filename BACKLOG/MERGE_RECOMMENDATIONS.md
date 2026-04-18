# Branch Merge Recommendations

## Analysis Date
2026-04-18

## Current Main Commit
`fc190ab15d6d0641c2a2c5546569e5198cf4114f`

## Ready to Merge ✅

### 1. `fix/orphaned-benchmark-reference`
- **Status**: Already merged (0 commits ahead/behind)
- **Content**: Removes orphaned benchmark reference
- **Action**: None needed (already in main)

### 2. `feature/pedersen-pcs-engine`
- **Commits**: 1 ahead, 0 behind
- **Last updated**: 2026-04-16
- **Conflicts**: None
- **Content**: Pedersen PCS Engine implementation
- **Recommendation**: ✅ **Safe to merge**
- **Command**: `git checkout main && git merge feature/pedersen-pcs-engine`

### 3. `auto-review/zkfinal`
- **Commits**: 1 ahead, 6 behind
- **Last updated**: 2026-04-16
- **Conflicts**: None
- **Content**: Auto-review consolidation with PERFORMANCE.md conflict fix
- **Recommendation**: ✅ **Safe to merge** (will require rebase first due to being behind)
- **Command**:
  ```bash
  git checkout auto-review/zkfinal
  git rebase main
  git checkout main
  git merge auto-review/zkfinal
  ```

### 4. `optimization-ideas`
- **Commits**: 2 ahead, 34 behind
- **Last updated**: 2026-04-13
- **Conflicts**: None
- **Content**: GLV + Batch MSM kernel (marked as WIP with correctness bug)
- **Recommendation**: ⚠️ **Review before merging**
- **Reason**: Marked as WIP with known correctness bug
- **Action**: Needs completion or removal

### 5. `feature/binary-fri-engine` ⭐
- **Commits**: 24 ahead, 0 behind
- **Last updated**: 2026-04-18
- **Conflicts**: None
- **Content**: Binary FRI implementation + extensive performance investigation
- **Recommendation**: ✅ **High priority to merge**
- **Highlights**:
  - Binary FRI prover and FRI fold engine
  - ShaderCache integration for NTT
  - CPU-side GLV decomposition (fixes Metal kernel bugs)
  - GPU Merkle tree building for Circle STARK
  - Comprehensive performance regression analysis
  - Updated PERFORMANCE.md with beta macOS notice
- **Command**: `git checkout main && git merge feature/binary-fri-engine`

## Needs Attention ⚠️

### `backlog/groth16-recursive`
- **Status**: Already merged (0 commits ahead)
- **Action**: None needed

## Remote Branches (Not Local)

### `ane-exploration`
- **Commits**: 1 ahead
- **Last updated**: 2026-04-12
- **Content**: ANE (Apple Neural Engine) exploration
- **Recommendation**: ⚠️ **Experimental** - keep separate for now

### `auto-review/zk202604160210`
- **Commits**: 1 ahead
- **Last updated**: 2026-04-16
- **Content**: Auto-review consolidation
- **Recommendation**: Can be merged or superseded by `auto-review/zkfinal`

### `import-metal-fix`
- **Commits**: 5 ahead
- **Last updated**: 2026-04-15
- **Content**: Metal import fix for MTLBuffer scope errors
- **Recommendation**: ✅ **Should merge** - appears to be a bug fix
- **Action**: Checkout and review for merge

## Merge Order Recommendations

### Option 1: Conservative Merge
```bash
# 1. Merge import-metal-fix (bug fix)
git checkout main
git pull origin import-metal-fix
git merge origin/import-metal-fix

# 2. Merge pedersen-pcs-engine (new feature)
git merge feature/pedersen-pcs-engine

# 3. Merge binary-fri-engine (major feature)
git merge feature/binary-fri-engine

# 4. Merge auto-review/zkfinal (requires rebase)
git checkout auto-review/zkfinal
git rebase main
git checkout main
git merge auto-review/zkfinal
```

### Option 2: Consolidated Merge
```bash
# Create a merge integration branch
git checkout main
git checkout -b merge-integration

# Merge all ready branches
git merge feature/pedersen-pcs-engine
git merge feature/binary-fri-engine
git merge origin/import-metal-fix

# Test the combined result
swift test

# If tests pass, merge back to main
git checkout main
git merge merge-integration
```

## Branches to Delete (Post-Merge)

After successful merges, these branches can be deleted:
- `fix/orphaned-benchmark-reference` (already merged)
- `backlog/groth16-recursive` (already merged)
- Merged branches (after confirmation)

## Branches to Keep

- `optimization-ideas` - Keep until WIP is completed
- `ane-exploration` - Keep for ongoing ANE research
- `before-optimizations` - Keep as performance baseline reference

## Summary

| Branch | Priority | Action | Conflicts |
|--------|----------|--------|-----------|
| `feature/binary-fri-engine` | **High** | Merge | None |
| `feature/pedersen-pcs-engine` | **Medium** | Merge | None |
| `origin/import-metal-fix` | **Medium** | Review & Merge | Unknown |
| `auto-review/zkfinal` | Low | Rebase & Merge | None |
| `optimization-ideas` | Low | Complete WIP | None |

## Next Steps

1. **Immediate**: Merge `import-metal-fix` (bug fix)
2. **High Priority**: Merge `feature/binary-fri-engine` (24 commits, well-tested)
3. **Medium Priority**: Merge `feature/pedersen-pcs-engine`
4. **Low Priority**: Rebase and merge `auto-review/zkfinal`
5. **Review**: Complete or remove `optimization-ideas`
