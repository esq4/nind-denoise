# Code Cleanup Report - nind-denoise

**Date:** 2025-09-19  
**Analysis Period:** Recent commits and current codebase state

## Summary

Conducted comprehensive code review and cleanup of the nind-denoise repository to remove deprecated code, fix structural
issues, and streamline the codebase.

## Actions Taken

### 1. Test File Organization

**Issue:** Test file misplaced in source directory  
**Action:** Moved `src/nind_denoise/rl_pt/test_rl_deconvolution.py` → `tests/test_rl_deconvolution.py`  
**Status:** ✅ Completed  
**Note:** Updated imports to use dynamic import pattern following project guidelines, but module has complex relative
import dependencies that prevent it from running properly in tests directory.

### 2. Deprecated Files Identified

**Files identified as deprecated but not removed (user intervention required):**

- `src/ImageProcessor.py` - Old class-based implementation duplicating pipeline.py functionality
- `src/Snippet.py` - Another deprecated implementation with docopt-style CLI

**Analysis:** Both files contain duplicate functionality now properly implemented in `src/nind_denoise/pipeline.py` and
`src/denoise.py`. No imports or references found in active codebase.

### 3. Code Duplication Found

**Issue:** Identical duplicate files

- `src/nind_denoise/tools/graph_utils.py`
- `src/nind_denoise/libs/graph_utils.py`

**Analysis:** Files are identical (diff shows no differences). No Python imports found. Only referenced in deployment
script `make_clean_repo.sh`.

### 4. Outdated Infrastructure

**File:** `src/nind_denoise/tools/make_clean_repo.sh`  
**Issues:**

- References non-existent paths (`src/common/freelibs`)
- Appears to create the graph_utils.py duplication (lines 25, 27)
- May be outdated deployment script

## Recommendations for Further Cleanup

1. **Fix test imports** - Resolve RL test relative import issues
2. **Review dataset tools** - Many training/dataset tools may not be needed for production use
3. **Review neural network modules** in `src/nind_denoise/networks/` - Many appear unused by current pipeline
4. **Review tool scripts** in `src/nind_denoise/tools/` - Many may be development utilities not needed for main
   functionality
5. **Consolidate utilities** - Multiple utility libraries with potential overlap

## Current Active Components (Confirmed in Use)

- `src/denoise.py` - Main CLI entry point
- `src/nind_denoise/pipeline.py` - Core pipeline logic
- `src/nind_denoise/rl_pt/` - Richardson-Lucy deconvolution (PyTorch implementation)
- `src/config/` - Configuration files
- `tests/test_denoise_cli.py` - Working CLI test
- [incomplete - add more as they are found]

## Git History Analysis

Recent commits show:

- Addition of RL PyTorch implementation (active)
- Previous removal of test files (362764f, e51b5c5)
- Enhanced RL test cases that were later moved
- Rollback of some AI changes (491c5e2)

## Files Moved/Changed This Session

1. ✅ `src/nind_denoise/rl_pt/test_rl_deconvolution.py` → `tests/test_rl_deconvolution.py`
2. ✅ Updated imports in moved test file to use project guidelines pattern

## Next Steps

1. User decision on removing deprecated files
2. Remove duplicate graph_utils.py
3. Update deployment scripts or remove if obsolete
4. Consider broader review of neural network and tool modules for active use