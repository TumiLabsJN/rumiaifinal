# UpgradeBucketz: Support for 1-3 Winning Buckets

## For LLM Agents: Context and Audit Instructions

This document describes a proposed enhancement to support small TikTok accounts with fewer than 3 winning duration buckets.

**Status:** AUDITED - Ready for Implementation
**Audit Date:** 2025-11-27
**Auditor:** Claude Code agent

---

## IMPORTANT: Two Different "3"s in the Codebase

Before implementing, understand this critical distinction:

| Concept | Meaning | Variable? | Action |
|---------|---------|-----------|--------|
| **Duration Buckets** | How many winning duration ranges (e.g., 18-33s, 9-13s) | YES (1-3) | **CHANGE** - this enhancement |
| **K-Means Clusters** | ML clustering algorithm output per window | NO (fixed at 3) | **DO NOT CHANGE** |

**The K-Means "3 clusters" assertions in `rumiai_ml_batch.py:329-330, 432` and `model_training.py:873` are CORRECT and should NOT be modified.** They refer to the ML algorithm's hyperparameters, not bucket count.

---

## Problem Statement

**Trigger:** Running competitor analysis on @getrosabella (22 videos, 365 days) failed at Stage 2.6.

**Error:**
```
File: /home/jorge/rumiaifinal/ml_pipeline/stage2_content_analysis/validation.py, line 51
ValueError: Expected 3 selected buckets, found 2. Stage 2.5 may have failed.
```

**Root Cause:**
- Stage 1 (`/home/jorge/rumiaifinal/ml_pipeline/stage1_discovery/winner_analyzer.py`) correctly handles <3 buckets - it warns and proceeds with whatever buckets qualify
- Stage 2.6 validation has a hard requirement for exactly 3 buckets
- The sampling algorithm divides by 3 regardless of actual bucket count

**Business Impact:** Small competitor accounts cannot be analyzed.

---

## Source Documentation

The 3-bucket requirement was defined in:
- **File:** `/home/jorge/rumiaifinal/documentation_migration/rumiaibatch/STAGE_2.6_2.7_IMPL.md`
- **Line 88:** "3 selected buckets present" listed as validation check
- **Lines 203-206:** Edge case scenarios assume exactly 3 buckets

**Key insight from Stage 1 code:**
```python
# /home/jorge/rumiaifinal/ml_pipeline/stage1_discovery/winner_analyzer.py, lines 265-270
# Handle edge case - < 3 qualified buckets
if len(top_buckets) < TOP_BUCKETS_TO_PROCESS:
    logger.warning(
        f"Only {len(top_buckets)} bucket(s) qualified (≥{MIN_WINNER_PERCENTAGE}% winners). "
        f"Processing {len(top_buckets)} bucket(s) instead of {TOP_BUCKETS_TO_PROCESS}."
    )
```
Stage 1 already anticipates and handles <3 buckets gracefully.

---

## Files Requiring Modification

### 1. Stage 2.6 Validation (CRITICAL - Unblocks Pipeline)

**File:** `/home/jorge/rumiaifinal/ml_pipeline/stage2_content_analysis/validation.py`

**Current code (lines 48-54):**
```python
    # Validation 3: Check we have 3 buckets
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 803-808
    if len(manifest['selected_buckets']) != 3:
        raise ValueError(
            f"Expected 3 selected buckets, found {len(manifest['selected_buckets'])}. "
            "Stage 2.5 may have failed."
        )
```

**Proposed change:**
```python
    # Validation 3: Check we have at least 1 bucket (allow 1-3 for small datasets)
    # Modified: Original required exactly 3, relaxed for small competitor accounts
    if len(manifest['selected_buckets']) < 1:
        raise ValueError(
            f"Expected at least 1 selected bucket, found {len(manifest['selected_buckets'])}. "
            "Stage 2.5 may have failed."
        )
    if len(manifest['selected_buckets']) < 3:
        logger.warning(
            f"Only {len(manifest['selected_buckets'])} bucket(s) selected (typically 3). "
            f"Small dataset - proceeding with limited buckets."
        )
```

---

### 2. Stage 2.6 Sampling Calculation (CRITICAL - Fixes Math)

**File:** `/home/jorge/rumiaifinal/ml_pipeline/stage2_content_analysis/discovery.py`

**Current code (lines 92-93):**
```python
    # L18 FIX: 20 samples per bucket = balanced duration representation across 3 buckets
    target_per_bucket = sample_size // 3  # Default: 60 // 3 = 20 per bucket
```

**Proposed change:**
```python
    # Distribute samples evenly across actual bucket count (1-3 buckets)
    num_buckets = len(top_3_buckets)
    target_per_bucket = sample_size // num_buckets  # e.g., 60 // 2 = 30 per bucket
```

**Impact:** With 2 buckets, this changes from 20 samples/bucket (40 total) to 30 samples/bucket (60 total).

---

### 3. Stage 8 Report Generation (LOW PRIORITY - Cosmetic)

**File:** `/home/jorge/rumiaifinal/extract_competitor_data.py`

**Current code (line 1081):**
```python
        if i < 3:  # Add empty row between buckets (not after last one)
            tab_data.append(['', ''])
```

**Proposed change:**
```python
        if i < len(winner_data['top_3_buckets']):  # Add empty row between buckets (not after last one)
            tab_data.append(['', ''])
```

---

### 4. Multi-Competitor QR Code Calculation (LOW PRIORITY - Cosmetic)

**File:** `/home/jorge/rumiaifinal/extract_multi_competitor_data.py`

**Current code (lines 1061-1062):**
```python
    total_qr_codes = len(competitor_list) * 3 * 2  # N competitors × 3 buckets × 2 QR codes
    print(f"Generating {total_qr_codes} QR codes (2 per bucket × 3 buckets × {len(competitor_list)} competitors)...")
```

**Proposed change:**
```python
    # Calculate actual QR codes based on each competitor's bucket count
    total_qr_codes = 0
    for competitor in competitor_list:
        # Count will be calculated dynamically during iteration
        pass  # Actual count logged per competitor
    print(f"Generating QR codes (2 per bucket × variable buckets × {len(competitor_list)} competitors)...")
```

**Note:** The actual iteration loop at line 1081 already handles variable buckets correctly. This is just fixing the progress message.

---

### 5. Comment Fix (LOW PRIORITY - Cosmetic)

**File:** `/home/jorge/rumiaifinal/extract_client_data.py`

**Current code (line 627):**
```python
    # Formula names (9 total: 3 per bucket × 3 buckets)
```

**Proposed change:**
```python
    # Formula names (3 per bucket × N buckets where N=1-3)
```

---

## Files That Already Handle Variable Buckets (VERIFIED SAFE)

These files iterate dynamically over bucket lists and need NO changes:

| File | Key Code Pattern | Status |
|------|------------------|--------|
| `/home/jorge/rumiaifinal/rumiai_ml_batch.py:717-795` | `for bucket_name in winning_buckets:` | ✅ Safe |
| `/home/jorge/rumiaifinal/extract_competitor_data.py:105` | `for bucket in top_3_buckets:` | ✅ Safe |
| `/home/jorge/rumiaifinal/extract_client_data.py:190, 288` | `for bucket in winning_buckets:` | ✅ Safe |
| `/home/jorge/rumiaifinal/extract_creator_data.py:263` | `for bucket in winning_buckets:` | ✅ Safe |
| `/home/jorge/rumiaifinal/extract_multi_competitor_data.py:242, 400, 518, 580, 694, 763, 1078` | `for bucket in winning_buckets:` | ✅ Safe |
| `/home/jorge/rumiaifinal/run_stage2_only.py:110` | `winning_buckets = winner_analysis['top_3_buckets']` | ✅ Safe |
| `/home/jorge/rumiaifinal/ml_pipeline/stage2_5_organize/file_organizer.py:76` | `if len(winner_analysis['top_3_buckets']) == 0:` | ✅ Safe |

---

## Files That Should NOT Be Changed (K-Means Cluster Validation)

These assertions check that K-Means produces 3 clusters per window - this is ML hyperparameter validation, NOT bucket count:

| File | Line | Code | Why NOT to change |
|------|------|------|-------------------|
| `rumiai_ml_batch.py` | 329-330 | `assert len(window_km_data["clusters"]) == 3` | K-Means clustering config |
| `rumiai_ml_batch.py` | 432 | `assert len(data['clusters']) == 3` | K-Means clustering config |
| `model_training.py` | 873 | `if config['kmeans']['n_clusters'] != 3:` | K-Means hyperparameter |

---

## Test File Updates Required

**File:** `/home/jorge/rumiaifinal/test_step3_adaptive_sampling.py`

**Current code (lines 94-96):**
```python
        # Test 2a: Target per bucket calculation (sample_size // 3)
        has_target_calc = "target_per_bucket = sample_size // 3" in content
        self.assert_true(has_target_calc, "Target per bucket calculated (sample_size // 3)")
```

**Proposed change:**
```python
        # Test 2a: Target per bucket calculation (sample_size // num_buckets)
        has_target_calc = "target_per_bucket = sample_size // num_buckets" in content
        self.assert_true(has_target_calc, "Target per bucket calculated dynamically")
```

---

## Documentation Updates Required

**File:** `/home/jorge/rumiaifinal/documentation_migration/rumiaibatch/STAGE_2.6_2.7_IMPL.md`

**Changes:**
- Line 88: Change "3 selected buckets present" to "at least 1 selected bucket present"
- Lines 203-206: Add scenarios for 1-2 bucket edge cases

---

## Implementation Priority

| Priority | File | Line(s) | Type |
|----------|------|---------|------|
| **CRITICAL** | `validation.py` | 48-54 | Blocker fix |
| **CRITICAL** | `discovery.py` | 92-93 | Math fix |
| LOW | `extract_competitor_data.py` | 1081 | Cosmetic |
| LOW | `extract_multi_competitor_data.py` | 1061-1062 | Cosmetic |
| LOW | `extract_client_data.py` | 627 | Comment |
| LOW | `test_step3_adaptive_sampling.py` | 94-96 | Test update |

---

## Validation Plan

After implementation, test with:

1. **@getrosabella** (2 buckets) - Should now complete
2. **@golinutrition** (3 buckets) - Regression test, should still work
3. **Unit test** with 1 bucket - Edge case

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Reports with <3 buckets have fewer data points | Low | Warning already logged |
| Test suite breaks | Low | Update test file |
| K-Means confusion | Medium | Document clearly (done above) |

---

**Document Created:** 2025-11-27
**Trigger:** @getrosabella competitor analysis failed with 2 buckets (22 videos in 365 days)
**Status:** AUDITED - Ready for Implementation
**Author:** Claude Code agent

---

## Audit Checklist (COMPLETED)

- [x] Read `/home/jorge/rumiaifinal/ml_pipeline/stage2_content_analysis/validation.py` lines 40-70
- [x] Read `/home/jorge/rumiaifinal/ml_pipeline/stage2_content_analysis/discovery.py` lines 85-210
- [x] Read `/home/jorge/rumiaifinal/extract_competitor_data.py` lines 1060-1085
- [x] Grep for `// 3` and `!= 3` in all Python files to find any missed hardcoding
- [x] Confirm Stages 3-7 in `rumiai_ml_batch.py` iterate dynamically (no bucket count assumptions)
- [x] Review each file in "Files Using top_3_buckets" table
- [x] Distinguish K-Means cluster count (3, fixed) from duration bucket count (1-3, variable)
- [x] Identify additional cosmetic fixes in extract_multi_competitor_data.py

---

## Revert Instructions

If this enhancement causes issues, here's how to revert each change:

### Critical Changes (must revert together)

**1. `validation.py` line 48-54** - Restore exact 3-bucket requirement:
```python
# REVERT TO:
    # Validation 3: Check we have 3 buckets
    # Source: ContentAnalysisCHILD.md Section 6.1 lines 803-808
    if len(manifest['selected_buckets']) != 3:
        raise ValueError(
            f"Expected 3 selected buckets, found {len(manifest['selected_buckets'])}. "
            "Stage 2.5 may have failed."
        )
```

**2. `discovery.py` lines 92-93** - Restore hardcoded division by 3:
```python
# REVERT TO:
    # L18 FIX: 20 samples per bucket = balanced duration representation across 3 buckets
    target_per_bucket = sample_size // 3  # Default: 60 // 3 = 20 per bucket
```

### Low Priority Changes (optional revert)

**3. `extract_competitor_data.py` line 1081:**
```python
# REVERT TO:
        if i < 3:  # Add empty row between buckets (not after last one)
```

**4. `extract_multi_competitor_data.py` lines 1061-1062:**
```python
# REVERT TO:
    total_qr_codes = len(competitor_list) * 3 * 2  # N competitors × 3 buckets × 2 QR codes
    print(f"Generating {total_qr_codes} QR codes (2 per bucket × 3 buckets × {len(competitor_list)} competitors)...")
```

**5. `extract_client_data.py` line 627:**
```python
# REVERT TO:
    # Formula names (9 total: 3 per bucket × 3 buckets)
```

**6. `test_step3_adaptive_sampling.py` lines 94-96:**
```python
# REVERT TO:
        # Test 2a: Target per bucket calculation (sample_size // 3)
        has_target_calc = "target_per_bucket = sample_size // 3" in content
        self.assert_true(has_target_calc, "Target per bucket calculated (sample_size // 3)")
```

### Git Revert Option

If all changes are in a single commit:
```bash
git revert <commit-hash>
```

---

## Risk Analysis: Will This Break the Flow?

### Risk Level: **LOW** (for accounts with 3 buckets)

| Scenario | Risk | Reason |
|----------|------|--------|
| Account with 3 buckets (normal) | **NONE** | Code paths unchanged - `len() < 1` is false, warning skipped |
| Account with 2 buckets (like @getrosabella) | **LOW** | New code path, but follows same pattern as Stage 1 |
| Account with 1 bucket (edge case) | **MEDIUM** | Untested, but mathematically sound |
| Account with 0 buckets | **NONE** | Still fails with clear error (as intended) |

### Why Low Risk for 3-Bucket Accounts:

1. **Validation change**: `!= 3` → `< 1` means 3-bucket accounts pass both checks
2. **Sampling change**: `// 3` → `// len(top_3_buckets)` where len=3 produces identical result (60//3=20)
3. **Cosmetic changes**: Only affect display/comments, not logic

### Potential Issues:

| Issue | Likelihood | Impact | Mitigation |
|-------|------------|--------|------------|
| 2-bucket taxonomy discovery has different sample distribution | Medium | Low | More samples per bucket (30 vs 20) - actually better coverage |
| Report layout slightly different with 2 buckets | Low | Low | Cosmetic only |
| 1-bucket edge case has no middle window data | Low | Medium | Test with 1-bucket account before production |

### Recommendation:

**Safe to implement.** The changes are:
- Mathematically equivalent for 3-bucket accounts
- Additive (new warning, not changed behavior) for normal flow
- Only activate new code paths for <3 bucket accounts (which currently fail anyway)
