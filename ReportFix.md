# ReportFix.md - Duplicate Classification Normalization Fix

**Created**: 2025-12-04
**Updated**: 2025-12-04 (expanded scope after discovery)
**Issue**: Duplicate entries in ENGAGEMENT_DRIVER and CONTENT_TACTIC report fields
**Affected Files**: 5 files (1 source + 4 aggregation with duplicate `aggregate_content_classifications()` function)
**Root Cause**: Inconsistent string formatting (spaces vs underscores) in Stage 2.7 classification outputs

---

## Problem Statement

The competitor report Excel output shows duplicate entries for the same classification value:

```
ENGAGEMENT_DRIVER_1: Explaining Scientific Mechanisms (45%)
ENGAGEMENT_DRIVER_4: Explaining Scientific Mechanisms (16%)

CONTENT_TACTIC_1: Direct To Camera (74%)
CONTENT_TACTIC_4: Direct To Camera (23%)
```

These are the SAME category appearing twice because the underlying data has inconsistent formatting.

---

## Discovery Evidence

### Evidence 1: Raw Classification Files Have Both Formats

Query executed on `/home/jorge/rumiaifinal/data/clients/statesidegrowers/competitors/ayunonutricion/top_top/content_analysis/validated/`:

```python
from collections import Counter
# Aggregated engagement_drivers from all *_content.json files:
{
    "explaining_scientific_mechanisms": 82,  # snake_case
    "explaining scientific mechanisms": 29,  # space-separated
    "showing_product_packaging": 46,
    "showing product packaging": 28,
    "direct_to_camera": 137,
    "direct to camera": 42
}
```

### Evidence 2: Taxonomy File Uses Spaces (Not Snake_Case)

File: `/home/jorge/rumiaifinal/data/clients/statesidegrowers/competitors/ayunonutricion/top_top/content_taxonomies/ayunonutricion_taxonomy.json`

```json
{
  "engagement_drivers": [
    "personal testimonial sharing",
    "showing product packaging",
    "explaining scientific mechanisms"
  ],
  "content_tactics": [
    "direct to camera",
    "product in hand demonstration",
    "step by step explanation"
  ]
}
```

### Evidence 3: Taxonomy Validation Does NOT Enforce Snake_Case for These Fields

File: `/home/jorge/rumiaifinal/ml_pipeline/stage2_content_analysis/taxonomy_validation.py` (lines 931-946)

The validation only checks:
- Items are non-empty strings
- Items have length >= 2
- No duplicates within the array

Snake_case enforcement (lines 918-921) only applies to `content_categories` and `hook_strategies`.

### Evidence 4: Classification Prompt Says "Copy EXACTLY" But LLM Inconsistent

File: `/home/jorge/rumiaifinal/ml_pipeline/stage2_content_analysis/classification.py` (lines 231, 235)

```
IMPORTANT: You MUST copy the category name EXACTLY as written in the taxonomy.
String Matching: Copy category names character-for-character from taxonomy above.
```

Claude Haiku sometimes normalizes to snake_case despite this instruction.

---

## Fix Location

**Function**: `aggregate_content_classifications()` - duplicated in 4 files (not shared module)

| File | Function Line | Aggregation Lines |
|------|---------------|-------------------|
| `extract_competitor_data.py` | 154 | 203-217 |
| `extract_client_data.py` | 50 | 99-113 |
| `extract_creator_data.py` | 51 | 100-114 |
| `extract_multi_competitor_data.py` | 121 | 170-184 |

**Note**: These are 4 separate copies of the same function, not imports from a shared module.

---

## Required Changes

**Apply to ALL 4 files listed above.**

### Change 1: Add Normalization Helper Function

Insert after imports section (exact line varies by file, place after last import):

```python
def normalize_classification_key(value: str) -> str:
    """
    Normalize classification values to snake_case for consistent aggregation.

    Handles LLM output inconsistency where the same category appears as both:
    - "explaining scientific mechanisms" (space-separated)
    - "explaining_scientific_mechanisms" (snake_case)

    Args:
        value: Raw classification string from *_content.json

    Returns:
        Normalized snake_case string
    """
    if not value:
        return value
    return value.strip().lower().replace(' ', '_')
```

### Change 2: Apply Normalization in aggregate_content_classifications()

Current code (lines 203-217):

```python
for pain in data.get('pain_points', []):
    if pain:
        pain_points[pain] += 1

for keyword in data.get('keywords', []):
    if keyword:
        keywords[keyword] += 1

for driver in data.get('engagement_drivers', []):
    if driver:
        engagement_drivers[driver] += 1

for tactic in data.get('content_tactics', []):
    if tactic:
        content_tactics[tactic] += 1
```

Replace with:

```python
for pain in data.get('pain_points', []):
    if pain:
        pain_points[normalize_classification_key(pain)] += 1

for keyword in data.get('keywords', []):
    if keyword:
        keywords[normalize_classification_key(keyword)] += 1

for driver in data.get('engagement_drivers', []):
    if driver:
        engagement_drivers[normalize_classification_key(driver)] += 1

for tactic in data.get('content_tactics', []):
    if tactic:
        content_tactics[normalize_classification_key(tactic)] += 1
```

---

## Fix Strategy Comparison

| Fix | Location | Files | Fixes Future | Fixes Existing |
|-----|----------|-------|--------------|----------------|
| A: Taxonomy files | `*_taxonomy.json` | N | ❌ | ❌ |
| B: Taxonomy validation | `taxonomy_validation.py` | 1 | ✅ | ❌ |
| C: Aggregation layer | `extract_*.py` | 4 | ✅ | ✅ |
| D: Stage 2.7 normalization | `classification.py` | 1 | ✅ | ❌ |
| **D + C (RECOMMENDED)** | Both | **5** | ✅ | ✅ |

### Alternative A: Fix Taxonomy Files to Use Snake_Case
- **Rejected**: Would require re-running Stage 2.7 classification (~5 min per competitor)
- **Rejected**: Would not fix existing classified data
- **Rejected**: LLM may still produce inconsistent output

### Alternative B: Fix taxonomy_validation.py to Enforce Snake_Case
- **Rejected**: Only prevents future issues, does not fix existing data
- **Rejected**: Still requires re-classification

### Alternative C: Normalize at Aggregation Only
- **Advantage**: Fixes existing data without re-running pipeline
- **Advantage**: Handles both current and future LLM inconsistencies
- **Disadvantage**: 4 duplicate files to modify
- **Disadvantage**: Doesn't fix the source - future consumers would also need normalization

### Alternative D: Fix Stage 2.7 `normalize_classification_schema()`
- **Advantage**: Fix at source (1 file, 1 function)
- **Advantage**: All future `*_content.json` files will be consistent
- **Disadvantage**: Does NOT fix existing `*_content.json` files

### RECOMMENDED: D + C (Both Layers)
- **Fix D**: Prevents future inconsistencies at the source
- **Fix C**: Handles existing inconsistent data at consumption
- **Result**: Complete fix for past and future data

---

## Potential Breakage

### 1. Report Field Display Names

Current code (lines 967-970) converts snake_case to Title Case for display:

```python
tab_data.append([f'ENGAGEMENT_DRIVER_{i}', driver.replace('_', ' ').title()])
```

**Impact**: None. Normalization to snake_case will be converted back to Title Case for display.

**Example**:
- Input: `"explaining scientific mechanisms"` or `"explaining_scientific_mechanisms"`
- After normalization: `"explaining_scientific_mechanisms"`
- After `.replace('_', ' ').title()`: `"Explaining Scientific Mechanisms"`

### 2. Taxonomy Description Lookup

**Impact**: None. Verified that `engagement_drivers` and `content_tactics` do NOT use taxonomy description lookups.

The code at lines 966-970 directly converts to Title Case for display:
```python
# Engagement Drivers (no descriptions in taxonomy)
for i, (driver, count) in enumerate(top_4_drivers, 1):
    tab_data.append([f'ENGAGEMENT_DRIVER_{i}', driver.replace('_', ' ').title()])
```

The `load_taxonomy_descriptions()` function (lines 627-703) builds mappings but they are NOT used for `engagement_drivers` or `content_tactics` - only for `content_categories` and `hook_strategies` which have `{name, definition}` format.

### 3. Percentage Calculation

Current code (line 968):

```python
pct = round((count / total_classified) * 100) if total_classified > 0 else 0
```

**Impact**: Positive. Percentages will be more accurate because duplicates are merged.

**Before fix**: `"explaining scientific mechanisms"` (29) + `"explaining_scientific_mechanisms"` (82) = 111 total, but reported as two separate 26% and 74% entries.

**After fix**: `"explaining_scientific_mechanisms"` (111) = single entry at 100%.

---

## Upstream/Downstream Impact

### Upstream (Fix D modifies source)

**Fix D modifies**:
- `ml_pipeline/stage2_content_analysis/classification.py` - Stage 2.7 classification normalization

**NOT modified**:
- `ml_pipeline/stage2_content_analysis/taxonomy_validation.py` - Taxonomy validation
- `*_content.json` files - Existing classification outputs remain unchanged

**Note**: Fix D affects future `*_content.json` files only. Existing files retain inconsistent formatting, which is handled by Fix C.

### Downstream (Positive Impact)

**Report consumers will see**:
- Deduplicated ENGAGEMENT_DRIVER entries
- Deduplicated CONTENT_TACTIC entries
- More accurate percentage distributions
- No duplicate categories in top-4 lists

**Multi-competitor reports** (`extract_multi_competitor_data.py`):
- Uses similar aggregation logic
- May need same fix applied (check lines 482-555 `aggregate_per_bucket_content()`)

---

## Verification Steps

After implementing the fix:

1. Re-generate report:
```bash
python extract_competitor_data.py --client statesidegrowers --competitor ayunonutricion --mode top --strategy top
```

2. Check Excel output for duplicates:
```python
import pandas as pd
df = pd.read_excel('data/clients/statesidegrowers/competitors/ayunonutricion/top_top/reports/competitor/ayunonutricion_analysis_data.xlsx')

# Should return 0 (no duplicates)
engagement_drivers = df[df['Field Name'].str.contains('ENGAGEMENT_DRIVER_\d+$', regex=True, na=False)]['Value'].tolist()
print(f"Duplicates: {len(engagement_drivers) - len(set(engagement_drivers))}")
```

3. Verify percentages sum correctly:
```python
pcts = df[df['Field Name'].str.contains('ENGAGEMENT_DRIVER_\d+_PCT', regex=True, na=False)]['Value'].tolist()
print(f"Sum: {sum(int(p) for p in pcts)}%")  # Should be close to 100%
```

---

## Files to Modify

### Fix D: Stage 2.7 Source Fix (1 file)

| File | Function | Line |
|------|----------|------|
| `ml_pipeline/stage2_content_analysis/classification.py` | `normalize_classification_schema()` | 1184 |

**Change**: Add normalization for list fields in `normalize_classification_schema()`:

```python
def normalize_classification_schema(...):
    # ... existing code ...

    if flow_type == "full":
        normalized = llm_output.copy()
        normalized['caption_analysis']['hashtag_count'] = hashtag_count

        # ADD: Normalize list field values to snake_case
        for field in ['pain_points', 'keywords', 'engagement_drivers', 'content_tactics']:
            if field in normalized and isinstance(normalized[field], list):
                normalized[field] = [
                    v.strip().lower().replace(' ', '_') if v else v
                    for v in normalized[field]
                ]
```

### Fix C: Aggregation Layer Fix (4 files)

| File | Function At Line | Add Helper After | Modify Aggregation Lines |
|------|------------------|------------------|--------------------------|
| `extract_competitor_data.py` | 154 | line 17 (`import qrcode`) | 203-217 |
| `extract_client_data.py` | 50 | line 18 (`import qrcode`) | 109-123 |
| `extract_creator_data.py` | 51 | line 20 (after `from pathlib import Path`) | 105-119 |
| `extract_multi_competitor_data.py` | 121 | line 19 (`import qrcode`) | 171-185 |

**Note**: Line numbers are approximate. Search for `for driver in data.get('engagement_drivers'` to locate exact position.

**Total**: 5 files, 9 changes (1 source fix + 4 function additions + 4 aggregation modifications)

---

## Implementation Checklist

### Fix D: Stage 2.7 Source Fix
- [ ] `ml_pipeline/stage2_content_analysis/classification.py` - Add list field normalization in `normalize_classification_schema()`

### Fix C: Aggregation Layer (repeat for each of 4 files)
- [ ] Add `normalize_classification_key()` helper function after imports
- [ ] Update `pain_points` aggregation to use normalization
- [ ] Update `keywords` aggregation to use normalization
- [ ] Update `engagement_drivers` aggregation to use normalization
- [ ] Update `content_tactics` aggregation to use normalization

### Files
- [ ] `classification.py` - Apply Fix D
- [ ] `extract_competitor_data.py` - Apply Fix C
- [ ] `extract_client_data.py` - Apply Fix C
- [ ] `extract_creator_data.py` - Apply Fix C
- [ ] `extract_multi_competitor_data.py` - Apply Fix C

### Verification
- [ ] Run verification steps on `extract_competitor_data.py` output
- [ ] Verify no import errors in all 5 files
- [ ] Test Stage 2.7 classification on a new video to verify Fix D works
