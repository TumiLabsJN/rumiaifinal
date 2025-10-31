# Stage 8 Implementation Refactor Plan

## Overview

**Purpose**: Document required changes to Stage8MVP2.md to use simplified data sources (`raw_discovery.json` + `taxonomy.json`) instead of per-video classification files and `aggregate_content_classifications()`.

**Impact**: Significant simplification - eliminates complex aggregation logic and reduces dependency on Stage 2.7 per-video analysis.

---

## New Data Source Structure

### Current (Complex) Approach:
```
/buckets/bucket_18-33s/
└── content_analysis/
    ├── {video_id_1}_content.json  (120 individual files)
    ├── {video_id_2}_content.json
    └── ...
```

### New (Simplified) Approach:
```
/content_taxonomies/
├── {hashtag}_raw_discovery.json  (aggregated frequencies)
└── {hashtag}_taxonomy.json       (definitions/descriptions)
```

---

## Files Requiring Changes

### 1. Stage8MVP_Reports.md
**Scope**: Update all Dynamic Fields table entries that reference `aggregate_content_classifications()`

**Lines to Modify**: 409, 411, 435, 482-484, 507, 544, 550, 556, 803, 805, 850, 852, 854-856, 910-911

**Change Type**: Update data source paths from:
- `aggregate_content_classifications(bucket_path, "top")`
- TO: `{base_path}/content_taxonomies/{hashtag}_raw_discovery.json`

**Example Change**:
```markdown
OLD:
| Content Categories (Top 3) | Stage 7 | **Base Function**: `aggregate_content_classifications(bucket_path, "top")` → **Wrapper**: `get_top_n_from_field(bucket_path, "content_category", n=3, "top")` | Array of strings | [...] |

NEW:
| Content Categories (Top 3) | Stage 2.6 | Load `{base_path}/content_taxonomies/{hashtag}_raw_discovery.json` → `discovered_patterns.content_categories` → Take first 3 items (already sorted by frequency) | Array of strings | [...] |
```

---

### 2. Stage8MVP2.md - Section 3.1 (Report 2: extract_creator_data.py)

#### A. Complete Field List (Lines 240-295)
**No changes needed** - Field names remain the same

#### B. Function Documentation to REMOVE (Lines 337-527)

**DELETE Section 0.5.1**: `aggregate_content_classifications()`
- Lines: ~337-413
- Reason: Replaced by simple JSON loading

**DELETE Section 0.5.1.1**: `get_top_n_from_field()` wrapper
- Lines: ~414-527
- Reason: Not needed - raw_discovery.json already sorted by frequency

**KEEP Section 0.5.1.2**: `get_descriptions_from_taxonomy()`
- Lines: ~528-637
- Reason: Still needed to get descriptions from taxonomy.json
- **MINOR UPDATE**: Change source path to `/content_taxonomies/{hashtag}_taxonomy.json`

#### C. Extraction Implementation (Lines 1350-1432)

**Lines 1384-1397** - PHASE 1: HOOK
```python
# OLD (Complex):
aggregated = aggregate_content_classifications(bucket_path, "top")
top_3_hooks = get_top_n_from_field(aggregated, "hook_strategy", n=3)
hook_descriptions = get_descriptions_from_taxonomy(top_3_hooks, "hook_strategy")

# NEW (Simple):
discovery_path = f"{base_path}/content_taxonomies/{args.hashtag}_raw_discovery.json"
with open(discovery_path, 'r') as f:
    discovery = json.load(f)

top_3_hooks = [h["name"] for h in discovery["discovered_patterns"]["hook_strategies"][:3]]
hook_descriptions = get_descriptions_from_taxonomy(top_3_hooks, "hook_strategy", args.hashtag, base_path)
```

**Lines 1407-1418** - PHASE 2: MIDDLE (Keywords, Tactics)
```python
# OLD (Complex):
top_3_keywords = get_top_n_from_field(aggregated, "keywords", n=3)
top_2_tactics = get_top_n_from_field(aggregated, "content_tactics", n=2)

# NEW (Simple):
top_3_keywords = discovery["discovered_patterns"]["trending_keywords"][:3]
top_2_tactics = discovery["discovered_patterns"]["content_tactics"][:2]
```

**Lines 1425-1432** - PHASE 3: CLOSING (CTA Types)
```python
# OLD (Complex):
top_3_ctas = get_top_n_from_field(aggregated, field="caption_cta_type", n=3)

# NEW (Simple):
top_3_ctas = [c["name"] for c in discovery["discovered_patterns"]["closing_strategies"][:3]]
```

---

### 3. Stage8MVP2.md - Section 3.2 (Report 1: extract_client_data.py)

**Lines 1750-1850** - Similar pattern to Report 2

**Changes Needed**:
- Replace all `aggregate_content_classifications()` calls
- Use `raw_discovery.json` from hashtag-level path (not bucket-level)
- Same simplified extraction pattern

**Key Difference**: Report 1 aggregates ACROSS all buckets, so needs to:
1. Load discovery file from base path (not bucket path)
2. Data is already aggregated at hashtag level

---

### 4. Stage8MVP2.md - Section 3.3 (Report 3: extract_competitor_data.py)

**Lines 3240-3340** - Competitor content strategy extraction

**Changes Needed**:
- Replace `aggregate_content_classifications()` for competitor data
- Path: `/data/clients/{client}/competitors/{handle}/{mode}_{strategy}/content_taxonomies/{handle}_raw_discovery.json`
- Same extraction pattern as Report 2

---

### 5. Stage8MVP2.md - Section 3.4 (Report 4: extract_multi_competitor_data.py)

**Lines 5818-5856** - Per-bucket content intelligence

**Challenge**: Report 4 needs per-bucket data for each competitor

**Options**:
1. **Option A**: Generate per-bucket raw_discovery.json files (one per bucket per competitor)
2. **Option B**: Use hashtag-level raw_discovery.json (less precise but simpler)

**Recommended**: Option A - Generate per-bucket discovery files during Stage 2.6

**Changes Needed**:
```python
# For each competitor + bucket combination
for competitor in competitors:
    for bucket in competitor_buckets:
        discovery_path = f"{base_path}/competitors/{competitor}/{mode}_{strategy}/buckets/bucket_{bucket}/content_taxonomies/{competitor}_{bucket}_raw_discovery.json"

        # Load and extract top N patterns
        with open(discovery_path, 'r') as f:
            bucket_discovery = json.load(f)

        top_2_categories = [c["name"] for c in bucket_discovery["discovered_patterns"]["content_categories"][:2]]
        # etc...
```

---

### 6. Stage8MVP.md - Section 0.5

**Section 0.5.1** (Lines 170-413): **DELETE ENTIRE SECTION**
- Remove `aggregate_content_classifications()` documentation
- Remove examples and implementation details

**Section 0.5.1.1** (Lines 414-527): **DELETE ENTIRE SECTION**
- Remove `get_top_n_from_field()` wrapper documentation

**Section 0.5.1.2** (Lines 528-637): **UPDATE**
- Keep `get_descriptions_from_taxonomy()` but update path references
- Change from reading `/config/taxonomies/{type}.json`
- TO: `/content_taxonomies/{hashtag}_taxonomy.json`

---

## Benefits of This Refactor

### 1. Simplification
- ❌ Remove ~200 lines of complex aggregation logic
- ❌ Remove dependency on 120+ per-video classification files
- ✅ Simple JSON loading replaces complex Counter aggregation
- ✅ ~80% reduction in code complexity

### 2. Performance
- **OLD**: Load 120 files, parse JSON, aggregate Counters, calculate percentages
- **NEW**: Load 1 file, extract arrays
- **Speed improvement**: ~100x faster

### 3. Data Pipeline
- **OLD**: Stage 2.7 individual video analysis → aggregate in Stage 8
- **NEW**: Stage 2.6 discovery already aggregated → use directly in Stage 8
- **Eliminates**: Stage 2.7 per-video classification step entirely

### 4. Maintainability
- Fewer functions to maintain
- Simpler data flow
- Easier debugging (1 file vs 120 files)

---

## Missing Data Considerations

### What raw_discovery.json Does NOT Provide:

1. **Caption Analysis Fields**:
   - `caption_hook_type` (e.g., "question", "statement")
   - `caption_cta_type` (e.g., "link_in_bio", "save_post")
   - `emoji_usage`, `caption_length`
   - `hashtag_count` (mean/min/max)

2. **Performance Group Filtering**:
   - Cannot filter by "top" vs "bottom" performers
   - Discovery file aggregates ALL videos in hashtag

### Solutions:

**Option 1**: Generate TWO discovery files per bucket
- `{hashtag}_top_raw_discovery.json` (top performers only)
- `{hashtag}_bottom_raw_discovery.json` (bottom performers only)

**Option 2**: Add caption analysis to raw_discovery.json
- Extend Stage 2.6 to include caption analysis patterns
- Add `caption_strategies` section with hook_type, cta_type frequencies

**Option 3**: Keep minimal per-video analysis for caption data only
- Only classify caption fields (5 fields vs 12 fields)
- Much lighter than full content analysis

**Recommended**: Option 2 - Extend raw_discovery.json schema

---

## Implementation Priority

### Phase 1: Core Qualitative Fields (LOW HANGING FRUIT)
**Impact**: 80% of report fields
**Effort**: Low
**Files Modified**: Stage8MVP2.md Sections 3.1, 3.2, 3.3

**Changes**:
- Content categories
- Hook strategies
- Engagement drivers
- Pain points
- Keywords
- Content tactics

**Uses**: Existing raw_discovery.json (no schema changes)

---

### Phase 2: Caption Analysis Extension (MEDIUM EFFORT)
**Impact**: Remaining 20% of report fields
**Effort**: Medium
**Files Modified**: Stage 2.6 discovery script + Stage8MVP2.md

**Add to raw_discovery.json**:
```json
"caption_strategies": {
  "hook_types": [
    {"name": "question", "frequency": 12, "percentage": 25.0},
    {"name": "statement", "frequency": 8, "percentage": 16.7}
  ],
  "cta_types": [
    {"name": "link_in_bio", "frequency": 15, "percentage": 31.3},
    {"name": "save_post", "frequency": 10, "percentage": 20.8}
  ],
  "hashtag_stats": {
    "mean": 7.5,
    "median": 7,
    "min": 3,
    "max": 15
  }
}
```

---

### Phase 3: Performance Group Separation (OPTIONAL)
**Impact**: Top vs Bottom comparison features
**Effort**: High
**Files Modified**: Stage 2.6 discovery script

**Generate**: Separate discovery files per performance group
- Path structure: `/buckets/bucket_{name}/content_taxonomies/{hashtag}_top_discovery.json`
- Path structure: `/buckets/bucket_{name}/content_taxonomies/{hashtag}_bottom_discovery.json`

**Use Case**: Report 2's contrastive analysis (top vs bottom patterns)

---

## Token Estimate for Full Implementation

### Documentation Updates:
- Stage8MVP_Reports.md: ~20 field mappings to update = ~500 tokens
- Stage8MVP.md: Delete 2 sections, update 1 section = ~300 tokens
- Stage8MVP2.md: Update 4 report implementations = ~2000 tokens

### Code Changes:
- Report 1 extraction: ~400 tokens
- Report 2 extraction: ~600 tokens
- Report 3 extraction: ~500 tokens
- Report 4 extraction: ~800 tokens

**Total Estimated**: ~5,100 tokens for complete refactor

**Your Assessment**: Correct - planning fits in current context, but implementation would push limits

---

## Recommendation

**DO THIS REFACTOR** - The benefits far outweigh the effort:

1. ✅ Massive simplification (remove ~200 lines of complex code)
2. ✅ Better performance (100x faster)
3. ✅ Simpler data pipeline (eliminate Stage 2.7 step)
4. ✅ Easier to maintain
5. ✅ Existing data files already support this approach

**Next Steps**:
1. Extend raw_discovery.json schema to include caption analysis (Phase 2)
2. Update Stage8MVP2.md extraction implementations (Phase 1)
3. Update Stage8MVP_Reports.md field mappings
4. Delete obsolete functions from Stage8MVP.md Section 0.5.1

---

## Appendix: Line-by-Line Change Map

### Stage8MVP2.md Detailed LOC Changes

**Section 3.1 (Report 2)**:
- Line 337-413: DELETE `aggregate_content_classifications()`
- Line 414-527: DELETE `get_top_n_from_field()`
- Line 528-637: UPDATE `get_descriptions_from_taxonomy()` path
- Line 1384-1397: REPLACE aggregation with JSON load (PHASE 1)
- Line 1407-1418: REPLACE aggregation with JSON load (PHASE 2)
- Line 1425-1432: REPLACE aggregation with JSON load (PHASE 3)

**Section 3.2 (Report 1)**:
- Line 1750-1850: REPLACE all aggregation calls with JSON loads

**Section 3.3 (Report 3)**:
- Line 3240-3340: REPLACE all aggregation calls with JSON loads

**Section 3.4 (Report 4)**:
- Line 5818-5856: REPLACE per-bucket aggregation with bucket-level discovery files

### Stage8MVP_Reports.md Field Mapping Updates

Every occurrence of `aggregate_content_classifications()` needs path update to `raw_discovery.json`

**Affected Lines**: 409, 411, 435, 482, 483, 484, 507, 544, 550, 556, 803, 805, 850, 852, 854, 855, 856, 910, 911

### Stage8MVP.md Section Deletions

- Lines 170-413: DELETE Section 0.5.1
- Lines 414-527: DELETE Section 0.5.1.1
- Lines 528-637: UPDATE Section 0.5.1.2 (keep but modify paths)

---

**End of Implementation Plan**
