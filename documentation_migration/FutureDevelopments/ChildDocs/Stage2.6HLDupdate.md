# Stage 2.6/2.7 HLD Update Tracking Document

> **Purpose**: Track all updates needed for ContentAnalysisCHILD.md to align with refined prompts
> **Created**: 2025-01-28
> **Source Documents**:
> - 2.6HashtagCritique.md (Discovery prompt - already matches)
> - 2.7ClassificationCritique.md (Classification prompt - needs updates)
> - ContentAnalysisCHILDTI.md (TI - already updated)

---

## Overview

ContentAnalysisCHILD.md (HLD) needs updates to reflect:
1. Refined Stage 2.7 Classification prompt (3-zone structure, simplified schema)
2. Updated field names: `pain_points` (was `audience_pain_points`), `keywords` (was `trending_keywords`)
3. Reduced field count: 12 fields (was 23 fields)
4. New metadata: `taxonomy_version` field
5. Simplified caption analysis: 8 subfields (was 13 subfields)

**IMPORTANT DISTINCTION**:
- **Taxonomy** (Stage 2.6 output): Uses `audience_pain_points` and `trending_keywords` ✓ (no change)
- **Classification** (Stage 2.7 output): Uses `pain_points` and `keywords` ✓ (needs update)

---

## Section 1: Success Criteria (Lines 43-48)

### Current:
```
- [ ] Stage 2.7 classifies 120 videos in < 15 minutes (Haiku API, 40 per bucket × 3 buckets)
- [ ] Taxonomy schema validates: all 6 fields present (content_categories, hook_strategies, audience_pain_points, trending_keywords, engagement_drivers, content_tactics)
- [ ] Classification output includes complete schema: 10 core fields + 12 caption_analysis subfields
```

### Update To:
```
- [ ] Stage 2.7 classifies 120 videos in < 15 minutes (Haiku API, 40 per bucket × 3 buckets)
- [ ] Taxonomy schema validates: all 6 fields present (content_categories, hook_strategies, audience_pain_points, trending_keywords, engagement_drivers, content_tactics)
- [ ] Classification output includes complete schema: 12 fields total (6 core + 1 caption_analysis object with 8 subfields + 5 metadata fields)
- [ ] Classification uses refined prompt with 3-zone structure and grounding rules
```

**Rationale**: Update field count description, add prompt quality criterion

---

## Section 2: Stage 2.3.4 Classification Prompt (Lines 392-453)

### Status: 🔴 CRITICAL - Complete Rewrite Needed

### Current:
- Old simplified prompt (~60 lines)
- No system message separation
- 13 caption_analysis subfields
- Includes hashtag broad/niche/branded categorization
- Missing grounding rules
- Missing evidence priority hierarchy

### Update To:
- Refined prompt from 2.7ClassificationCritique.md (~200 lines)
- System message + User message structure
- 8 caption_analysis subfields
- Removed hashtag categorization
- Added GROUNDING RULE
- Added evidence priority (transcript > caption > hashtags)
- 3-zone structure (Zone 1: Taxonomy, Zone 2: Caption Analysis, Zone 3: Output)

**Source**: 2.7ClassificationCritique.md lines 420-892 (Final Refined Prompt section)

**Note**: This is the largest change in the document

---

## Section 3: Schema Definition Table (Section 5.1.4 - Lines 675-680)

### Status: ✅ NO CHANGE NEEDED

**Rationale**: This table describes **Taxonomy schema** (Stage 2.6 output), which correctly uses:
- `audience_pain_points`
- `trending_keywords`

Leave as-is.

---

## Section 4: Schema Definition Table (Section 5.2.2 - Lines 712-760)

### Status: 🟡 MODERATE - Field Names + Descriptions Need Update

### Current (line 724-727):
```
| `audience_pain_points` | array[string] | From taxonomy | No | Detected pain points (can be empty array) |
| `trending_keywords` | array[string] | From taxonomy | No | Detected keywords (can be empty array) |
```

### Update To:
```
| `pain_points` | array[string] | From taxonomy | No | Detected pain points (can be empty array) |
| `keywords` | array[string] | From taxonomy | No | Detected keywords (can be empty array) |
```

### Additional Updates Needed:
- Line 712: Update section header note
- Add `taxonomy_version` field row
- Update caption_analysis description: "8 subfields" (not 12)
- Add note about field name changes

---

## Section 5: Example JSON Outputs (Multiple Locations)

### Status: 🟡 MODERATE - 8 locations need field name updates

**Locations**:

1. **Line 750-755** (Section 5.2.2 example):
```
OLD:
  "audience_pain_points": ["menstrual_discomfort", "feminine_wellness"],
  "trending_keywords": ["yoni", "steaming", "holistic", "tcm"],

NEW:
  "pain_points": ["menstrual_discomfort", "feminine_wellness"],
  "keywords": ["yoni", "steaming", "holistic", "tcm"],
```

2. **Line 1181-1184** (Section 8.3 example):
```
OLD:
  "audience_pain_points": ["menstrual_discomfort"],
  "trending_keywords": ["holistic"],

NEW:
  "pain_points": ["menstrual_discomfort"],
  "keywords": ["holistic"],
```

3. **Line 1534-1536** (Appendix C.1 example 1):
```
OLD:
  "audience_pain_points": ["menstrual_discomfort"],
  "trending_keywords": ["holistic", "wellness"],

NEW:
  "pain_points": ["menstrual_discomfort"],
  "keywords": ["holistic", "wellness"],
```

4. **Line 1569-1571** (Appendix C.1 example 2):
```
OLD:
  "audience_pain_points": [],
  "trending_keywords": ["protein", "meal_prep"],

NEW:
  "pain_points": [],
  "keywords": ["protein", "meal_prep"],
```

**Additional locations**: Lines 203-207, 418-422, 1491-1500 (taxonomy examples - NO CHANGE, these are correct)

---

## Section 6: Validation Logic (Section 6.3 - Lines 959-1004)

### Status: 🟡 MODERATE - Field names in validation examples

### Current (lines 968-970):
```python
    core_fields = [
        'video_id', 'content_category', 'hook_strategy', 'audience_pain_points',
        'trending_keywords', 'engagement_drivers', 'content_tactics',
```

### Update To:
```python
    core_fields = [
        'video_id', 'taxonomy_version', 'content_category', 'hook_strategy',
        'pain_points', 'keywords', 'engagement_drivers', 'content_tactics',
```

### Current (lines 1002-1003):
```python
    array_fields = ['audience_pain_points', 'trending_keywords', 'engagement_drivers', 'content_tactics']
```

### Update To:
```python
    array_fields = ['pain_points', 'keywords', 'engagement_drivers', 'content_tactics']
```

---

## Section 7: Configuration Constants (Lines 598-603)

### Status: ✅ NO CHANGE NEEDED

**Rationale**: REQUIRED_TAXONOMY_FIELDS describes **taxonomy schema** (Stage 2.6), not classification output.
Correctly uses `audience_pain_points` and `trending_keywords`.

---

## Section 8: Discovery Output Validation (Lines 933-936)

### Status: ✅ NO CHANGE NEEDED

**Rationale**: Validates **taxonomy schema** (Stage 2.6 output).
Correctly uses `audience_pain_points` and `trending_keywords`.

---

## Section 9: Taxonomy Input Validation (Lines 855-859)

### Status: ✅ NO CHANGE NEEDED

**Rationale**: Validates **taxonomy schema** before Stage 2.7.
Correctly uses `audience_pain_points` and `trending_keywords`.

---

## Section 10: References to "10 core fields + 12 caption subfields"

### Status: 🟢 LOW - Text updates for field counts

**Locations**:
- Line 47: "10 core fields + 12 caption_analysis subfields" → "12 fields (6 core + 8 caption_analysis + 5 metadata)"
- Line 509: Same update
- Line 722: Update section description

---

## Section 11: Taxonomy Examples (Multiple Locations)

### Status: ✅ NO CHANGE NEEDED

**Lines**: 203-207, 276-281, 405-408, 1168-1171, 1491-1500

**Rationale**: These show **taxonomy schema** (Stage 2.6 output).
Correctly use `audience_pain_points` and `trending_keywords`.

---

## Section 12: Meta References to Stage 2.7

### Status: 🟢 LOW - Minor text clarifications

**Lines**:
- Line 36-37: Add note about refined prompt
- Line 1352: Update "if Haiku misclassification rate >20%" context with "using refined 3-zone prompt"

---

## Summary Statistics

| Priority | Count | Sections |
|----------|-------|----------|
| 🔴 CRITICAL | 1 | Stage 2.3.4 Prompt (complete rewrite) |
| 🟡 MODERATE | 3 | Schema table, Example JSONs (8 instances), Validation logic |
| 🟢 LOW | 2 | Field count text references, Meta references |
| ✅ NO CHANGE | 5 | Taxonomy-related sections (correctly use old names) |

**Total Edits**: ~15-20 discrete changes

---

## Implementation Order

1. **✅ Critical First**: Update Stage 2.3.4 Classification Prompt (lines 392-453)
2. **Moderate**: Update Schema Definition Section 5.2.2 (lines 712-760)
3. **Moderate**: Update Validation Logic (lines 959-1004)
4. **Moderate**: Update 8 Example JSON outputs (lines 750, 1181, 1534, 1569)
5. **Low**: Update field count text references (lines 47, 509, 722)

---

## Verification Checklist

After all updates:
- [ ] All Stage 2.7 classification outputs use `pain_points` and `keywords`
- [ ] All Stage 2.6 taxonomy references still use `audience_pain_points` and `trending_keywords`
- [ ] Stage 2.3.4 prompt matches 2.7ClassificationCritique.md final prompt
- [ ] Schema table (Section 5.2.2) has 12 rows + caption_analysis note
- [ ] All 8 example JSONs updated with new field names
- [ ] Validation logic uses new field names
- [ ] Field count updated: "12 fields" not "10 core + 12 caption"
- [ ] `taxonomy_version` field documented

---

## Notes

**Key Principle**:
- Taxonomy (Stage 2.6) = Input = `audience_pain_points`, `trending_keywords`
- Classification (Stage 2.7) = Output = `pain_points`, `keywords`

This distinction must be preserved throughout the document.

**Change Log**:
| Date | Change | Status |
|------|--------|--------|
| 2025-01-28 | Document created | Tracking |
| 2025-01-28 | Updates in progress | Executing |
