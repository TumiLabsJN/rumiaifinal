# Decision: Show Engagement Only for Winning Buckets (Option A)

**Date**: 2025-10-23
**Status**: ✅ **APPROVED**
**Impact**: Stage 8 MVP reports, Template 1 structure

---

## Decision

**Show engagement metrics ONLY for the 3 winning buckets** in Template 1 (Hashtag → Client report), not all 8 duration buckets.

---

## Why This Decision

### Data Availability Constraint

**Current pipeline output structure**:

**File**: `selection_manifest.json`
**Location**: `/data/clients/{client}/hashtags/{hashtag}/top_contrastive/`

**Contains**:
- Video IDs for **3 winning buckets ONLY** (e.g., 13-18s, 18-33s, 60-90s)
- Top performers (40 videos per bucket)
- Bottom performers (10 videos per bucket)

**Does NOT contain**:
- ❌ Video IDs for non-winning buckets (0-3s, 3-9s, 9-13s, 33-60s, 90-120s)
- ❌ `avg_views` for any buckets

**Result**: Cannot calculate engagement for buckets we don't have video IDs for.

---

## What We Can Show

### Template 1: Section 2 Performance Table

**APPROVED FORMAT**:
```
Your Top 3 Performing Durations:

Duration | Avg Views  | Avg Engagement | Rating
---------|------------|----------------|------------
18-33s   | 490K       | 1.4%           | ⭐⭐⭐⭐⭐  ← BEST
13-18s   | 520K       | 1.2%           | ⭐⭐⭐⭐
60-90s   | 310K       | 1.3%           | ⭐⭐⭐

These 3 durations represent 75.9% of top-performing #hashtag content.
Your 9 creative reports focus exclusively on these high-opportunity durations.
```

**NOT showing**:
- All 8 buckets table with gaps/N/A values
- Duration distribution chart (show separately without engagement)

---

## Documentation Updates Required

### **1. Stage8MVP_Reports.md - Template 1**

**Section to modify**: "Page 2, Section 2: Performance by Duration"

**Change**:
- OLD: Shows all 8 buckets with engagement
- NEW: Shows only 3 winning buckets with engagement

**Lines**: ~136-157

---

### **2. Stage8MVP.md - Section 3.2**

**Section**: `extract_client_data.py` output format

**Change**:
- Update example output to show 3 buckets instead of 8
- Add note: "Only winning buckets shown (data availability constraint)"

**Lines**: ~901-913

---

### **3. Stage8MVP_Reports.md - Template 1 Dynamic Fields**

**Section**: Dynamic Fields table for Section 2

**Change**:
- Update to reflect 3 rows instead of 8
- Note: "Only winning buckets have engagement data"

**Lines**: ~151-159

---

## Future: If We Need All 8 Buckets

### What Would Be Required

**Option B Implementation** (if client demands full 8-bucket view):

**A. Modify Stage 1 / Stage 2.5 selection logic**:
```python
# Add to selection output
{
  "all_videos_by_bucket": {
    "0-3s": ["vid1", "vid2", ...],  # ALL videos, not just selected
    "3-9s": [...],
    // ... all 8 buckets
  },
  "selected_buckets": ["18-33s", "13-18s", "60-90s"]
}
```

**B. Calculate avg_views from original Apify data**:
- Load all video metadata from `unified_analysis/`
- Group by duration bucket
- Calculate avg views + engagement for all 8 buckets

**C. Update `extract_client_data.py`**:
- Read extended selection manifest
- Calculate engagement for all 8 buckets
- Output full table

**Effort**: +1 day (Stage 1 changes + extraction script updates)

---

## Rationale for Option A

**✅ Pros**:
1. **No code changes needed** - works with current pipeline output
2. **Honest** - only shows what we deeply analyzed (ML patterns, formulas)
3. **Focused** - highlights the 3 high-opportunity durations
4. **Consistent** - matches selection methodology (we picked top 3 for a reason)
5. **Professional** - no gaps or "N/A" values in tables

**Trade-offs**:
- Client doesn't see full 8-bucket market distribution
- Can still show duration distribution chart separately (just counts, no engagement)

---

## Implementation Status

- ✅ Decision documented (this file)
- ⏸️ Stage8MVP_Reports.md Template 1 updates needed
- ⏸️ Stage8MVP.md Section 3.2 updates needed

**Next Action**: Update Template 1 structure in Stage8MVP_Reports.md to reflect 3-bucket output
