# Field Validation Guide

**Purpose**: This guide documents the methodology, classifications, and standards used to validate dynamic fields in Stage 8 MVP Reports.

**Created**: 2025-01-24
**Scope**: Report 2 (Hashtag → Creator), Page 2 ("How to Execute")
**Status**: 18 fields validated across 4 sections

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Classifications & Validation Statuses](#2-data-classifications--validation-statuses)
3. [Data Types & Sources](#3-data-types--sources)
4. [Dynamic Functions Concept](#4-dynamic-functions-concept)
5. [Document Structure & Navigation](#5-document-structure--navigation)
6. [Validation Methodology](#6-validation-methodology)
7. [Quick Reference](#7-quick-reference)

---

## 1. Overview

### 1.1 What is Field Validation?

Field validation is the process of verifying that each dynamic field in a report template:
1. **Has a documented data source** (where the data comes from)
2. **Has a defined calculation/aggregation method** (how to process the data)
3. **Has the correct implementation status** (what's ready, what's pending)

### 1.2 Why Validate?

- ✅ **Ensures data integrity**: Every field can be populated with real data
- ✅ **Documents dependencies**: Clear understanding of what needs to be built
- ✅ **Prevents assumptions**: Don't mark fields as ready based on documentation alone
- ✅ **Tracks implementation progress**: Know what's done and what's pending

### 1.3 Documents Involved

| Document | Purpose |
|----------|---------|
| `Stage8MVP_Reports.md` | Report templates with dynamic field definitions |
| `Stage8MVP.md` | Function reference documentation (Section 0.5) |
| `ContentAnalysisCHILDTI.md` | Qualitative data schema (Stage 2.7) |
| `SystemArchitecturev2.md` | Temporal windows schema (quantitative data) |

---

## 2. Data Classifications & Validation Statuses

### 2.1 Three Validation Cases

We identified **three distinct cases** for field validation:

#### **Case A: ✅ READY (Function + Data Exists)**

**Criteria**:
1. ✅ Function exists and is documented
2. ✅ Data source exists and is populated
3. ✅ Field can be populated in reports NOW

**Status Label**: `✅ **READY** (Function exists, data verified)`

**Example**: *None found in current validation (all fields pending Stage 2.7 or Stage 7)*

---

#### **Case B: ⚠️ FUNCTION READY, AWAITING DATA**

**Criteria**:
1. ✅ Function exists (wrapper or base function documented)
2. ❌ Data source doesn't exist yet (pipeline stage not run)
3. ❌ Field CANNOT be populated until data pipeline runs

**Status Labels**:
- `⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA**` - Content Analysis data missing
- `⚠️ **FUNCTION READY, AWAITING STAGE 2.6 TAXONOMY**` - Taxonomy file missing

**Applies To** (7 fields):
- Content Categories (Top 3)
- Engagement Drivers (Top 3)
- Hook Strategies (Top 3)
- Pain Points (Top 5)
- Keywords (Top 8)
- Content Tactics (Top 4)
- CTA Type

**Why Functions Exist But Data Doesn't**:
- Base function: `aggregate_content_classifications()` (Section 0.5.1) ✅
- Wrapper function: `get_top_n_from_field()` (Section 0.5.1.1) ✅
- Data source: Stage 2.7 classification files `{video_id}_content.json` ❌ (not run yet)

---

#### **Case C: ⚠️ PENDING QUANTITATIVE LLM OUTPUT (STAGE 7)**

**Criteria**:
1. ✅ Raw data exists in pipeline (temporal_windows_updated.json files)
2. ❌ Stage 7 aggregation layer not implemented
3. ❌ Stage 7 needs to calculate averages, map categories, format for reports

**Status Label**: `⚠️ **Pending Quantitative LLM Output** (Stage 7)`

**Applies To** (9 fields):
- Word count (semantic)
- Visual direction
- Energy description (Hook)
- Scene changes rate
- Text overlay count
- Energy standard (Build & Prove)
- CTA Example Phrase
- Peak Energy Note
- Visual Cue

**Why Pending**:
- Raw data like `temporal_windows.hook.word_count` exists ✅
- But Stage 7 needs to:
  1. Aggregate across winning cluster (40+ videos)
  2. Calculate averages
  3. Map to semantic categories (e.g., word_count → "2 sentences, moderate pace")
  4. Format for report display

---

### 2.2 Decision Tree: Which Status to Use?

```
Does the function exist?
  ├─ NO → Document as "NEW FUNCTION NEEDED"
  └─ YES → Does the data source exist?
      ├─ NO → Which pipeline stage creates it?
      │   ├─ Stage 2.6 → "AWAITING STAGE 2.6 TAXONOMY"
      │   ├─ Stage 2.7 → "AWAITING STAGE 2.7 DATA"
      │   └─ Stage 7 → "Pending Quantitative LLM Output"
      └─ YES → Is the field ready to use?
          ├─ YES → ✅ "READY"
          └─ NO → ⚠️ Document the blocker
```

---

## 3. Data Types & Sources

### 3.1 Qualitative Data (Content Analysis)

**Source**: Stage 2.7 Content Analysis
**Documentation**: `ContentAnalysisCHILDTI.md`
**File Location**: `{bucket_path}/content_analysis/{video_id}_content.json`

**Schema**:
```json
{
  "video_id": "7526250443832331550",
  "content_category": "recipe_tutorial",        // Single selection
  "hook_strategy": "problem_solution",          // Single selection
  "pain_points": ["bloating", "low_energy"],    // Multiple selection
  "keywords": ["protein", "gut_health"],        // Multiple selection
  "engagement_drivers": ["personal_testimony"], // Multiple selection
  "content_tactics": ["direct_to_camera"],      // Multiple selection
  "caption_analysis": {
    "cta_type": "link_in_bio",
    "emoji_usage": "some",
    "caption_length": "short"
  }
}
```

**Characteristics**:
- ✅ Human-interpretable categories
- ✅ Discovered patterns from LLM analysis (Stage 2.6)
- ✅ Classified per video by LLM (Stage 2.7)
- ✅ Requires aggregation across cluster for reports

**Used For**:
- Content Categories, Hook Strategies
- Pain Points, Keywords
- Engagement Drivers, Content Tactics
- CTA Type

---

### 3.2 Quantitative Data (Temporal Windows)

**Source**: Stage 2 ML Pipeline (9 services)
**Documentation**: `SystemArchitecturev2.md`
**File Location**: `{bucket_path}/analysis/insights/{video_id}_temporal_windows_updated.json`

**Schema** (simplified):
```json
{
  "video_id": "7526250443832331550",
  "temporal_windows": {
    "hook": {
      "word_count": 12,
      "energy_level": 0.67,
      "eye_contact_rate": 0.85,
      "average_face_size": 0.42
    },
    "middle_segments": [
      {
        "segment_name": "segment_1",
        "scene_count": 2,
        "text_overlay_count": 3,
        "energy_level": 0.55,
        "duration": 3.5
      }
    ],
    "closing": {
      "energy_level": 0.91
    }
  }
}
```

**Characteristics**:
- ✅ Numeric/measurable values
- ✅ Generated by ML models (YOLO, Whisper, MediaPipe, etc.)
- ✅ Already aggregated per video
- ✅ Requires Stage 7 to aggregate across cluster and categorize

**Used For**:
- Word count, Energy levels
- Visual direction (eye contact, face size)
- Scene changes, Text overlays

---

### 3.3 Key Difference

| Aspect | Qualitative | Quantitative |
|--------|-------------|--------------|
| **Data Type** | Categories, labels | Numbers, metrics |
| **Source** | LLM classification | ML model detection |
| **Example** | "personal_testimony" | 0.67 (energy level) |
| **Documentation** | ContentAnalysisCHILDTI.md | SystemArchitecturev2.md |
| **Aggregation** | Frequency count, Top N | Average, sum, range |

---

## 4. Dynamic Functions Concept

### 4.1 The Problem They Solve

**Context**: Competitor analysis data is stored in directories like:
```
/data/clients/{client}/competitors/drinkpoppi/top_contrastive/
/data/clients/{client}/competitors/nike/top_top/
/data/clients/{client}/competitors/lululemon/top_contrastive/
```

**Notice**: The strategy directory varies (`top_contrastive` vs `top_top`) depending on analysis mode.

**Challenge**: How do functions find the right path when the strategy directory isn't consistent?

---

### 4.2 Old Approach (Static Paths) ❌

```python
def calculate_views(file_path: str):
    """User must pass full path manually"""
    data = load("/data/clients/test/competitors/nike/top_contrastive/winner_analysis.json")
    return data['avg_views']

# Problem: Caller must know exact strategy for each competitor!
views = calculate_views("/data/clients/test/competitors/nike/top_contrastive/winner_analysis.json")
```

**Issues**:
- ❌ Caller must know internal directory structure
- ❌ Hard to maintain (strategy names might change)
- ❌ Inconsistent function signatures
- ❌ Error-prone (easy to pass wrong path)

---

### 4.3 New Approach (Dynamic Discovery) ✅

```python
def calculate_views(client_id: str, competitor_handle: str):
    """Function discovers the path automatically"""
    base = f"/data/clients/{client_id}/competitors/{competitor_handle}/"

    # Find directory starting with "top_" (could be top_contrastive, top_top, etc.)
    dirs = [d for d in os.listdir(base) if d.startswith('top_')]
    if not dirs:
        raise ValueError(f"No analysis directory found for {competitor_handle}")

    strategy_dir = dirs[0]  # Use whatever exists

    # Now load the file
    data = load(f"{base}/{strategy_dir}/winner_analysis.json")
    return data['avg_views']

# Clean interface: just pass client and competitor
views = calculate_views("test_run", "@nike")
```

**Benefits**:
- ✅ Caller just passes `(client_id, competitor_handle)` - simple!
- ✅ Function handles path discovery internally
- ✅ Works regardless of strategy used
- ✅ Consistent signatures across all functions
- ✅ Easier to test and debug
- ✅ Self-documenting code

---

### 4.4 When to Use Dynamic Functions

**Use Dynamic Discovery When**:
- Directory structure varies between runs (mode/strategy folders)
- Multiple valid paths exist for the same data
- You want to abstract internal structure from callers

**Use Static Paths When**:
- File location is always the same
- Path is configuration-driven
- Performance is critical (avoid directory scanning)

---

## 5. Document Structure & Navigation

### 5.1 Two-Document System

RumiAI uses a **separation of concerns** approach:

| Document | Purpose | Audience |
|----------|---------|----------|
| **Stage8MVP_Reports.md** | Report templates + field definitions | Report designers, stakeholders |
| **Stage8MVP.md** | Function implementation details | Developers, engineers |

---

### 5.2 Stage8MVP_Reports.md (Field Definitions)

**Purpose**: Define WHAT data each report field needs

**Structure**:
```
Report 2: Hashtag → Creator
├── Page 1: What's Working
└── Page 2: How to Execute
    ├── VIDEO CATEGORY
    ├── HOOK (0-3s)
    ├── BUILD & PROVE (3s to last 3s)
    └── CLOSING (last 3s)
```

**Each Section Contains**:
1. **Template Text** - What users see in the report
2. **Dynamic Fields Table** - Data requirements for each field

**Dynamic Fields Table Format**:
```markdown
| Template Field | Source | JSON Field/Calculation | Data Type | Example | Validated |
|----------------|--------|------------------------|-----------|---------|-----------|
| Keywords (Top 8) | Stage 7 | **Base Function**: `aggregate_content_classifications()` → **Wrapper**: `get_top_n_from_field(field="keywords", n=8)` | Array[String] | ["protein", "gut_health", ...] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
```

---

### 5.3 Stage8MVP.md Section 0.5 (Function Documentation)

**Purpose**: Define HOW to implement the functions

**Structure**:
```
Section 0.5: Data Processing Functions
├── 0.5.1: Content Analysis Aggregation
│   └── aggregate_content_classifications()
├── 0.5.1.1: Top N Selection Wrapper
│   └── get_top_n_from_field()
├── 0.5.2: QR Code Video Selection
├── 0.5.3: Hashtag Extraction
└── ... (more functions)
```

**Each Function Contains**:
1. **Purpose** - What problem it solves
2. **Input Parameters** - What it needs
3. **Returns** - What it outputs
4. **Example Implementation** - Code snippet
5. **Usage Examples** - How to call it

**Example**:
```markdown
#### 0.5.1.1: Top N Selection Wrapper

**Function**: `get_top_n_from_field(bucket_path, field_name, n=3, performance_group="top")`

**Purpose**: Extract Top N items from a classification field for report display

**Input Parameters**:
- `bucket_path`: Path to bucket folder
- `field_name`: "content_category", "keywords", etc.
- `n`: Number of top items (default: 3)

**Returns**: Array of top N item names
```

---

### 5.4 Navigation Flow

**When Validating a Field**:
1. Start in `Stage8MVP_Reports.md` → Find the report section
2. Read the Dynamic Fields table → Identify the field
3. Check the "Source" column → Find the function reference (e.g., "Section 0.5.1")
4. Go to `Stage8MVP.md` → Section 0.5.1
5. Read function documentation → Understand implementation
6. Verify data sources exist → Update validation status

---

## 6. Validation Methodology

### 6.1 Step-by-Step Process

**For Each Report Section**:

#### **Step 1: Identify Section to Validate**
- Work through reports section by section
- Each section has template text + Dynamic Fields table

#### **Step 2: Extract Dynamic Fields**
- List all fields in the section
- Note current validation status (if any)

#### **Step 3: Validate Field-by-Field**
For each field:

1. **Find Data Source**
   - Where does the data come from? (Stage 2.6, 2.7, 7, etc.)
   - Does the file/data exist?
   - Example: `{bucket_path}/content_analysis/{video_id}_content.json`

2. **Verify Function Exists**
   - Is there a function to process this data?
   - Check `Stage8MVP.md` Section 0.5
   - Example: `get_top_n_from_field()` in Section 0.5.1.1

3. **Check Calculation Method**
   - How is the data aggregated/calculated?
   - Example: "Sum all `top_performers` array lengths across buckets"

4. **Determine Validation Status**
   - Use decision tree from Section 2.2
   - Apply appropriate status label

5. **Update Dynamic Fields Table**
   - Add "Validated" column if missing
   - Mark field with correct status
   - Document function references

#### **Step 4: Share Findings & Get Approval**
- Show what you found (don't just document silently)
- Wait for user confirmation before continuing
- Ask permission to document changes

---

### 6.2 Validation Principles

**1. Only Validate What You Can Verify**
- ❌ DON'T mark fields as validated based on documentation alone
- ✅ DO inspect actual pipeline output files to confirm

**2. Use Actual Examples**
- ❌ Generic: `"@rival_brand"`
- ✅ Real: `"@drinkpoppi"` (from actual config.json)

**3. Document Calculation Methods**
- ❌ Vague: "Sum of videos"
- ✅ Specific: "Sum all `top_performers` + `bottom_performers` array lengths across all buckets in `videos_by_bucket`"

**4. Distinguish Function Types**
- **Base Function**: Returns raw aggregated data (Counter objects)
- **Wrapper Function**: Formats data for specific report needs
- Document both when used together

---

### 6.3 Example Validation

**Field**: Keywords (Top 8)

**Step 1 - Find Data Source**:
```bash
# Search for content analysis files
find /data/clients -name "*_content.json" | head -3
```
Result: ❌ No files found (Stage 2.7 not run yet)

**Step 2 - Verify Function**:
Check `Stage8MVP.md` Section 0.5.1.1:
- Base: `aggregate_content_classifications()` ✅ Exists
- Wrapper: `get_top_n_from_field()` ✅ Exists

**Step 3 - Check Schema**:
From `ContentAnalysisCHILDTI.md`:
```json
{
  "keywords": ["protein", "gut_health"]  // ✅ Field exists in schema
}
```

**Step 4 - Determine Status**:
- Function exists ✅
- Data doesn't exist ❌ (Stage 2.7 pending)
- **Status**: ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA**

**Step 5 - Update Table**:
```markdown
| Keywords (Top 8) | Stage 7 | **Base**: `aggregate_content_classifications()` → **Wrapper**: `get_top_n_from_field(field="keywords", n=8)` | Array[String] | [...] | ⚠️ **FUNCTION READY, AWAITING STAGE 2.7 DATA** |
```

---

## 7. Quick Reference

### 7.1 Report 2, Page 2 - All Validated Fields

**Total**: 18 fields across 4 sections

#### **VIDEO CATEGORY** (3 fields)
| Field | Status |
|-------|--------|
| Content Categories (Top 3) | ⚠️ FUNCTION READY, AWAITING STAGE 2.7 DATA |
| Content Category Descriptions | ⚠️ FUNCTION READY, AWAITING STAGE 2.6 TAXONOMY |
| Engagement Drivers (Top 3) | ⚠️ FUNCTION READY, AWAITING STAGE 2.7 DATA |

**Note**: Engagement Drivers display as title case, no descriptions needed (self-explanatory)

---

#### **HOOK** (5 fields)
| Field | Status |
|-------|--------|
| Hook Strategies (Top 3) | ⚠️ FUNCTION READY, AWAITING STAGE 2.7 DATA |
| Hook Strategy Descriptions | ⚠️ FUNCTION READY, AWAITING STAGE 2.6 TAXONOMY |
| Word count (semantic) | ⚠️ Pending Quantitative LLM Output (Stage 7) |
| Visual direction | ⚠️ Pending Quantitative LLM Output (Stage 7) |
| Energy description | ⚠️ Pending Quantitative LLM Output (Stage 7) |

---

#### **BUILD & PROVE** (6 fields)
| Field | Status |
|-------|--------|
| Pain Points (Top 5) | ⚠️ FUNCTION READY, AWAITING STAGE 2.7 DATA |
| Keywords (Top 8) | ⚠️ FUNCTION READY, AWAITING STAGE 2.7 DATA |
| Content Tactics (Top 4) | ⚠️ FUNCTION READY, AWAITING STAGE 2.7 DATA |
| Scene changes rate | ⚠️ Pending Quantitative LLM Output (Stage 7) |
| Text overlay count | ⚠️ Pending Quantitative LLM Output (Stage 7) |
| Energy standard | ⚠️ Pending Quantitative LLM Output (Stage 7) |

---

#### **CLOSING** (4 fields)
| Field | Status |
|-------|--------|
| CTA Type | ⚠️ FUNCTION READY, AWAITING STAGE 2.7 DATA |
| CTA Example Phrase | ⚠️ Pending Quantitative LLM Output (Stage 7) |
| Peak Energy Note | ⚠️ Pending Quantitative LLM Output (Stage 7) |
| Visual Cue | ⚠️ Pending Quantitative LLM Output (Stage 7) |

---

### 7.2 Status Summary

| Status | Count | Next Action |
|--------|-------|-------------|
| ⚠️ FUNCTION READY, AWAITING STAGE 2.7 DATA | 7 | Run Stage 2.7 Content Analysis |
| ⚠️ FUNCTION READY, AWAITING STAGE 2.6 TAXONOMY | 2 | Run Stage 2.6 Taxonomy Discovery + Curation |
| ⚠️ Pending Quantitative LLM Output (Stage 7) | 9 | Implement Stage 7 LLM Report Generation |
| **Total** | **18** | Complete Stages 2.6, 2.7, and 7 |

---

### 7.3 Key Functions Reference

| Function | Section | Purpose |
|----------|---------|---------|
| `aggregate_content_classifications()` | 0.5.1 | Aggregate Stage 2.7 classifications into Counter objects |
| `get_top_n_from_field()` | 0.5.1.1 | Extract Top N items from classification field |
| `get_descriptions_from_taxonomy()` | TBD | Extract definitions from taxonomy for categories/hooks |
| `get_visual_direction()` | 0.5.7 | Categorize visual direction from eye contact + face size |

---

## Appendix A: Common Issues & Solutions

### Issue 1: File Modified Externally

**Symptom**: Edit tool fails with "File has been modified"

**Cause**: External process (linter, auto-save) modifying file

**Solution**: Re-read file before editing

---

### Issue 2: Validation Status Confusion

**Symptom**: Unclear which status to use

**Cause**: Misunderstanding of what's implemented vs what exists as data

**Solution**: Use decision tree in Section 2.2

**Remember**:
- Function exists ≠ Field is ready
- Data in pipeline ≠ Field is ready
- Both function AND data must exist for ✅ READY

---

### Issue 3: Static vs Dynamic Descriptions

**Symptom**: Trying to hardcode descriptions for discovered patterns

**Cause**: Assuming all categories are predefined

**Solution**:
- Categories with definitions in taxonomy (content_categories, hook_strategies) → Read from taxonomy
- Categories without definitions (engagement_drivers) → Use smart formatting (snake_case → Title Case)

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Aggregation** | Combining data from multiple videos into summary statistics |
| **Base Function** | Core function that returns raw data (e.g., Counter objects) |
| **Bucket** | Duration-based grouping of videos (e.g., 13-18s, 18-33s) |
| **Classification** | LLM-assigned category for a video (from Stage 2.7) |
| **Cluster** | K-means grouping of videos with similar patterns (from Stage 6) |
| **Content Analysis** | Qualitative categorization of video content (Stage 2.6/2.7) |
| **Dynamic Field** | Report field populated by pipeline data (not static text) |
| **Qualitative Data** | Categorical/label data from LLM classification |
| **Quantitative Data** | Numeric/measurable data from ML models |
| **Taxonomy** | Structured classification system with 6 categories |
| **Temporal Windows** | Time-based segments (hook, middle, closing) with quantitative metrics |
| **Wrapper Function** | Function that formats base function output for specific use case |
| **Winning Cluster** | K-means cluster with highest performing videos |

---

**End of Guide**

For questions or updates, refer to:
- Stage8MVP_Reports.md (report templates)
- Stage8MVP.md (function documentation)
- ContentAnalysisCHILDTI.md (content analysis schema)
