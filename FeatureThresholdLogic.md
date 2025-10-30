# Feature Threshold Logic - Semantic Range Definition Process

## 🎯 Project Context

**What we're doing**: Implementing Approach C (Full Semantic Dictionaries) for RumiAI Stage 7 `supplementary_insights` output.

**Goal**: Convert raw numeric ML features into creator-friendly semantic labels.
- Example: `average_face_size: 0.058` → `"Face size in opening: 72% of top performers use medium shots vs 15% of bottom"`

**Why**: Current output has technical jargon (`hook_energy_max: 0.13 in top vs 0.16 in bottom`) that creators can't use. We need actionable language (`"High energy delivery in opening: 72% of top performers vs 15% of bottom"`).

**Approach**: Python-only formatting (no LLM) with semantic interpretation dictionaries for all 26 base features.

**Key Decision**: Using Approach C = Full semantic dictionaries with human-readable labels (e.g., "wide shot", "close-up", "high energy").

---

## 📋 Standard Workflow (4 Steps)

### **Step 1: Research and Define Semantic Ranges**

**Process**:
1. Extract production data for the feature across all buckets
2. Identify data range (min, max, typical values)
3. Determine metric type (ratio/variance/count/continuous/duration)
4. Determine direction (higher_is_more / lower_is_better / neutral)
5. Propose semantic ranges with thresholds

**Deliverable**: Initial range definitions with rationale

---

### **Step 2: Present Methodology Table**

**Process**: For each feature, document the methodology used to determine thresholds

**Format**:
| Feature | Method Used | Data Range | Rigor Level | Rationale |
|---------|-------------|------------|-------------|-----------|
| `feature_name` | Quartile/Semantic/Domain | (min, max) | High/Medium/Low | Brief explanation |

**Three Methodology Types**:

1. **Quartile-Based (Data-Driven)** 📊
   - Use: Continuous metrics without clear semantic categories
   - Process: Calculate 25th, 50th, 75th percentiles from production data
   - Example: energy_level, pitch_scatter_ratio
   - Rigor: High

2. **Semantic Categories (Logic-Driven)** 🧠
   - Use: Count-based features with obvious meaning
   - Process: Define logical groupings (e.g., 1=solo, 2=duo, 3-5=small group)
   - Example: person_count, day_of_week
   - Rigor: High (when categories are culturally obvious)

3. **Domain Expertise (Industry Standards)** 📚
   - Use: Features with established industry terminology
   - Process: Apply videography/audio engineering/creator standards
   - Example: average_face_size (cinematography shot types)
   - Rigor: Medium (depends on how well standards map to data)

**Deliverable**: Methodology table showing decision process for each feature

---

### **Step 3: Present Output Examples**

**Process**: For each feature, show concrete examples of what values produce what labels

**Format**:
```
Feature: average_face_size

Example outputs:
- Value: 0.045 → Label: "wide shot" → Description: "face occupies <6% of frame"
- Value: 0.078 → Label: "medium shot" → Description: "face occupies 6-10% of frame"
- Value: 0.125 → Label: "close-up" → Description: "face occupies 10-20% of frame"
- Value: 0.350 → Label: "extreme close-up" → Description: "face occupies >20% of frame"

Context in full output:
"Face size in opening: 72% of top performers use wide shots vs 15% of bottom (avg: medium shot)"
```

**Purpose**: Ensure thresholds make semantic sense when applied to real values

**Deliverable**: 3-5 example value-to-label mappings per feature

---

### **Step 4: Discussion and Adjustment**

**Process**: Review all three components together and identify issues

**Common issues to check**:
- [ ] Do the threshold values make sense given the data range?
- [ ] Are the semantic labels appropriate for the values they represent?
- [ ] Do boundaries feel right? (e.g., is 0.06 really the boundary between "wide" and "medium"?)
- [ ] Are there edge cases or outliers not covered?
- [ ] Does the output sound natural when used in a sentence?

**Deliverable**: Finalized semantic ranges, ready for implementation

---

## 🎯 Application: Category 1 - Visual Composition (Completed)

### **Step 1: Research Results**

#### Feature 1: average_face_size
- **Data range**: 0.034 - 0.142
- **Metric type**: ratio (0.0-1.0 scale)
- **Direction**: higher_is_closer
- **Production data**:
  - hook: top=0.057-0.096, bottom=0.069-0.129
  - closing: top=0.057, bottom=0.117
  - Thresholds: high=0.064-0.142, low=0.034-0.066

#### Feature 2: person_count
- **Data range**: 1.0 - 25.4 (outliers exist)
- **Metric type**: count
- **Direction**: neutral
- **Production data**:
  - closing: top=3.59, bottom=25.4
  - Thresholds: high=2.0, low=1.0

#### Feature 3: object_count
- **Data range**: 2.28 - 7.68
- **Metric type**: count
- **Direction**: neutral
- **Production data**:
  - middle_aggregate: top=6.24, bottom=7.0
  - Thresholds: high=7.68, low=2.28

#### Feature 4: overlay_unique_count
- **Data range**: 1.0 - 5.08
- **Metric type**: count
- **Direction**: neutral
- **Production data**:
  - hook: top=2.83, bottom=3.58
  - closing: top=2.91, bottom=5.08
  - Thresholds: high=3.0, low=1.0-2.0

---

### **Step 2: Methodology Table**

| Feature | Method Used | Data Range | Rigor Level | Rationale |
|---------|-------------|------------|-------------|-----------|
| `average_face_size` | Domain + Data | (0.034, 0.142) | Medium | Applied cinematography shot types (wide/medium/close-up/extreme) to observed data range. Thresholds (0.06, 0.10, 0.20) are estimates based on rough quartiles + domain knowledge. Could be refined with exact quartile calculation. |
| `person_count` | Semantic | (1.0, 5.0 typical) | High | Count-based with obvious semantic categories: 1=solo, 2=duo, 3-5=small group, 5+=large group. Culturally meaningful, not arbitrary. |
| `object_count` | Data Range | (2.28, 7.68) | Low | Rough estimates based on observed range. Should calculate quartiles for more rigor. Thresholds (3.0, 6.0, 10.0) are approximations. |
| `overlay_unique_count` | Data Range | (1.0, 5.08) | Low | Rough estimates based on observed range. Thresholds (0.5, 2.5, 4.5) are approximations. Should validate with quartiles. |

---

### **Step 3: Output Examples**

#### **Feature 1: average_face_size**

**Proposed ranges**:
```
(0.0, 0.06, 'wide shot', 'face occupies <6% of frame')
(0.06, 0.10, 'medium shot', 'face occupies 6-10% of frame')
(0.10, 0.20, 'close-up', 'face occupies 10-20% of frame')
(0.20, 1.0, 'extreme close-up', 'face occupies >20% of frame')
```

**Example outputs**:
- **Value: 0.045** → Label: **"wide shot"** → Description: "face occupies <6% of frame"
- **Value: 0.058** (top performer avg) → Label: **"wide shot"** → Description: "face occupies <6% of frame"
- **Value: 0.078** → Label: **"medium shot"** → Description: "face occupies 6-10% of frame"
- **Value: 0.084** (bottom performer avg) → Label: **"medium shot"** → Description: "face occupies 6-10% of frame"
- **Value: 0.125** → Label: **"close-up"** → Description: "face occupies 10-20% of frame"
- **Value: 0.350** → Label: **"extreme close-up"** → Description: "face occupies >20% of frame"

**In context**:
```
"Face size in opening: 72% of top performers use wide shots vs 15% of bottom (avg: medium shot)"
```

---

#### **Feature 2: person_count**

**Proposed ranges**:
```
(0, 1.5, 'solo', 'single person on screen')
(1.5, 2.5, 'duo', 'two people visible')
(2.5, 5.0, 'small group', '3-5 people visible')
(5.0, 100, 'large group', 'more than 5 people')
```

**Example outputs**:
- **Value: 1.0** → Label: **"solo"** → Description: "single person on screen"
- **Value: 1.8** → Label: **"duo"** → Description: "two people visible"
- **Value: 3.6** (top performer avg) → Label: **"small group"** → Description: "3-5 people visible"
- **Value: 4.5** → Label: **"small group"** → Description: "3-5 people visible"
- **Value: 8.0** → Label: **"large group"** → Description: "more than 5 people"
- **Value: 25.4** (outlier) → Label: **"large group"** → Description: "more than 5 people"

**In context**:
```
"Number of people in closing: 68% of top performers use small group vs 29% of bottom (avg: large group)"
```

---

#### **Feature 3: object_count**

**Proposed ranges**:
```
(0, 3.0, 'minimal objects', 'very few objects/props visible')
(3.0, 6.0, 'moderate objects', 'balanced visual elements')
(6.0, 10.0, 'many objects', 'rich visual environment')
(10.0, 100, 'cluttered', 'visually dense/busy composition')
```

**Example outputs**:
- **Value: 2.0** → Label: **"minimal objects"** → Description: "very few objects/props visible"
- **Value: 2.28** (observed low) → Label: **"minimal objects"** → Description: "very few objects/props visible"
- **Value: 4.5** → Label: **"moderate objects"** → Description: "balanced visual elements"
- **Value: 6.24** (top performer avg) → Label: **"many objects"** → Description: "rich visual environment"
- **Value: 7.68** (observed high) → Label: **"many objects"** → Description: "rich visual environment"
- **Value: 12.0** → Label: **"cluttered"** → Description: "visually dense/busy composition"

**In context**:
```
"Visual elements across middle: 72% of top performers use many objects vs 15% of bottom (avg: moderate objects)"
```

---

#### **Feature 4: overlay_unique_count**

**Proposed ranges**:
```
(0, 0.5, 'no text', 'no text overlays present')
(0.5, 2.5, 'minimal text', '1-2 text elements')
(2.5, 4.5, 'moderate text', '3-4 text elements')
(4.5, 20, 'heavy text', '5+ text elements')
```

**Example outputs**:
- **Value: 0.0** → Label: **"no text"** → Description: "no text overlays present"
- **Value: 1.5** → Label: **"minimal text"** → Description: "1-2 text elements"
- **Value: 2.83** (top performer avg) → Label: **"moderate text"** → Description: "3-4 text elements"
- **Value: 3.8** → Label: **"moderate text"** → Description: "3-4 text elements"
- **Value: 5.08** (bottom performer avg) → Label: **"heavy text"** → Description: "5+ text elements"
- **Value: 8.0** → Label: **"heavy text"** → Description: "5+ text elements"

**In context**:
```
"Text overlays in opening: 72% of top performers use moderate text vs 15% of bottom (avg: heavy text)"
```

---

### **Step 4: Visual Calibration & Finalization** ✅ COMPLETED

**Recalibration Process**: Used 6 real TikTok images to validate thresholds

#### **Visual Evidence Analysis**

| Image | Value | Visual Description | Initial Label (OLD) | Calibrated Label (NEW) | Match? |
|-------|-------|-------------------|---------------------|----------------------|--------|
| Img06 | 0.0312 | Full body + environment | wide shot | wide shot | ✅ |
| Img05 | 0.0514 | Upper body (chest up) | **wide shot** ❌ | **medium shot** ✅ | Adjusted |
| Img04 | 0.0864 | Upper body (chest up) | medium shot | medium shot | ✅ |
| Img03 | 0.1127 | Head + shoulders | close-up | close-up | ✅ |
| Img02 | 0.2013 | Tight head shot | extreme close-up | **tight close-up** ✅ | Added category |
| Img01 | 0.4561 | Face dominates frame | extreme close-up | extreme close-up | ✅ |

#### **Key Findings from Visual Calibration**:

1. **Original threshold (0.06) was too high for "wide shot"**
   - 0.0514 showed upper body (medium), not full body (wide)
   - Adjusted boundary from 0.06 → **0.04**

2. **Needed more granularity above 0.15**
   - 0.2013 (tight head) vs 0.4561 (face dominates) looked very different
   - Added **"tight close-up"** category (0.15-0.30)
   - Now have 5 categories instead of 4

3. **Final calibrated ranges for average_face_size**:
   ```
   (0.0, 0.04, 'wide shot', 'full body visible with environment')
   (0.04, 0.09, 'medium shot', 'upper body (chest/waist up)')
   (0.09, 0.15, 'close-up', 'head and shoulders prominent')
   (0.15, 0.30, 'tight close-up', 'head fills most of frame')
   (0.30, 1.0, 'extreme close-up', 'face dominates entire frame')
   ```

#### **Validation Results**:

✅ **average_face_size**: RECALIBRATED - 5 categories, thresholds adjusted based on visual evidence
✅ **person_count**: APPROVED - semantic categories are logical and appropriate
✅ **object_count**: APPROVED - rough estimates acceptable for now, can refine later if needed
✅ **overlay_unique_count**: APPROVED - labels make sense for observed values

**Status**: Category 1 (Visual Composition) - **FINALIZED** after visual calibration

---

## 📊 Status Tracker

| Category | Features | Step 1 | Step 2 | Step 3 | Step 4 | Status |
|----------|----------|--------|--------|--------|--------|--------|
| **Visual Composition** | 4 | ✅ | ✅ | ✅ | ⏸️ | Awaiting review |
| **Energy/Performance** | 4 | ⏳ | ⏳ | ⏳ | ⏳ | Not started |
| **Audio/Speech** | 4 | ⏳ | ⏳ | ⏳ | ⏳ | Not started |
| **Eye Contact/Gaze** | 3 | ⏳ | ⏳ | ⏳ | ⏳ | Not started |
| **Scene/Pacing** | 4 | ⏳ | ⏳ | ⏳ | ⏳ | Not started |
| **Movement/Temporal/Metadata** | 7 | ⏳ | ⏳ | ⏳ | ⏳ | Not started |

**Total**: 26 features across 6 categories

---

## 🔄 Next Steps

1. ✅ Review Category 1 output examples (Step 4)
2. Make any adjustments to thresholds based on discussion
3. Proceed to Category 2 following the same 4-step process
4. Repeat for all 6 categories

---

## 🔄 File Update Workflow

After completing each category's 4-step process, update **THREE files** in this order:

### **1. FeatureThresholdLogic.md** (This File) - Process Documentation
- **Purpose**: Document methodology, visual examples, recalibration decisions
- **Location**: Step 4 section for each category
- **What to update**: Add visual calibration table, key findings, validation results

### **2. LLMOutputFix.md** - Final Implementation Decisions
- **Purpose**: Finalized semantic range definitions for implementation
- **Location**: Search for "CATEGORY X: [Name] (N features)" section
- **What to update**: The SEMANTIC_INTERPRETATIONS dictionary entries
- **Format**:
  ```python
  'feature_name': {
      'metric_type': 'ratio|variance|count|continuous|duration',
      'direction': 'higher_is_more|lower_is_better|higher_is_closer|neutral',
      'unit': 'descriptive unit',
      'data_range': (min, max),  # From production/visual data
      'ranges': [
          (min1, max1, 'label1', 'description1'),
          (min2, max2, 'label2', 'description2'),
          # ... 3-5 ranges per feature
      ],
      'notes': 'Methodology: [method used]. [Key findings]. [Any recalibration notes].'
  },
  ```

### **3. config/semantic_interpretations.py** - Code Implementation
- **Purpose**: Actual Python code used by Stage 7
- **Location**: Same structure as LLMOutputFix.md, just in Python format
- **What to update**: The SEMANTIC_INTERPRETATIONS dictionary
- **Note**: Keep in sync with LLMOutputFix.md

---

## 📝 Notes

- This document will be updated as each category is completed
- All final definitions go into `config/semantic_interpretations.py`
- Rationale and methodology stay documented here for transparency
- Visual calibration examples stay here as reference for future features
