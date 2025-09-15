# Feature Benchmark Framework for ML Pipeline
**Created**: 2025-01-28
**Purpose**: Standardized criteria for evaluating features across all 8 RumiAI analysis flows

## 📍 START HERE - Document Guide

**This is the main guide document for the feature review process.**

1. **Read this document first** to understand the methodology and criteria
2. **Review MLA.md files** (e.g., creative_densityMLA.md) for feature compatibility data
3. **Use FeaturesMLMVP.md** to document features selected for the MVP
4. **Use FutureExtraFeatures.md** to document rejection reasons

**Key Principle**: A feature goes to FeaturesMLMVP.md if it's adaptable for AT LEAST ONE algorithm (RF or K-means). Features can appear in both files if they work for one algorithm but not the other.

---

## Manual Review Methodology

### Decision Flow Diagram

```
Feature Presented by Jorge
         ↓
Check RF Compatibility
         ↓
Check KM Compatibility  
         ↓
At least one = Yes?
    ↙        ↘
   YES        NO
    ↓          ↓
Goes to     Goes to
FeaturesMLMVP.md   FutureExtraFeatures.md
    +              (completely rejected)
    ↓
Any rejections?
    ↙    ↘
   YES    NO
    ↓      ↓
Also in   Done
FutureExtraFeatures.md
(document rejection reasons)
```

### Step-by-Step Process for Each Feature

#### Processing Approach: Individual Review
**Important**: We review features ONE BY ONE, not in batches. Jorge will present each feature individually for joint decision-making.

#### Step 1: Jorge Presents Feature
Jorge will share:
- **Flow name**: e.g., Creative Density
- **Feature name**: e.g., accelerationPattern  
- **What it is**: Brief description (e.g., "How quickly density changes (rate of change)")
- **Jorge's position**: Either a question or observation, such as:
  - "This feature seems interpreted, ML can't work with that right? Should we not include it?"
  - "This feature is hardcoded and it's not the time to fix it, would you agree?"

Example Message:
```
CONTEXT
---
Creative Density    avgDensity    "- Meaning: On average, there are ~24 creative elements visible per second
- What it measures: The average 'busy-ness' of the video"

INSTRUCTION
---
Keeping in mind our MLMVP2.md goals. Help me understand this Feature better and its impacts to our ML Model.
1. Is avgDensity important given our current model?
```

#### Step 2: Claude Reviews Documentation
Claude will:
1. Check the relevant MLA.md file (e.g., creative_densityMLA.md)
2. Review these specific columns:
   - Data Type
   - RF Adaptable & Transformation
   - KM Adaptable & Transformation  
   - Difficulty levels
   - Information Loss
   - Confidence scores
3. Review our ML goals found in MLMVP2.md, to answer according to that document

#### Step 2.2: Joint Decision
Claude provides:
- Data type and ML compatibility facts
- Transformation requirements
- Recommendation based on benchmark criteria
- Jorge and Claude decide together: KEEP or REMOVE

#### Step 2.3: If KEEP - Add to Shortlist
Add feature to **FeaturesMLMVP.md** with:
- Reason: Brief justification
- Compatibility notes

#### Step 2.4: Document Feature Decision
**For FeaturesMLMVP.md** (if compatible with at least one algorithm):
- Add feature to table
- Fill all 11 columns including transformations and difficulties
- Flag "Seems Repetitive?" for within-flow redundancy

**For FutureExtraFeatures.md** (document any algorithm rejections):
- Add feature if rejected by either RF or KM
- Table structure: | Feature | Reason for Removal RF | Reason for Removal KM | Difficulty | Future Value | What Needs to Change | Notes |
- Use "N/A" in removal column if that algorithm accepts the feature
- Note: A feature can appear in BOTH files if it works for one algorithm but not the other

### Files We'll Be Working With

**Input Files** (8 total):
- creative_densityMLA.md
- emotional_journeyMLA.md
- metadata_analysisMLA.md
- person_framingMLA.md
- scene_pacingMLA.md
- speech_analysisMLA.md
- temporal_markersMLA.md
- visual_overlay_analysisMLA.md

**Output Files**:
- **FeaturesMLMVP.md** - Features selected for ML pipeline
- **FutureExtraFeatures.md** - Features not selected but potentially valuable

## Process Overview

### Phase 1: Individual Flow Review (Current)
- Review each flow's features manually, one by one
- Jorge presents feature + position
- Claude checks MLA.md documentation
- Joint decision: KEEP → FeaturesMLMVP.md or REMOVE → FutureExtraFeatures.md
- Note: Repetitive features across flows are OK in Phase 1

### Phase 2: Cross-Flow Analysis (Future)
- Analyze pre-selected features from all flows
- Map out duplicates and redundancies
- Create final canonical feature set

---

## Document Relationships

### How the Three Documents Work Together

**FeaturesMLMVP.md**:
- Contains ALL features compatible with RF and/or K-means
- A feature needs to work with AT LEAST ONE algorithm to be here
- Shows transformation requirements and difficulties for both algorithms
- Tracks within-flow repetition via "Seems Repetitive?" column

**FutureExtraFeatures.md**:
- Documents rejection reasons for any algorithm incompatibility
- Features can appear here even if they're in FeaturesMLMVP.md (if rejected by one algorithm)
- "N/A" in removal column means that algorithm accepts the feature
- Helps track what would need to change for future inclusion

**Key Decision Rules**:
1. RF=Yes, KM=Yes → Feature in FeaturesMLMVP.md only
2. RF=Yes, KM=No → Feature in BOTH files (kept for RF, rejection documented for KM)
3. RF=No, KM=Yes → Feature in BOTH files (kept for KM, rejection documented for RF)
4. RF=No, KM=No → Feature in FutureExtraFeatures.md only (completely rejected)

---

## Benchmark Criteria

### 1. Temporal Architecture Redundancy Check
**Question**: Is this feature better captured by MLMVP2's temporal windows?

**Evaluation**:
- Check if feature captures temporal patterns (arrays over time, curves, progressions)
- Compare against MLMVP2 temporal windows:
  - Hook Window (0-3s): First impression features
  - Middle Window: Shape analysis, temporal events (g1-g5), bins
  - Closing Window (last 3s): CTA and conclusion features

**Decision Rule**:
- **REMOVE** if temporal windows capture the same pattern more effectively
- **KEEP** if feature provides unique non-temporal insight

**Examples**:
- `densityCurve` → REMOVE (replaced by middle window shape features)
- `multiModalPeaks` → REMOVE (replaced by temporal_events array)
- `avgDensity` → KEEP (useful summary statistic not tied to temporal position)

---

### 2. ML Algorithm Compatibility
**Question**: Can this feature be effectively used by both Random Forest and K-means?

**Evaluation Matrix**:
| Algorithm | Requirement | Transformation |
|-----------|------------|----------------|
| Random Forest | Can handle any feature type | One-hot encoding for categoricals |
| K-means | Needs numerical features | Label encoding + scaling |

**Decision Rule**:
- **STRONG KEEP** if both algorithms can use with minimal transformation (Low difficulty)
- **CONDITIONAL KEEP** if one algorithm struggles but feature is high-value
- **REMOVE** if neither algorithm can effectively use the feature

**Compatibility Scoring**:
- Both algorithms adaptable + Low transformation = Priority 1
- One algorithm adaptable + Medium transformation = Priority 2  
- Complex transformation + High info loss = Priority 3
- Not adaptable = Remove

---

### 3. Information Density vs Complexity
**Question**: Does the feature's value justify its transformation complexity?

**Evaluation**:
- **Transformation Effort**: Low / Medium / High
- **Information Loss**: None / Low (0-20%) / Medium (20-50%) / High (>50%)
- **Feature Uniqueness**: Does it capture something no other feature does?

**Decision Rule**:
- **KEEP** if Low/Medium effort AND Low/None info loss
- **EVALUATE** if High effort BUT unique high-value information
- **REMOVE** if High effort AND High info loss OR redundant information

**Examples**:
- Simple numerical (avgDensity): Low effort, No loss → KEEP
- Complex array (densityCurve): High effort, High loss → REMOVE
- Categorical (pacingStyle): Medium effort, Low loss → KEEP

---

### 4. Feature Redundancy Within Flow
**Question**: Does this feature duplicate information from other features in the same flow?

**Repetition Types** (for categorization):
1. **Direct Derivative**: One feature calculated from another
   - Example: `densityClassification` derived from `avgDensity`
2. **Statistical Variant**: Different statistical measures of same data
   - Example: `avgDensity`, `maxDensity`, `minDensity`, `stdDeviation`
3. **High Correlation**: Features that likely move together
   - Example: `totalElements` and `elementsPerSecond`
4. **Different Representation**: Different ways to capture same pattern
   - Example: `volatility` and `densityShifts` (both capture instability)
5. **Complementary**: Related but distinct (usually marked "No" for repetitive)

**How to Flag**:
- **Main Table**: Use Yes/Maybe/No/N/A in "Seems Repetitive?" column
- **Repetition Analysis Table**: Create separate table for Yes/Maybe features with:
  - Related Features
  - Repetition Type (one of the 5 above)
  - Explanation

**Decision Rule**:
- Flag repetitive features for Phase 2 review
- Document relationships in Repetition Analysis table
- Keep all in Phase 1 (deduplication happens in Phase 2)

---

### 5. Semantic vs Raw Features
**Question**: Is this an interpreted/semantic feature or raw measurement?

**Evaluation**:
- **Raw**: Direct measurements from ML models (counts, percentages, durations)
- **Semantic**: Human interpretations (pacingStyle, cognitiveLoadCategory)

**Decision Rule**:
- **PREFER RAW** for ML training (let models discover patterns)
- **KEEP SEMANTIC** only if:
  - Captures expert knowledge not derivable from raw features
  - Provides valuable categorization for stratified analysis
  - Has proven predictive power

**Examples**:
- `totalElements` (raw count) → KEEP
- `pacingStyle` (semantic interpretation) → EVALUATE value
- `cognitiveLoadCategory` (derived from density) → LIKELY REMOVE

---

### 6. Constant or Near-Constant Features
**Question**: Does this feature vary across videos?

**Evaluation**:
- Check if feature is hardcoded or always returns same value
- Check if feature has extremely low variance across samples

**Decision Rule**:
- **REMOVE** if constant (no predictive power)
- **REMOVE** if >95% of samples have same value
- **KEEP** if meaningful variation exists

**Example**:
- `densityProgression` (hardcoded to "stable") → REMOVE

---

### 7. Transformation Standards (Aligned with MLA.md files)

**RF/KM Adaptable Values**:
- **Yes** = Fully adaptable and will be used in MVP
- **Partial** = Technically adaptable but not selected for MVP (too complex/high effort)
- **No** = Not adaptable at all (technically impossible)

**RF/KM Transformation Categories**:

When Adaptable = **Yes** or **Partial**:
- **"Already numerical"** = Numerical features that need no transformation for RF
- **"Already binary (0/1)"** = Boolean features ready to use
- **"Scale to [0,1]"** = Need normalization (typically for K-means)
- **"Log scale + normalize"** = For skewed distributions (e.g., view counts)
- **"One-hot encode (X features)"** = Categorical encoding for RF
- **"Label encode + scale"** = Categorical encoding for K-means
- **"Extract [specific metric]"** = Extraction from complex structures
- **"Cyclical encoding (sin/cos)"** = For time-based features
- **Specific descriptions** = For unique transformations

When Adaptable = **No**:
- Leave transformation blank or explain why incompatible
- Example: "Too interpretive" or "Dynamic categories"

**RF/KM Difficulty Values** (Transformation difficulty):
- **N/A** = No transformation needed (already in correct format) OR cannot be adapted
- **Low** = Simple transformation (e.g., scaling, basic encoding)
- **Medium** = Moderate complexity (e.g., extraction, multiple steps)
- **High** = Complex transformation with significant effort

Note: This aligns with the Difficulty columns in both FeaturesMLMVP.md and FutureExtraFeatures.md

**Field Population Scenarios**:

1. **Feature works for algorithm and will be used**:
   - Adaptable: Yes
   - Transformation: Specific transformation needed
   - Difficulty: Low/Medium (should be manageable for MVP)

2. **Feature could work but too complex for MVP**:
   - Adaptable: Partial
   - Transformation: Description + "(not for MVP)" or "(too complex)"
   - Difficulty: High (showing why it's not selected)

3. **Feature cannot work for algorithm**:
   - Adaptable: No
   - Transformation: [blank] or brief reason why impossible
   - Difficulty: N/A

Example: Complex nested structure
- RF Adaptable: Partial, RF Transformation: "Extract 10+ features (not for MVP)", RF Difficulty: High
- KM Adaptable: Yes, KM Transformation: "Extract count only", KM Difficulty: Low

---

## Feature Selection Table Template

For each flow, create an entry in FeaturesMLMVP.md with:

| Source | Feature | Reason | RF Adaptable | RF Transformation | RF Difficulty | KM Adaptable | KM Transformation | KM Difficulty | Seems Repetitive? | Notes |
|--------|---------|---------|--------------|-------------------|---------------|--------------|-------------------|---------------|-------------------|-------|
| flow_name | feature_name | Brief justification | Yes/Partial/No | See categories above | Low/Medium/High/N/A | Yes/Partial/No | See categories above | Low/Medium/High/N/A | Yes/No/Maybe/N/A | Additional context |

---

## Prompt Template for Other Flows

When analyzing each new flow, use this template:

```
I need to evaluate features from [FLOW_NAME] for ML pipeline compatibility.

Context:
- We have MLMVP2's temporal windows that handle temporal patterns
- We need features compatible with Random Forest and/or K-means
- Phase 1: Select features per flow (duplicates OK across flows)
- Phase 2: Cross-flow deduplication (later)
- Decision rule: If compatible with AT LEAST ONE algorithm → FeaturesMLMVP.md
- Features can appear in BOTH files if they work for one algorithm but not the other

Please:
1. Read [FLOW_NAME]MLA.md to understand current features
2. Apply the 6 benchmark criteria from FeatureBenchmark.md:
   - Temporal redundancy check
   - ML algorithm compatibility  
   - Information density vs complexity
   - Within-flow redundancy
   - Semantic vs raw features
   - Constant feature check

3. For each feature:
   - Determine RF compatibility (Yes/No) with transformation and difficulty
   - Determine KM compatibility (Yes/No) with transformation and difficulty
   - Flag "Seems Repetitive?" (Yes/No/Maybe) - WITHIN this flow only
   - If compatible with at least one → KEEP in FeaturesMLMVP.md
   - If incompatible with one/both → document rejection reasons in FutureExtraFeatures.md

4. Add to FeaturesMLMVP.md (11 columns):
   - All features compatible with RF and/or KM
   - Fill transformation details and difficulty levels
   - Flag within-flow repetition

5. Add to FutureExtraFeatures.md (7 columns):
   - Document rejection reasons for any incompatible algorithm
   - Use "N/A" for the algorithm that accepts it

Output format:
- Brief analysis per feature
- Updated entries for FeaturesMLMVP.md
- Updated entries for FutureExtraFeatures.md (if any rejections)
- Count of kept vs removed features per algorithm
```

---

## Success Metrics

- **Target**: 15-30 high-quality features per flow
- **Minimum RF compatibility**: 80% of kept features
- **Maximum complexity**: No more than 20% high-complexity transformations
- **Redundancy**: <10% redundant features within flow