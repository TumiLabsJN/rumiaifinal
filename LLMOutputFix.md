# Stage 7 LLM Output Fix: Creator-Friendly Output

**Date**: 2025-10-28
**Status**: Fix Proposed (Not Yet Implemented)

---

## Issues Overview

Stage 7 Phase 2 outputs contain technical details that confuse content creators. Three main issues identified:

| Issue | Current Output | Creator Impact | Root Cause |
|-------|---------------|----------------|------------|
| **#1: Supplementary Insights** | Raw ML feature names + numeric values | ❌ Meaningless jargon ("hook_energy_max: 0.13") | Pre-filled data bypasses LLM transformation |
| **#2: Granular Middle Structure** | Separate middle_1, middle_2, middle_3 in `structure` | ❌ Confusing multi-part structure | LLM mirrors input granularity |
| **#3: Step-by-Step Template** | Granular middle segments + numeric values (0.77) + second markers (3-6s) | ❌ Technical jargon + confusing timings | LLM mirrors input granularity + includes ML feature values |

**Common Pattern**: LLM receives raw/granular data → outputs it as-is instead of synthesizing creator-friendly insights

**Solution**: Update prompts to instruct LLM to transform data + add validation to enforce format

---

## Issue #1: Supplementary Insights - Raw Technical Data

The `supplementary_insights` section of `winning_formulas.json` outputs raw ML feature names and numeric values that are not creator-friendly, while the `creative_reports` section successfully generates natural language insights.

### Current Output (Lines 127-140 of winning_formulas.json)

```json
"supplementary_insights": {
  "universal_principles": [
    "closing_energy_variance: 0.00 in top vs 0.00 in bottom (gap: 0.00)",
    "hook_energy_max: 0.13 in top vs 0.16 in bottom (gap: 0.02)",
    "closing_pitch_scatter_ratio: 0.74 in top vs 0.59 in bottom (gap: 0.15)",
    "hook_longest_scene: 1.61 in top vs 1.30 in bottom (gap: 0.31)",
    "hook_average_face_size: 0.06 in top vs 0.07 in bottom (gap: 0.01)"
  ],
  "cross_window_patterns": [
    "xwin_eye_contact_consistency: Decreases from 0.22 (bottom) to 0.14 (top)"
  ]
}
```

**Issues**:
- ❌ Raw ML feature names (`closing_energy_variance`, `hook_longest_scene`)
- ❌ Meaningless numeric values without context (0.74, 0.59)
- ❌ Includes non-discriminative features (gap: 0.00)
- ❌ No percentage-based insights ("applies to X% of videos")
- ❌ Not actionable for content creators

### Expected Output (per MLPlanningv2.md lines 3248-3262)

```json
"supplementary_insights": {
  "universal_principles": [
    "High eye contact rate (88% vs 45% for top vs bottom) - applies to 78% of videos",
    "Consistent energy maintenance across windows (std dev ≤0.15) - found in 65% of top performers",
    "Extended opening scenes (1.6s vs 1.3s average) - present in 72% of winning videos",
    "Dynamic vocal delivery with pitch variation (0.74 vs 0.59 scatter ratio) - used by 68% of top creators"
  ],
  "cross_window_patterns": [
    "78% of high-performing videos use 'bookend' eye contact pattern (high in hook/closing, lower in middle)",
    "Energy progression: 65% build energy gradually, 12% maintain consistent energy, 23% show variable patterns",
    "Scene pacing: 82% start with longer scenes (1.5s+) then accelerate to rapid cuts (<0.8s) in closing"
  ]
}
```

**Requirements**:
- ✅ Plain English feature descriptions
- ✅ Percentage-based insights with distribution data
- ✅ Actionable targets and thresholds
- ✅ Creator-friendly language (no ML jargon)

---

## Root Cause Analysis

### File Locations

1. **Data Generation**: `/ml_pipeline/stage7_llm_analysis/stage7_preprocessing.py`
   - `generate_universal_principles()` (lines 444-520)
   - `generate_cross_window_patterns()` (lines 535-620)

2. **Prompt Construction**: `/ml_pipeline/stage7_llm_analysis/stage7_prompts.py`
   - `build_phase2_prompt()` (lines 560-870)
   - Pre-filled JSON output schema (lines 842-843)

### The Bug

In `stage7_prompts.py` lines 842-843:

```python
# THE PROBLEM: Pre-filling supplementary_insights with raw Python-generated strings
"supplementary_insights": {
  "universal_principles": {json.dumps(universal_principles)},  # ← Raw data inserted!
  "cross_window_patterns": {json.dumps(cross_window_patterns)}  # ← Raw data inserted!
}
```

**What happens**:
1. Python functions generate raw technical strings (e.g., `"hook_energy_max: 0.13 in top vs 0.16 in bottom (gap: 0.02)"`)
2. These strings are embedded directly into the LLM output schema using `json.dumps()`
3. The LLM **copies the pre-filled data as-is** instead of generating natural language
4. Result: Raw technical data in final output

**Why creative_reports works correctly**:
- The `creative_reports` section does NOT pre-fill data
- The schema shows empty array structure: `"creative_reports": []`
- The LLM **generates natural language from scratch** based on input data
- Result: Polished, creator-friendly insights

---

## Issue #2: Granular Middle Structure Confuses Creators

### Problem Statement

The `structure` field in `creative_reports` outputs granular middle segments (middle_1, middle_2, middle_3, etc.) that confuse creators. They don't understand why the middle is split into multiple parts, and explaining the technical bucketing adds no value.

### Current Output (Bucket 33-60s example)

```json
{
  "report_id": 1,
  "structure": {
    "hook": "High-Energy Visual Hook (Cluster 1)",
    "middle_1": "Sustained Energy Middle (Cluster 2)",
    "middle_2": "Building Tension Middle (Cluster 0)",
    "middle_3": "Peak Engagement Middle (Cluster 1)",
    "closing": "Strong CTA Closer (Cluster 2)"
  }
}
```

**Issues**:
- ❌ Creators don't know why there are 5 middle segments
- ❌ middle_1, middle_2, middle_3 distinction is technical (temporal bucketing detail)
- ❌ Explaining duration of each segment adds no actionable value
- ❌ Cluttered structure makes it harder to grasp overall strategy

### Expected Output

```json
{
  "report_id": 1,
  "structure": {
    "hook": "High-Energy Visual Hook (Cluster 1)",
    "middle": "Builds tension progressively from sustained energy through peak engagement moments",
    "closing": "Strong CTA Closer (Cluster 2)"
  }
}
```

**Requirements**:
- ✅ Simple 3-part structure: hook, middle, closing
- ✅ Middle description synthesizes overall strategy (not granular details)
- ✅ Maintains data input accuracy (LLM still analyzes all middle windows)
- ✅ Creator-friendly (easy to understand progression)

### Root Cause Analysis

**File Location**: `/ml_pipeline/stage7_llm_analysis/stage7_prompts.py`
- `build_phase2_prompt()` (lines 560-870)

**The Issue**:
The LLM receives middle window data from Phase 1 and naturally mirrors the granular structure in its output. The prompt doesn't instruct the LLM to synthesize middle segments into a single field.

**Current Flow**:
1. Phase 1 generates separate analyses: `hook_analysis.json`, `middle_1_analysis.json`, `middle_2_analysis.json`, ..., `closing_analysis.json`
2. Phase 2 receives all window data
3. LLM outputs structure mirroring input granularity (middle_1, middle_2, ...)
4. Creators see confusing multi-part middle structure

**Why This Happens**:
- Phase 1 needs granular analysis for accuracy (each window has different patterns)
- Phase 2 prompt doesn't explicitly instruct LLM to aggregate middle windows
- LLM defaults to preserving input structure

### Proposed Solution: LLM Aggregation

**Strategy**: Instruct LLM to synthesize all middle windows into single "middle" field while maintaining analytical accuracy.

**Approach**:
1. Phase 1 continues to analyze each middle window separately (no changes)
2. Phase 2 receives all middle window data (no changes)
3. **Prompt instructs LLM**: Synthesize all middle segments into single "middle" description
4. Output has simple 3-part structure: hook, middle, closing

**Implementation**: Update prompt in `build_phase2_prompt()`

Add to prompt after cluster path analysis section (around line 750):

```python
prompt += """

## Structure Field Guidelines

**Context**: You received window analyses for hook, middle_1, middle_2, ..., middle_N, closing.
Each middle window has its own cluster assignment and features.

**Your Task** (✅ C6 FIX - Clarifies analysis vs output):

1. **USE all middle window data when determining cluster paths**
   - Analyze each middle window separately for pattern detection
   - Consider all middle clusters when choosing which path to feature in reports
   - Example: Path [1,2,0,2,1] includes all 3 middle windows (2,0,2)

2. **When WRITING the structure field, synthesize into ONE middle field**
   - Don't output middle_1, middle_2, middle_3 as separate keys
   - Describe the OVERALL middle strategy in a single "middle" field
   - Show progression if clusters change, or consistency if same cluster

**Synthesis Strategy** (✅ M4 FIX - Pattern detection for repetition):

1. **Same Cluster Throughout** (e.g., [1,2,2,2,1]):
   - "Maintains [cluster description] throughout middle"
   - Use: "sustains", "keeps consistent", "continues"

2. **Progressive Transition** (e.g., [1,0,1,2,2,1]):
   - "Transitions from [cluster X] through [cluster Y] to [cluster Z]"
   - Use: "builds", "progresses", "evolves"

3. **Alternating/Oscillating Pattern** (e.g., [1,0,2,0,2,1] or [1,2,1,2,1]):
   - "Alternates between [pattern A] and [pattern B]"
   - Identify the TWO dominant clusters that repeat
   - Example: Path [1,2,1,2,1] → "Alternates between Cluster 1 and Cluster 2"
   - Use: "oscillates", "cycles", "alternates"

4. **Complex/Mixed Pattern** (e.g., [1,0,0,1,2,2,1]):
   - Describe the overall narrative arc
   - Break into phases: "Starts with X, transitions through Y, builds to Z"
   - Use: "moves through", "transitions across", "evolves from... to"

**Temporal Progression Language** (✅ M3 FIX - Examples for demonstration):

Example 1 - **Same Cluster Throughout** (Path: [1,2,2,2,1]):
```
"middle": "Maintains sustained visual energy throughout (Cluster 2)"
```

Example 2 - **Progressive Transition** (Path: [0,1,1,2,2,0]):
```
"middle": "Builds from minimal static approach (Cluster 1) through sustained energy moments (Cluster 2) before transitioning back to intimate framing"
```

Example 3 - **Alternating Pattern** (Path: [1,0,2,0,2,1]):
```
"middle": "Alternates between high-energy peaks (Cluster 2) and calm intimate moments (Cluster 0), creating dynamic rhythm"
```

Example 4 - **Complex Progression** (Path: [2,0,0,1,2,2]):
```
"middle": "Transitions from sustained energy opening through extended intimate storytelling, building back to peak engagement before closing"
```

**Key Phrases to Use**:
- **Consistency**: "maintains", "sustains", "keeps", "continues"
- **Building**: "builds", "escalates", "intensifies", "progresses"
- **Transitions**: "shifts from... to", "moves through", "transitions", "evolves"
- **Alternation**: "alternates between", "oscillates", "varies between", "cycles through"

**CRITICAL**: Do NOT output separate middle_1, middle_2, middle_3 fields - creators don't need this granularity

**Output Format**:
{
  "structure": {
    "hook": "Description of hook cluster strategy",
    "middle": "Overall middle strategy (synthesized from all middle windows)",  // ← SINGLE FIELD
    "closing": "Description of closing cluster strategy"
  }
}

**Examples**:

✅ CORRECT (Single middle field):
{
  "structure": {
    "hook": "High-energy multi-person opening",
    "middle": "Builds tension through sustained energy, transitioning from intimate moments to peak engagement",
    "closing": "Strong CTA with group dynamic"
  }
}

❌ INCORRECT (Granular middle fields):
{
  "structure": {
    "hook": "High-energy multi-person opening",
    "middle_1": "Sustained energy",
    "middle_2": "Building tension",
    "middle_3": "Peak engagement",
    "closing": "Strong CTA"
  }
}

**Special Cases** (✅ C7 FIX - Explicit handling):

1. **Short videos (3-9s bucket)** - NO middle window exists:
   - **OMIT the "middle" key entirely** from structure
   - Output only: `{"hook": "...", "closing": "..."}`
   - Do NOT include `"middle": null` or `"middle": "N/A"`

2. **Medium videos (13-18s, 18-33s)** - 1-2 middle windows:
   - Include "middle" key with synthesized description
   - Example: `{"hook": "...", "middle": "Transitions from sustained energy to peak engagement", "closing": "..."}`

3. **Long videos (33-60s, 60-90s)** - 3-5 middle windows:
   - Include "middle" key showing overall progression
   - Example: `{"hook": "...", "middle": "Builds from intimate moments through sustained energy, reaching peak engagement before final visual hook", "closing": "..."}`

**IMPORTANT**: NEVER output middle_1, middle_2, middle_3, etc. regardless of how many middle windows exist in the input data

"""
```

**Why This Works**:
- ✅ **Simple implementation**: Prompt change only, no code refactoring
- ✅ **Natural synthesis**: LLM can describe temporal progression intelligently
  - Not: "Cluster 2, then Cluster 0, then Cluster 1"
  - But: "Builds tension from intimate close-ups to dynamic group scenes"
- ✅ **Maintains accuracy**: LLM still analyzes all middle windows for cluster path detection
- ✅ **Creator-friendly**: Simple 3-part structure easy to understand and implement
- ✅ **Aligns with architecture**: Same approach as Issue #1 (LLM transforms data → natural language)

**Validation (Optional)**:

Add to `validate_supplementary_insights_schema()` function:

```python
def validate_creative_report_structure(report: dict) -> tuple[bool, list[str]]:
    """
    Validate structure field has hook/middle/closing (not middle_1, middle_2).

    Returns:
        tuple[bool, list[str]]: (is_valid, error_messages)
    """
    errors = []
    structure = report.get('structure', {})

    # Check for granular middle keys (should not exist)
    granular_middle_keys = [k for k in structure.keys() if k.startswith('middle_') and k != 'middle']
    if granular_middle_keys:
        errors.append(
            f"Structure contains granular middle keys {granular_middle_keys}. "
            f"Expected single 'middle' field instead."
        )

    # Check for required keys
    if 'hook' not in structure:
        errors.append("Missing 'hook' in structure")
    if 'closing' not in structure:
        errors.append("Missing 'closing' in structure")
    # Note: 'middle' is optional for short videos (3-9s may have no middle window)

    # ✅ M6 FIX: Strict whitelist validation (enforces consistency)
    allowed_keys = {'hook', 'middle', 'closing'}
    extra_keys = set(structure.keys()) - allowed_keys

    if extra_keys:
        errors.append(
            f"Structure contains unexpected keys: {extra_keys}. "
            f"Only allowed: {allowed_keys}. "
            f"Schema consistency more valuable than flexibility for downstream processing."
        )

    # Validate all values are descriptive strings
    for key, value in structure.items():
        if not isinstance(value, str):
            errors.append(f"structure.{key} must be string, got {type(value)}")
        elif len(value) < 10:
            errors.append(
                f"structure.{key} is too short ({len(value)} chars). "
                f"Must be descriptive (minimum 10 characters)."
            )

    is_valid = len(errors) == 0
    return is_valid, errors
```

**Integration with Phase 2 Validation**:

Update retry loop in `run_phase2_synthesis()` to validate all creative reports:

```python
# After existing supplementary_insights validation
is_valid, validation_errors = validate_supplementary_insights_schema(synthesis)

# ADD: Validate creative_reports structure
for i, report in enumerate(synthesis.get('creative_reports', [])):
    structure_valid, structure_errors = validate_creative_report_structure(report)
    if not structure_valid:
        validation_errors.extend([f"Report {i+1}: {e}" for e in structure_errors])
        is_valid = False
```

---

## Issue #3: Step-by-Step Template with Granular Middle + Numeric Values

### Problem Statement

The `step_by_step_template` field in `creative_reports` outputs granular middle segments with numeric feature values and exact second markers that confuse creators.

**TWO Problems**:
1. Granular middle segments (Middle_1, Middle_2, Middle_3, Middle_4) with second markers (3-6s, 6-12s)
2. Meaningless numeric values (0.77, 0.42) from normalized ML features

### Current Output (Bucket 18-33s example)

```json
{
  "report_id": 1,
  "step_by_step_template": [
    "Hook (0-3s): Strong eye contact (0.77), prominent face presence (0.42), establish direct connection",
    "Middle_1 (3-6s): Transition to pure visual storytelling (0.00 speech), let visuals speak",
    "Middle_2 (6-12s): Continue silent visual approach (0.00 speech), minimal direct gaze (0.31)",
    "Middle_3 (12-18s): Re-engage with strong eye contact (0.72), balanced energy (0.45)",
    "Middle_4 (18-23s): Return to silent visual hook (0.00 words), peak energy (0.40)",
    "Closing (23-26s): Visual-first silent closer, minimal verbal content (0.09), indirect gaze (0.19)"
  ]
}
```

**Issues**:
- ❌ **Granular middle segments**: Middle_1, Middle_2, Middle_3, Middle_4 (6 total steps)
- ❌ **Second markers**: (0-3s), (3-6s), (6-12s) - creators don't need exact timings
- ❌ **Numeric values**: (0.77), (0.42), (0.31) - creators don't understand normalized features
- ❌ **Inconsistent with structure**: step_by_step_template has 6 steps but structure has 3 parts

**Creator Confusion**:
- "Why 6 steps? I thought the structure only had 3 parts?"
- "What does (0.77) mean? Is that good or bad?"
- "Do I need to match exact timings like 3-6s?"
- "How do I replicate 0.42 face presence?"

### Expected Output

```json
{
  "report_id": 1,
  "step_by_step_template": [
    "Hook: Establish strong eye contact and prominent face presence for direct connection",
    "Middle: Transition to pure visual storytelling, let visuals speak with minimal verbal content, then re-engage with strong eye contact and balanced energy before returning to silent visual hook",
    "Closing: Use visual-first silent closer with minimal verbal content and indirect gaze"
  ]
}
```

**Requirements**:
- ✅ Simple 3-part structure (matches `structure` field)
- ✅ No numeric values (0.77, 0.42) - use descriptive language
- ✅ No second markers (3-6s, 6-12s) - remove exact timings
- ✅ Action-oriented language ("Establish", "Transition", "Use")
- ✅ Middle synthesizes progression across all middle windows

### Root Cause Analysis

**File Location**: `/ml_pipeline/stage7_llm_analysis/stage7_prompts.py`
- `build_phase2_prompt()` (lines 560-870)

**The Issue**:
The LLM receives middle window data and naturally outputs granular steps mirroring Phase 1 analysis. Additionally, the LLM includes numeric feature values from cluster centroids because there's no instruction to remove them.

**Current Flow**:
1. Phase 1 analyzes hook, middle_1, middle_2, ..., closing separately
2. Each window has cluster centroid data (e.g., `eye_contact: 0.77`)
3. Phase 2 LLM receives all window data with numeric values
4. LLM outputs step_by_step_template mirroring granular structure + includes numbers
5. Creators see confusing 6-step template with ML jargon

**Why This Happens**:
- No prompt instruction to synthesize middle segments
- No instruction to remove numeric values from descriptions
- LLM defaults to preserving input structure and data format

### Proposed Solution: LLM Aggregation + Number Removal

**Strategy**: Instruct LLM to synthesize middle windows AND remove numeric values from step-by-step template.

**Approach**:
1. Phase 1 continues to analyze each middle window separately (no changes)
2. Phase 2 receives all middle window data including numeric values (no changes)
3. **Prompt instructs LLM**:
   - Synthesize all middle segments into single "Middle:" entry
   - Remove all numeric values (0.77, 0.42, etc.)
   - Remove second markers (3-6s, 6-12s, etc.)
   - Use action-oriented descriptive language
4. Output has simple 3-part template: Hook, Middle, Closing

**Implementation**: Update prompt in `build_phase2_prompt()`

Add to prompt after Structure Field Guidelines (around line 233):

```python
prompt += """

## Step-by-Step Template Guidelines

**IMPORTANT**: For step_by_step_template field:

1. **Synthesize middle segments** into single "Middle:" entry (same as structure field)
2. **Remove ALL numeric values** - creators don't understand normalized features like (0.77), (0.42)
3. **Remove second markers** - creators don't need exact timings like (3-6s), (6-12s)
4. **Use descriptive language** - focus on WHAT to do, not technical measurements
5. **Action-oriented verbs** - "Establish", "Transition", "Use", "Maintain"

**Output Format**:
{
  "step_by_step_template": [
    "Hook: [Action-oriented description without numbers or timings]",
    "Middle: [Synthesized progression of all middle segments without numbers]",
    "Closing: [Action-oriented description without numbers]"
  ]
}

**Examples**:

✅ CORRECT (Simple 3-part, no numbers, no timings):
{
  "step_by_step_template": [
    "Hook: Establish strong eye contact and prominent face presence for direct connection",
    "Middle: Transition to pure visual storytelling, let visuals speak with minimal verbal content, then re-engage with strong eye contact and balanced energy before returning to silent visual hook",
    "Closing: Use visual-first silent closer with minimal verbal content and indirect gaze"
  ]
}

❌ INCORRECT (Granular segments + numbers + timings):
{
  "step_by_step_template": [
    "Hook (0-3s): Strong eye contact (0.77), prominent face presence (0.42)",
    "Middle_1 (3-6s): Transition to visual storytelling (0.00 speech)",
    "Middle_2 (6-12s): Continue silent approach (0.00 speech), minimal gaze (0.31)",
    "Closing (23-26s): Visual-first closer (0.09 verbal), indirect gaze (0.19)"
  ]
}

**Guidelines for Removing Numbers**:
- NOT: "Strong eye contact (0.77)" → YES: "Maintain strong eye contact"
- NOT: "Prominent face presence (0.42)" → YES: "Use close-up framing with prominent face presence"
- NOT: "Minimal verbal content (0.09)" → YES: "Use minimal verbal content"
- NOT: "Pure visual storytelling (0.00 speech)" → YES: "Use pure visual storytelling"
- NOT: "Balanced energy (0.45)" → YES: "Maintain balanced energy"

**Guidelines for Synthesizing Middle**:
- Describe the **progression** across middle segments
- Use transition words: "then", "followed by", "before"
- Example: "Transition to visual storytelling, let visuals speak with minimal verbal content, then re-engage with strong eye contact"

**Special Cases**:
- Short videos (3-9s): May only have "Hook" + "Closing" (no middle)
- Medium videos (13-18s, 18-33s): Synthesize 1-2 middle segments into single "Middle"
- Long videos (33-60s, 60-90s): Synthesize 3-5 middle segments into single "Middle" showing progression

"""
```

**Why This Works**:
- ✅ **Simple implementation**: Prompt change only, no code refactoring
- ✅ **Removes technical jargon**: No more (0.77) or (0.42) confusing creators
- ✅ **Consistent UX**: All 3 fields (structure, step_by_step_template, temporal_progressions) use simple hook/middle/closing
- ✅ **Action-oriented**: Creators know exactly what to do
- ✅ **Maintains accuracy**: LLM still analyzes all middle windows and numeric values for cluster paths

**Validation Function**:

Add new validation function in `stage7_llm_analysis.py`:

```python
def validate_step_by_step_template(report: dict) -> tuple[bool, list[str]]:
    """
    Validate step_by_step_template has simple format without granular middle or numbers.

    Checks:
    1. No granular middle segments (Middle_1, Middle_2, etc.)
    2. No second markers (3-6s, 6-12s, etc.)
    3. No numeric feature values in parentheses (0.77), (0.42)
    4. Has required Hook and Closing entries
    5. Simple list (≤5 items for most videos)

    Returns:
        tuple[bool, list[str]]: (is_valid, error_messages)

    Source: LLMOutputFix.md - Issue #3
    """
    import re
    errors = []
    template = report.get('step_by_step_template', [])

    if not isinstance(template, list):
        errors.append(f"step_by_step_template must be list, got {type(template)}")
        return False, errors

    # Check 1: No granular middle segments (Middle_1, Middle_2, etc.)
    for i, step in enumerate(template):
        if not isinstance(step, str):
            errors.append(f"step_by_step_template[{i}] must be string, got {type(step)}")
            continue

        # Detect granular middle format
        if re.search(r'Middle_\d+', step):
            errors.append(
                f"step_by_step_template[{i}] contains granular middle segment (Middle_1, Middle_2, etc.). "
                f"Expected single 'Middle:' entry instead."
            )

        # Check 2: Detect second markers (3-6s, 6-12s, 0-3s, etc.)
        if re.search(r'\d+-\d+s', step) or re.search(r'\(\d+-\d+s\)', step):
            errors.append(
                f"step_by_step_template[{i}] contains second markers (e.g., '(3-6s)', '6-12s'). "
                f"Remove exact timings - creators don't need this granularity."
            )

        # ✅ C8 FIX: Check 3 - Detect numeric values (multiple patterns)
        # Pattern 1: Numbers in parentheses: (0.77), (0.42), (0.00 speech)
        if re.search(r'\(\d+\.\d+\s*\w*\)', step):
            errors.append(
                f"step_by_step_template[{i}] contains numeric values in parentheses (e.g., '(0.77)'). "
                f"Remove all numeric measurements."
            )

        # Pattern 2: Bare decimal numbers between 0.0-1.0 (normalized features)
        # Matches: "0.77 eye contact", "energy 0.42", but NOT "1 person" or "2 scenes"
        bare_decimal = re.search(r'\b0\.\d+\b', step)
        if bare_decimal:
            errors.append(
                f"step_by_step_template[{i}] contains bare decimal value '{bare_decimal.group()}'. "
                f"Remove all numeric measurements - use descriptive language only."
            )

        # Pattern 3: Percentage-like decimals in wrong context: "77% normalized"
        # This catches when LLM converts 0.77 → "77%" incorrectly
        if re.search(r'\d+%\s*(normalized|score|value|ratio)', step, re.IGNORECASE):
            errors.append(
                f"step_by_step_template[{i}] contains technical percentage format. "
                f"Use natural language (e.g., 'strong' not '77% score')."
            )

        # Pattern 4: Scientific/technical number formats
        if re.search(r'\d+\.\d+[a-z_]+', step):  # Matches: "0.77energy", "0.42_face"
            errors.append(
                f"step_by_step_template[{i}] contains technical number format. "
                f"Remove all numeric measurements."
            )

    # Check 4: Must have at least Hook and Closing
    step_types = [s.split(':')[0].strip() for s in template if isinstance(s, str) and ':' in s]

    if 'Hook' not in step_types:
        errors.append("step_by_step_template missing 'Hook:' entry")
    if 'Closing' not in step_types:
        errors.append("step_by_step_template missing 'Closing:' entry")
    # 'Middle' is optional for short videos (3-9s may have no middle window)

    # Check 5: Should be simple list (≤5 items for most videos)
    # Most videos should have 2-3 items (Hook, Middle, Closing or Hook, Closing)
    if len(template) > 5:
        errors.append(
            f"step_by_step_template has {len(template)} items (expected ≤5). "
            f"Synthesize middle segments into single 'Middle:' entry."
        )

    # ✅ M2 FIX: Check 6 - Minimum word count per section (prevents lazy outputs)
    for i, step in enumerate(template):
        if not isinstance(step, str) or ':' not in step:
            continue

        # Split into label and description
        parts = step.split(':', 1)
        if len(parts) != 2:
            continue

        label = parts[0].strip()
        description = parts[1].strip()
        word_count = len(description.split())

        # Minimum word counts by section
        min_words = {
            'Hook': 8,      # "Establish strong eye contact and prominent face presence for direct connection" = 11 words
            'Middle': 15,   # Longer because it synthesizes multiple segments
            'Closing': 8    # Similar to Hook
        }

        required = min_words.get(label, 5)  # Default 5 words for unknown labels

        if word_count < required:
            errors.append(
                f"step_by_step_template[{i}] '{label}' section is too short ({word_count} words). "
                f"Minimum {required} words required to prevent lazy outputs like 'Hook: Use eye contact.'"
            )

    is_valid = len(errors) == 0
    return is_valid, errors
```

**Integration with Phase 2 Validation**:

Update retry loop in `run_phase2_synthesis()` to validate step_by_step_template:

```python
# After existing structure validation
for i, report in enumerate(synthesis.get('creative_reports', [])):
    structure_valid, structure_errors = validate_creative_report_structure(report)
    if not structure_valid:
        validation_errors.extend([f"Report {i+1} structure: {e}" for e in structure_errors])
        is_valid = False

    # ADD: Validate step_by_step_template
    template_valid, template_errors = validate_step_by_step_template(report)
    if not template_valid:
        validation_errors.extend([f"Report {i+1} template: {e}" for e in template_errors])
        is_valid = False
```

---

## Combined Solution: Fix All Three Issues

All three issues share the same root cause: **LLM receives raw/granular data and outputs it as-is instead of transforming to creator-friendly format**.

**Unified Fix Strategy**:
1. Update prompts to instruct LLM on desired output format
2. Remove pre-filled data that bypasses LLM transformation (Issue #1)
3. Add validation to ensure LLM follows instructions (all issues)
4. Add retry logic for automatic recovery (all issues)

**Files to Modify**:
- `stage7_prompts.py`: Update Phase 2 prompt (all 3 issues)
- `stage7_preprocessing.py`: Update generator functions (Issue #1 only)
- `stage7_llm_analysis.py`: Add 3 validation functions (all issues)
- Test files: Update for new data structures (Issue #1 only)

**Summary by Issue**:

| Issue | Changes Required | Complexity |
|-------|-----------------|------------|
| **#1: Supplementary Insights** | Update 2 generator functions + prompt + validation | 🔴 High (function signature changes + tests) |
| **#2: Granular Middle Structure** | Add prompt guidelines + validation | 🟢 Low (prompt only) |
| **#3: Step-by-Step Template** | Add prompt guidelines + validation | 🟢 Low (prompt only) |

**Combined Impact**:
- Issues #2 and #3 are **low-complexity additions** to Issue #1 fix
- Same files modified (no additional file count)
- Minimal additional implementation time (+1 hour)

---

## Proposed Solution (Issue #1)

**Strategy**: Let the LLM generate natural language for `supplementary_insights`, just like it does for `creative_reports`.

### Phase 1: Update Data Structure (stage7_preprocessing.py)

Return **structured data** instead of pre-formatted strings so the LLM has access to all context.

#### Change 1: `generate_universal_principles()` (lines 444-520)

**Current**:
```python
def generate_universal_principles(rf_video_data: dict, top_n: int = 7) -> List[str]:
    """Returns formatted strings."""
    principles = []

    for feature_data in feature_importance[:top_n]:
        # Returns: "hook_energy_max: 0.13 in top vs 0.16 in bottom (gap: 0.02)"
        principle = f"{feature}: {top_avg:.2f} in top vs {bottom_avg:.2f} in bottom (gap: {gap:.2f})"
        principles.append(principle)

    return principles
```

**Proposed**:
```python
def generate_universal_principles(rf_video_data: dict, top_n: int = 7) -> List[dict]:
    """Returns structured data for LLM transformation."""
    principles = []

    for feature_data in feature_importance[:top_n]:
        # FILTER 1: Skip non-discriminative features (gap < 0.05)
        gap = feature_data.get('gap', 0)
        if gap < 0.05:
            logger.debug(f"Skipping low-gap feature: {feature_data['feature']} (gap={gap:.2f})")
            continue

        # FILTER 2: Skip derived features without distribution data
        top_avg = feature_data.get('top_performer_avg')
        bottom_avg = feature_data.get('bottom_performer_avg')
        if top_avg is None or bottom_avg is None:
            logger.debug(f"Skipping derived feature: {feature_data['feature']}")
            continue

        # Return structured data with all context
        principles.append({
            'feature': feature_data['feature'],
            'top_avg': top_avg,
            'bottom_avg': bottom_avg,
            'gap': gap,
            'importance': feature_data.get('importance', 0),
            'distribution': feature_data.get('distribution', {})  # NEW: Add if available from Stage 6
        })

    # Return top 5 most discriminative features only
    return principles[:5]
```

**Key Changes**:
- Return type: `List[str]` → `List[dict]`
- Filter out non-discriminative features (gap < 0.05)
- Include distribution data for percentage-based insights
- Reduce from 7 to 5 features (quality over quantity)

#### Change 2: `generate_cross_window_patterns()` (lines 535-620)

**Current**:
```python
def generate_cross_window_patterns(window_analyses: dict, rf_video_data: dict) -> List[str]:
    """Returns formatted strings."""
    patterns = []

    for feature_data in rf_video_data.get('feature_importance', []):
        if feature.startswith('xwin_'):
            # Returns: "xwin_eye_contact_consistency: Decreases from 0.22 (bottom) to 0.14 (top)"
            direction = "Increases" if top_avg > bottom_avg else "Decreases"
            pattern = f"{feature}: {direction} from {bottom_avg:.2f} (bottom) to {top_avg:.2f} (top)"
            patterns.append(pattern)

    return patterns
```

**Proposed**:
```python
def generate_cross_window_patterns(window_analyses: dict, rf_video_data: dict) -> List[dict]:
    """Returns structured cross-window progression data for LLM transformation."""
    patterns = []

    # Extract xwin features (cross-window derived features)
    for feature_data in rf_video_data.get('feature_importance', []):
        feature = feature_data['feature']

        if feature.startswith('xwin_'):
            top_avg = feature_data.get('top_performer_avg')
            bottom_avg = feature_data.get('bottom_performer_avg')
            gap = feature_data.get('gap', 0)

            # FILTER: Only include discriminative patterns (gap ≥ 0.05)
            if top_avg is not None and bottom_avg is not None and gap >= 0.05:
                patterns.append({
                    'feature': feature,
                    'top_avg': top_avg,
                    'bottom_avg': bottom_avg,
                    'gap': gap,
                    'direction': 'increase' if top_avg > bottom_avg else 'decrease',
                    'importance': feature_data.get('importance', 0),
                    'distribution': feature_data.get('distribution', {})
                })

    # Sort by gap (most discriminative first)
    patterns.sort(key=lambda x: x['gap'], reverse=True)

    # Return top 3-5 patterns
    return patterns[:5]
```

**Key Changes**:
- Return type: `List[str]` → `List[dict]`
- Filter by gap threshold (≥ 0.05)
- Sort by discriminative power (gap)
- Include direction and distribution data

**⚠️ M1 FIX - Token Usage Monitoring**:

Before deploying to production, test with largest bucket (60-90s with 6 windows) to verify token limits:

```python
# Add to run_phase2_synthesis() after prompt construction
prompt_tokens_est = len(prompt) // 4  # Rough estimate: 1 token ≈ 4 chars
logger.info(f"Phase 2 prompt estimated tokens: {prompt_tokens_est}")

if prompt_tokens_est > 100000:  # 50% of 200k context
    logger.warning(
        f"⚠️ Large prompt detected ({prompt_tokens_est} tokens). "
        f"Consider pruning distribution data if context limits approached."
    )
```

**Fallback Strategy** (if token limits hit):
1. Reduce `top_n` from 5 to 3 features
2. Omit distribution tercile data, keep only top/bottom averages
3. Log warning for investigation

**Action Required**: Monitor first production run on 60-90s bucket, revert to `List[str]` if >150k tokens consumed.

---

### Phase 2: Update LLM Prompt (stage7_prompts.py)

Provide structured data to LLM and instruct it to generate natural language.

#### Change 3: Prompt Instructions (around lines 767-793)

**Current**:
```python
prompt += f"""

## Supplementary Insights (Universal Principles + Cross-Window Patterns)

### Universal Principles (Applicable to ALL Videos)

Top {len(universal_principles)} RF features that predict success regardless of cluster path:

"""

for i, principle in enumerate(universal_principles, 1):
    prompt += f"{i}. {principle}\n"  # ← Just prints raw string

prompt += """

### Cross-Window Patterns (Temporal Progressions)

"""

if cross_window_patterns:
    for i, pattern in enumerate(cross_window_patterns, 1):
        prompt += f"{i}. {pattern}\n"  # ← Just prints raw string
else:
    prompt += "(Insufficient windows for cross-window pattern analysis)\n"
```

**Proposed**:
```python
prompt += """

## Supplementary Insights (Universal Principles + Cross-Window Patterns)

Your task is to transform raw ML features into creator-friendly insights that complement the cluster-based creative reports.

### Universal Principles (Applicable to ALL Videos)

Below are the top 5 RF features that predict success across all cluster paths. Transform each into natural language:

"""

for i, principle in enumerate(universal_principles, 1):
    prompt += f"""
**Feature {i}: {principle['feature']}**
- Top performers average: {principle['top_avg']:.2f}
- Bottom performers average: {principle['bottom_avg']:.2f}
- Gap (discriminative power): {principle['gap']:.2f}
- Feature importance: {principle['importance']:.3f}
"""
    # Include distribution data if available
    dist = principle.get('distribution', {})
    if dist:
        prompt += f"- Distribution (top): {json.dumps(dist.get('top', {}))}\n"
        prompt += f"- Distribution (bottom): {json.dumps(dist.get('bottom', {}))}\n"
    prompt += "\n"

prompt += """

**Your task for universal_principles**:

---

**STEP 1: Translate Feature Names Using This Dictionary** (✅ C1 FIX)

Use these EXACT translations (do not invent your own):

| Raw Feature Name | Plain English Translation |
|------------------|---------------------------|
| `hook_energy_max` | "High energy delivery in opening" |
| `hook_energy_variance` | "Consistent energy in opening" |
| `hook_energy_level` | "Energy intensity in opening" |
| `closing_energy_max` | "High energy delivery in closing" |
| `closing_energy_variance` | "Consistent energy in closing" |
| `hook_longest_scene` | "Extended opening scene duration" |
| `closing_pitch_scatter_ratio` | "Dynamic vocal pitch variation in closing" |
| `hook_pitch_scatter_ratio` | "Dynamic vocal pitch variation in opening" |
| `hook_average_face_size` | "Close-up framing in opening" |
| `closing_average_face_size` | "Close-up framing in closing" |
| `xwin_eye_contact_consistency` | "Consistent eye contact throughout video" |
| `xwin_middle_to_closing_energy` | "Energy build from middle to closing" |
| `scene_duration_variance` | "Varied scene pacing" |
| `person_count` | "Number of people visible on screen" |
| `word_count` | "Script length and verbosity" |
| `eye_contact_rate` | "Direct eye contact with camera" |

If a feature is not in this list, translate literally (e.g., "hook_X" → "X in opening", "closing_X" → "X in closing").

---

**STEP 2: Calculate Percentages from Distribution Data** (✅ C2 FIX)

For each feature, you have distribution terciles:
- `top_performers.high_percentage` (% of top performers in high tercile, ≥ thresholds.high)
- `bottom_performers.high_percentage` (% of bottom performers in high tercile)

**Calculation Rules**:

1. **If top_performers.high_percentage > bottom_performers.high_percentage + 10%:**
   - Use Format A (distribution focus)
   - Example: "36% of top performers achieve high pitch variation (≥0.91) vs 29% of bottom"

2. **If gap < 0.10 OR distribution data missing:**
   - Use Format B (average focus)
   - Example: "Top performers average 1.6s vs bottom 1.3s (gap: 0.31s)"

3. **If percentages similar (within 10%) but gap exists:**
   - Use Format C (hybrid)
   - Example: "Maintain energy variance ≤0.001 (typical for 68% of top performers)"

**Example Calculation**:
```
Feature: closing_pitch_scatter_ratio
top_performers.high_percentage: 0.36 (36%)
bottom_performers.high_percentage: 0.286 (29%)
thresholds.high: 0.913

Since 36% > 29% + 10% = FALSE (only 7% difference), but gap is 0.15 (>0.10):
Use Format C: "Maintain dynamic vocal pitch variation (0.74 avg) - found in 36% of top performers"
```

---

**STEP 3: Provide Actionable Targets from Thresholds** (✅ C3 FIX)

Use the `distribution.thresholds.high` value as the target:
- This represents the top 33rd percentile (high tercile boundary)
- It's the threshold top performers typically exceed

**Rules**:
- Only include actionable targets when `thresholds.high` exists AND `gap ≥ 0.10`
- Round thresholds to 2 decimal places
- Format: "Aim for [feature description] ≥[thresholds.high]"

**Example**:
```
Feature: closing_pitch_scatter_ratio
thresholds.high: 0.913
gap: 0.15 (>0.10, so include target)

Output: "Aim for pitch variation ≥0.91 in closing"
```

**If gap < 0.10**: Skip actionable target (threshold not meaningfully discriminative)

---

**STEP 4: Choose Output Format Based on Data** (✅ C4 FIX)

**Format A** (Distribution focus - when top/bottom separation clear):
```
"[Description]: [X]% of top performers achieve [threshold description] vs [Y]% of bottom"

Example: "Dynamic vocal pitch variation in closing: 36% of top performers achieve ≥0.91 vs 29% of bottom"
```

**Format B** (Average focus - when distribution unclear or gap small):
```
"[Description]: Top performers average [top_avg] vs bottom [bottom_avg] (gap: [gap])"

Example: "Extended opening scene duration: Top performers average 1.6s vs bottom 1.3s (gap: 0.31s)"
```

**Format C** (Hybrid - when distribution exists but percentages similar):
```
"[Description]: Maintain [descriptive target] ([top_avg] avg) - found in [X]% of top performers"

Example: "Consistent energy in closing: Maintain variance ≤0.001 (avg 0.001) - found in 68% of top performers"
```

**Selection Logic**:
1. If `top_high% - bottom_high% > 10%` → Use Format A
2. If `gap < 0.10` OR `distribution == null` → Use Format B
3. Otherwise → Use Format C

**✅ CRITIQUE C2 FIX - Python Calculates Format Selection** (2025-10-29):

**Issue**: Original design required LLM to perform arithmetic and choose format, creating math error risk.

**Solution**: Python performs calculation and passes format instruction to LLM.

**Implementation**:
```python
# In generate_universal_principles() - Python decides format
if dist:
    top_high_pct = dist['top_performers']['high_percentage'] * 100
    bottom_high_pct = dist['bottom_performers']['high_percentage'] * 100
    gap = feature_data.get('gap', 0)

    # Python chooses format (deterministic)
    if top_high_pct - bottom_high_pct > 10:
        format_type = 'A'
    elif gap < 0.10 or dist is None:
        format_type = 'B'
    else:
        format_type = 'C'

principles.append({
    'feature': feature_data['feature'],
    'format_type': format_type,  # ← Python's decision
    # ... other fields
})

# In build_phase2_prompt() - Tell LLM which format to use
prompt += f"**Feature {i}** (Use Format {principle['format_type']})\n"
```

**Impact**: Eliminates LLM arithmetic errors, simplifies prompt complexity.

---

**✅ L4 FIX - Handling Null Distribution**:

Some features may have `distribution: null` (derived features without distribution data).

**When distribution is null**:
- **Always use Format B** (average focus)
- Do NOT attempt to calculate percentages
- Example: `"feature_name": {"top_avg": 0.45, "bottom_avg": 0.32, "gap": 0.13, "distribution": null}`
- Output: "Feature description: Top performers average 0.45 vs bottom 0.32 (gap: 0.13)"

**Do NOT output**:
- ❌ "Found in X% of top performers" (no distribution to calculate from)
- ❌ "Aim for ≥threshold" (no threshold available)
- ❌ Format A or C (require distribution data)

---

### Cross-Window Patterns (Temporal Progressions)

Below are temporal progression patterns that show how features evolve across windows:

"""

if cross_window_patterns:
    for i, pattern in enumerate(cross_window_patterns, 1):
        prompt += f"""
**Pattern {i}: {pattern['feature']}**
- Direction: {pattern['direction']}
- Bottom performers: {pattern['bottom_avg']:.2f}
- Top performers: {pattern['top_avg']:.2f}
- Gap: {pattern['gap']:.2f}
"""
        dist = pattern.get('distribution', {})
        if dist:
            prompt += f"- Distribution: {json.dumps(dist)}\n"
        prompt += "\n"

    prompt += """

**Your task for cross_window_patterns**:
1. **Describe the pattern in percentage terms**:
   - "78% of high-performing videos show [pattern]"
   - "Energy progression: 65% build gradually, 12% maintain steady, 23% vary"

2. **Focus on temporal narrative**:
   - How does the feature evolve from hook → middle → closing?
   - What percentage of winners follow this progression?

3. **Make it actionable**:
   - Not: "xwin_eye_contact_consistency decreases from 0.22 to 0.14"
   - But: "78% of top videos use 'bookend' eye contact pattern (high in hook/closing, lower in middle)"

**Example Transformations** (✅ L2 FIX - More examples):

Example 1:
- Input: `xwin_middle_to_closing_energy`
  - Direction: increase
  - Bottom: -0.01, Top: 0.05, Gap: 0.06
- Output: "65% of winning videos build energy from middle to closing (avg +0.05 increase vs -0.01 for bottom performers)"

Example 2:
- Input: `xwin_eye_contact_consistency`
  - Direction: decrease
  - Bottom: 0.22, Top: 0.14, Gap: 0.08
- Output: "78% of top videos maintain consistent eye contact throughout (variance 0.14) vs inconsistent patterns in bottom videos (variance 0.22)"

Example 3:
- Input: `xwin_hook_to_closing_scene_variance`
  - Direction: increase
  - Bottom: 0.05, Top: 0.12, Gap: 0.07
- Output: "Scene pacing strategy: 72% of winners start with varied pacing and accelerate through closing (variance increases 0.05 → 0.12)"

**Key Pattern**: Always explain WHAT the progression means for creators, not just the numbers.

"""
else:
    prompt += "(Only 1 window in this bucket - cross-window patterns not applicable)\n"
```

**Key Changes**:
- Provide structured data with all fields visible to LLM
- Add detailed transformation instructions with examples
- Include percentage-based formatting requirements
- Show concrete before/after examples

#### Change 4: Output Schema (lines 838-850)

**Current**:
```python
"supplementary_insights": {
  "universal_principles": {json.dumps(universal_principles)},  # ← PRE-FILLED!
  "cross_window_patterns": {json.dumps(cross_window_patterns)}  # ← PRE-FILLED!
},
```

**Proposed**:
```python
"supplementary_insights": {
  "universal_principles": [
    // Generate 5 creator-friendly insights from Feature 1-5 above
    // Format: "[Description] ([top]% vs [bottom]%) - applies to [X]% of videos"
    // Example: "High energy delivery in opening (0.36 vs 0.16 average) - found in 72% of top performers"
  ],
  "cross_window_patterns": [
    // Generate 3-5 temporal progression insights from Pattern 1-5 above
    // Format: "[X]% of high-performing videos [pattern description]"
    // Example: "78% of top videos use 'bookend' eye contact (high hook/closing, lower middle)"
  ]
},
```

**Key Changes**:
- Remove `json.dumps()` pre-filling
- Show empty array structure (like `creative_reports`)
- Add inline comments with format requirements and examples
- Let LLM generate natural language from scratch

---

## Implementation Checklist

### Phase 1: Backend Changes
- [ ] Update `generate_universal_principles()` in `stage7_preprocessing.py`
  - [ ] Change return type to `List[dict]`
  - [ ] Add gap filtering (threshold: 0.05)
  - [ ] Include distribution data in output
  - [ ] Reduce from 7 to 5 features

- [ ] Update `generate_cross_window_patterns()` in `stage7_preprocessing.py`
  - [ ] Change return type to `List[dict]`
  - [ ] Add gap filtering (threshold: 0.05)
  - [ ] Sort by discriminative power
  - [ ] Include distribution data in output

### Phase 2: Prompt Changes
- [ ] Update prompt instructions in `build_phase2_prompt()` in `stage7_prompts.py`
  - [ ] Add structured data display (lines 767-793)
  - [ ] Add transformation instructions with examples
  - [ ] Add percentage-based formatting requirements

- [ ] Update output schema in `build_phase2_prompt()` in `stage7_prompts.py`
  - [ ] Remove `json.dumps()` pre-filling (lines 842-843)
  - [ ] Add empty array structure with inline comments
  - [ ] Add format examples in comments

### Phase 3: Testing
- [ ] Run Stage 7 on existing test bucket (bucket_3-9s wellness)
- [ ] Verify `supplementary_insights` contains natural language
- [ ] Verify no raw feature names in output
- [ ] Verify percentage-based insights present
- [ ] Verify actionable targets included
- [ ] Compare output quality to `creative_reports` section

### Phase 4: Validation
- [ ] Check all 5-7 universal principles are creator-friendly
- [ ] Check 3-5 cross-window patterns use percentages
- [ ] Verify no features with gap < 0.05 appear
- [ ] Verify schema consistency across all reports
- [ ] Run integration tests on multiple buckets

---

## Expected Output After Fix

```json
{
  "creative_reports": [
    // ... (already working correctly)
  ],

  "supplementary_insights": {
    "universal_principles": [
      "Extended opening scenes (1.6s vs 1.3s average) - present in 72% of winning videos",
      "Dynamic vocal delivery with pitch variation (0.74 vs 0.59 scatter ratio) - used by 68% of top creators",
      "Consistent visual composition throughout (face size variance ≤0.15) - found in 81% of high-performing videos",
      "Strategic scene duration variation (0.83s avg, higher than bottom's 0.61s) - applies to 76% of top videos",
      "Minimal text overlay in hook (0.38 avg elements vs bottom's 0.52) - preferred by 65% of successful creators"
    ],
    "cross_window_patterns": [
      "78% of high-performing videos use 'bookend' eye contact pattern (strong in hook/closing, softer in middle)",
      "Energy progression: 65% build energy gradually hook→closing, 12% maintain steady, 23% show variable patterns",
      "Scene pacing strategy: 82% start with longer establishing shots (1.5s+) then accelerate to rapid cuts (<0.8s) in closing"
    ]
  },

  "path_statistics": {
    // ... (unchanged)
  }
}
```

---

## Risks & Considerations

### Risk 1: LLM Hallucination
**Issue**: LLM might invent percentage data not present in input
**Mitigation**:
- Provide explicit distribution data in structured format
- Add validation instructions: "Only use percentages if distribution data provided"
- Post-processing validation to check percentages match input data

### Risk 2: Distribution Data Availability
**Issue**: Stage 6 might not currently provide distribution data (e.g., "70% of videos have ≥0.6")
**Mitigation**:
- Check if Stage 6 `rf_video_analysis.json` includes distribution field
- If missing, update Stage 6 to calculate distribution percentiles
- Fallback: LLM uses only averages if distribution unavailable

### Risk 3: Prompt Token Limit
**Issue**: Detailed instructions + structured data might exceed context window
**Mitigation**:
- Monitor prompt length (current: ~{prompt_tokens_est} tokens)
- Prioritize top 5 features (not 7) to reduce size
- Test with largest bucket (most windows) to validate

### Risk 4: Output Schema Validation
**Issue**: LLM might not follow exact format requirements
**Mitigation**:
- Add retry logic with schema validation (already exists in Phase 2)
- Provide clear format examples in output schema comments
- Log validation failures for debugging

---

## Testing Strategy

### Test Case 1: Bucket 3-9s (Short Videos)
- **Input**: 2 windows (hook, closing), 32 videos
- **Expected**: 5 universal principles, 1-3 cross-window patterns
- **Validation**: No raw feature names, all insights have percentages

### Test Case 2: Bucket 18-33s (Medium Videos)
- **Input**: 4 windows (hook, middle_1, middle_2, closing), 40 videos
- **Expected**: 5 universal principles, 3-5 cross-window patterns
- **Validation**: Temporal progressions mention specific window transitions

### Test Case 3: Bucket 60-90s (Long Videos)
- **Input**: 6 windows, 40 videos
- **Expected**: 5 universal principles, 5 cross-window patterns
- **Validation**: Complex progressions described clearly

### Regression Test
- **Compare**: Old output (raw data) vs new output (natural language)
- **Metric**: Creator comprehension survey or readability score
- **Target**: 100% of insights should be actionable without ML knowledge

---

## Success Criteria

### Must Have ✅
- [ ] Zero raw feature names in output (e.g., no "hook_energy_max")
- [ ] All insights include percentage-based data
- [ ] 5-7 universal principles per report
- [ ] 3-5 cross-window patterns per report (if applicable)
- [ ] Schema validation passes for all test buckets

### Should Have 🎯
- [ ] Actionable targets included (e.g., "≥0.70 scatter ratio")
- [ ] Plain English throughout (5th grade reading level)
- [ ] Consistent tone with `creative_reports` section
- [ ] No features with gap < 0.05 included

### Nice to Have 🌟
- [ ] Distribution percentiles in parentheses (e.g., "found in top 25%")
- [ ] Visual feature groupings (e.g., "Visual Composition", "Audio Delivery")
- [ ] Cross-references between universal principles and cluster reports

---

## Rollback Plan

If LLM-based approach fails validation:

### Option A: Hybrid Approach
- Keep LLM for universal principles (simpler task)
- Use Python for cross-window patterns (more complex logic)

### Option B: Template-Based Fallback
- Create lookup table for feature name translations
- Use string templates with variable substitution
- Deterministic output, less flexible

### Option C: Revert to Current
- Keep raw data output temporarily
- Add post-processing script for PDF generation stage
- Defer fix to Stage 8 (report generation)

---

## ✅ CRITIQUE C6 FIX - Production Monitoring Strategy (2025-10-29)

**Issue**: Original rollback plan mentioned options but lacked concrete triggers, metrics, or automated rollback logic.

**Solution**: Comprehensive monitoring plan with defined thresholds and automated safeguards.

### Metrics to Track

1. **Retry Rate**
   - Metric: `retry_count / total_phase2_calls`
   - Alert threshold: >10%
   - Critical threshold: >20% (auto-rollback)
   - Measurement: Per-bucket, per-client, system-wide

2. **Validation Failure Rate**
   - Metric: `validation_failures / total_phase2_calls`
   - Alert threshold: >5%
   - Critical threshold: >15%
   - Measurement: By validation rule (which checks are failing)

3. **API Cost**
   - Metric: `actual_cost / expected_cost`
   - Alert threshold: >1.5× expected
   - Critical threshold: >2× expected
   - Measurement: Per-bucket, per-client, daily aggregate

4. **Processing Time**
   - Metric: `phase2_duration_seconds`
   - Alert threshold: >60s (expected ~20-30s)
   - Critical threshold: >120s
   - Measurement: P50, P95, P99 percentiles

### Monitoring Implementation

**Logging Requirements**:
```python
# In run_phase2_synthesis()
logger.info(f"Phase 2 attempt {attempt}/{MAX_RETRIES+1}: started")
logger.info(f"Phase 2 validation: {'passed' if valid else 'failed'}")
logger.info(f"Phase 2 cost: ${cost:.3f} (retries: {retry_count})")
logger.info(f"Phase 2 duration: {duration:.1f}s")

# Metrics for monitoring system
metrics.increment('stage7.phase2.attempts', tags=['bucket', 'client'])
metrics.increment('stage7.phase2.validation_failures', tags=['rule_type'])
metrics.histogram('stage7.phase2.cost', cost, tags=['bucket'])
metrics.histogram('stage7.phase2.duration', duration, tags=['bucket'])
```

### Rollback Triggers

**Automatic Rollback** (no human intervention):
- Retry rate >20% for 10 consecutive buckets
- Average cost >2× expected for 5 consecutive clients
- P95 processing time >120s for 1 hour

**Manual Rollback** (alert sent to team):
- Retry rate 10-20% sustained for 1 hour
- Validation failures >5% by specific rule
- Cost 1.5-2× expected sustained for 1 day

### Rollback Procedure

**Step 1: Detect Issue** (automated)
```python
if retry_rate > 0.20 and consecutive_high_retry_buckets >= 10:
    logger.critical("AUTO-ROLLBACK TRIGGERED: Retry rate exceeded threshold")
    feature_flag.set('stage7_llm_transformation', False)
    alert_team("Stage 7 auto-rollback triggered - retry rate >20%")
```

**Step 2: Revert to Previous Version**
- Feature flag toggles to fallback behavior
- Option A: Use previous version of functions (keep old code path)
- Option B: Use Python-only formatting (if implemented as fallback)
- Option C: Return to raw data temporarily (emergency only)

**Step 3: Root Cause Analysis** (manual)
- Export failed prompt examples
- Check which validation rules are failing
- Review LLM output samples
- Determine if issue is prompt, data, or LLM API

**Step 4: Fix and Redeploy**
- Address root cause
- Test on 10 representative buckets
- Gradual rollout with monitoring

### Gradual Rollout Strategy

**Phase 1: Canary** (first 48 hours)
- Enable for 10% of clients (randomly selected)
- Monitor metrics closely
- Rollback if any critical threshold hit

**Phase 2: Staged** (days 3-7)
- 25% → 50% → 75% over 5 days
- Each stage: monitor for 24h before increasing

**Phase 3: Full** (day 8+)
- 100% of clients
- Continue monitoring for 2 weeks
- Document lessons learned

---

## Related Documentation

- **MLPlanningv2.md** (lines 3248-3262): Expected supplementary insights format
- **LLMAnalysisCHILDTI.md**: Stage 7 technical implementation details
- **stage7_preprocessing.py** (lines 444-620): Current generator functions
- **stage7_prompts.py** (lines 560-870): Phase 2 prompt construction
- **winning_formulas.json** (lines 127-140): Current problematic output

---

## Verification Results (2025-10-28)

### ✅ Concern #1: Breaking Change Impact - VERIFIED

**Investigation**: Searched codebase for all callers of `generate_universal_principles()` and `generate_cross_window_patterns()`

**Files Affected**:
1. `ml_pipeline/stage7_llm_analysis/stage7_prompts.py` (lines 583, 586) - **Main caller**
2. `ml_pipeline/stage7_llm_analysis/tests/test_phase2_preprocessing.py` (lines 393, 428, 484, 518) - **Unit tests**
3. `ml_pipeline/stage7_llm_analysis/tests/test_p1_edge_cases.py` (lines 144, 198) - **Edge case tests**

**Risk Assessment**: 🟡 **Medium Risk**
- Total impact: 3 files, ~8 call sites
- Coordinated update feasible (all in same module)
- Test coverage exists - can validate changes work correctly

**Action Required**:
- Update all 3 files simultaneously
- Run test suite after changes to verify no regressions

---

### ✅ Concern #2: LLM Reliability - VALID CONCERN

**Investigation**: Reviewed existing validation logic in `stage7_llm_analysis.py` lines 513-589

**Current State** (Phase 2 Validation):
```python
# Line 542: Basic JSON parsing
synthesis = json.loads(response_text)

# Lines 563-566: Field count logging (no validation)
num_reports = len(synthesis.get('creative_reports', []))
logger.info(f"  Universal principles: {len(synthesis.get('universal_principles', []))}")

# NO RETRY LOGIC
# NO SCHEMA VALIDATION
```

**Problems Identified**:
1. ❌ **No schema validation** - doesn't check if fields have correct structure
2. ❌ **No retry logic** - if LLM generates bad format, saves invalid data
3. ❌ **No format checking** - doesn't verify universal_principles are natural language vs raw data
4. ✅ Markdown fence stripping works (lines 523-535)
5. ✅ JSON parsing with error logging works (lines 540-550)

**Risk Assessment**: 🔴 **High Risk**
- LLM could generate raw feature names (e.g., "hook_energy_max: 0.13")
- Invalid data would be saved to `winning_formulas.json` without detection
- No automatic recovery mechanism

**Mitigation Required**: Add validation + retry (see Section below)

---

### ✅ Concern #3: Distribution Data Availability - CONFIRMED

**Investigation**: Examined actual Stage 6 output from production run

**File**: `/data/clients/rollo_test2/hashtags/wellness/top_contrastive/buckets/bucket_3-9s/ml_analysis/rf_video_analysis.json`

**Distribution Data Structure** (lines 14-29):
```json
"distribution": {
  "thresholds": {
    "high": 0.000837,    // Top tercile boundary (66th percentile)
    "low": 0.000394      // Bottom tercile boundary (33rd percentile)
  },
  "top_performers": {
    "high_percentage": 0.36,      // 36% of top performers in high tercile
    "medium_percentage": 0.32,    // 32% in medium tercile
    "low_percentage": 0.32        // 32% in low tercile
  },
  "bottom_performers": {
    "high_percentage": 0.286,     // 28.6% of bottom performers in high tercile
    "medium_percentage": 0.143,   // 14.3% in medium tercile
    "low_percentage": 0.571       // 57.1% in low tercile
  }
}
```

**Example Feature with Distribution**:
```json
{
  "feature": "closing_pitch_scatter_ratio",
  "importance": 0.04467,
  "top_performer_avg": 0.74,
  "bottom_performer_avg": 0.59,
  "gap": 0.146,
  "distribution": {
    "thresholds": { "high": 0.913, "low": 0.638 },
    "top_performers": { "high_percentage": 0.36, "medium_percentage": 0.32, "low_percentage": 0.32 },
    "bottom_performers": { "high_percentage": 0.286, "medium_percentage": 0.143, "low_percentage": 0.571 }
  }
}
```

**Risk Assessment**: 🟢 **Low Risk - Data Available**

**Perfect for Percentage-Based Insights**:
- Can generate: "36% of top performers maintain high pitch variation (≥0.91)"
- Can compare: "36% of top vs 29% of bottom reach high tercile"
- Can provide thresholds: "Aim for pitch scatter ratio ≥0.91"

**Note**: Some features have `distribution: null` (e.g., "hour" feature, line 83) - these are derived features we already filter out.

---

## Proposed Validation Schema

### Schema Validation Function

Add to `stage7_llm_analysis.py` after line 592:

```python
def validate_supplementary_insights_schema(synthesis: dict, rf_video_data: dict) -> tuple[bool, list[str]]:
    """
    Validate supplementary_insights section of Phase 2 synthesis.

    Checks:
    1. supplementary_insights exists and has correct structure
    2. universal_principles is list of strings (not raw data)
    3. cross_window_patterns is list of strings
    4. No raw feature names present (cross-referenced against input) ✅ C5 FIX
    5. Percentage-based insights present

    Args:
        synthesis: LLM output
        rf_video_data: Original RF video data for cross-validation ✅ C5 FIX

    Returns:
        tuple[bool, list[str]]: (is_valid, error_messages)

    Source: LLMOutputFix.md - Concern #2 Mitigation + C5 Fix
    """
    import re
    errors = []

    # Check 1: supplementary_insights exists
    if 'supplementary_insights' not in synthesis:
        errors.append("Missing 'supplementary_insights' field")
        return False, errors

    insights = synthesis['supplementary_insights']

    # Check 2: universal_principles is list
    if 'universal_principles' not in insights:
        errors.append("Missing 'universal_principles' in supplementary_insights")
    elif not isinstance(insights['universal_principles'], list):
        errors.append(f"universal_principles must be list, got {type(insights['universal_principles'])}")
    else:
        principles = insights['universal_principles']

        # Check 3: All principles are strings
        for i, principle in enumerate(principles):
            if not isinstance(principle, str):
                errors.append(f"universal_principles[{i}] must be string, got {type(principle)}")

        # ✅ C5 FIX: Check 4 - Cross-reference against input feature names
        # Extract all raw feature names from input RF data
        input_features = {f['feature'] for f in rf_video_data.get('feature_importance', [])}

        for i, principle in enumerate(principles):
            if not isinstance(principle, str):
                continue

            principle_lower = principle.lower()

            # Check if ANY input feature name appears in the output
            for feature_name in input_features:
                # Check for exact feature name or with common suffixes
                if (feature_name in principle_lower or
                    feature_name.replace('_', ' ') in principle_lower or
                    feature_name.replace('_', '-') in principle_lower):
                    errors.append(
                        f"universal_principles[{i}] contains raw feature name '{feature_name}'. "
                        f"Must be translated to plain English using the translation dictionary."
                    )
                    break  # Only report first match per principle

        # Legacy regex check for snake_case patterns (catches features not in input)
        raw_feature_pattern = re.compile(r'\b[a-z]+_[a-z_]+\b')  # Matches snake_case anywhere

        for i, principle in enumerate(principles):
            if not isinstance(principle, str):
                continue

            matches = raw_feature_pattern.findall(principle.lower())
            if matches:
                # Filter out false positives (common phrases with underscores)
                false_positives = {'data_driven', 'well_defined', 'co_authored'}
                suspicious_matches = [m for m in matches if m not in false_positives]

                if suspicious_matches:
                    errors.append(
                        f"universal_principles[{i}] contains snake_case pattern(s): {suspicious_matches}. "
                        f"This may be a raw feature name. Use plain English only."
                    )

        # ✅ CRITIQUE C3 FIX - Whitelist Validation Enhancement (2025-10-29)
        # Additional check: Enforce EXACT translations from dictionary (stricter than cross-reference)

        ALLOWED_TRANSLATIONS = {
            'hook_energy_max': 'High energy delivery in opening',
            'closing_pitch_scatter_ratio': 'Dynamic vocal pitch variation in closing',
            'hook_longest_scene': 'Extended opening scene duration',
            'xwin_eye_contact_consistency': 'Consistent eye contact throughout video',
            'xwin_middle_to_closing_energy': 'Energy build from middle to closing',
            # ... (complete translation dictionary)
        }

        # For each feature in input, verify output uses EXACT translation
        for feature_name in input_features:
            expected_translation = ALLOWED_TRANSLATIONS.get(feature_name)
            if expected_translation:
                # Check if ANY principle uses this feature
                feature_found = False
                for i, principle in enumerate(principles):
                    if not isinstance(principle, str):
                        continue

                    if expected_translation.lower() in principle.lower():
                        feature_found = True
                        break

                    # If raw feature appears instead of translation, error
                    if feature_name.lower() in principle.lower():
                        errors.append(
                            f"universal_principles[{i}] contains raw feature '{feature_name}' "
                            f"instead of exact translation '{expected_translation}'. "
                            f"Must use exact phrases from translation dictionary."
                        )

        # This whitelist approach is STRICTER than the cross-reference check above
        # Cross-reference catches raw names, whitelist ensures EXACT translations used

        # Check 5: At least some percentage mentions
        percentage_count = sum(1 for p in principles if isinstance(p, str) and ('%' in p or 'percentage' in p.lower()))
        if len(principles) > 0 and percentage_count == 0:
            errors.append(
                f"universal_principles has {len(principles)} items but zero percentage-based insights - "
                f"expected at least some percentage data"
            )

        # ✅ M5 FIX: Check plausibility of percentages (catches hallucinations)
        for i, principle in enumerate(principles):
            if not isinstance(principle, str):
                continue

            # Extract all percentage values
            percentages = re.findall(r'(\d+(?:\.\d+)?)%', principle)

            for pct_str in percentages:
                pct = float(pct_str)

                # Check 1: Must be in valid range [0, 100]
                if pct < 0 or pct > 100:
                    errors.append(
                        f"universal_principles[{i}] contains invalid percentage: {pct}%. "
                        f"Must be between 0-100%."
                    )

                # Check 2: Suspiciously round numbers (multiples of 5 or 10) - likely hallucinated
                # Exception: 0%, 100% are plausible edge cases
                if pct not in [0, 100] and pct % 10 == 0 and pct != 50:
                    errors.append(
                        f"universal_principles[{i}] contains suspiciously round percentage: {pct}%. "
                        f"Distribution data should yield more precise values (e.g., 36%, 29%, not 30%, 40%). "
                        f"Verify against input distribution data."
                    )

                # Check 3: Very low percentages (<5%) are suspicious for tercile distributions
                # Terciles should yield ~33% per bucket, so <5% is implausible
                if 0 < pct < 5:
                    errors.append(
                        f"universal_principles[{i}] contains implausibly low percentage: {pct}%. "
                        f"Tercile distributions typically yield 10-40% per bucket."
                    )

        # Check 6: Minimum descriptive quality (prevents lazy outputs)
        for i, principle in enumerate(principles):
            if not isinstance(principle, str):
                continue

            if len(principle) < 30:
                errors.append(
                    f"universal_principles[{i}] is too short ({len(principle)} chars). "
                    f"Must be descriptive (minimum 30 characters)."
                )

    # Check 7: cross_window_patterns is list
    if 'cross_window_patterns' not in insights:
        errors.append("Missing 'cross_window_patterns' in supplementary_insights")
    elif not isinstance(insights['cross_window_patterns'], list):
        errors.append(f"cross_window_patterns must be list, got {type(insights['cross_window_patterns'])}")
    else:
        patterns = insights['cross_window_patterns']

        # Check 8: All patterns are strings
        for i, pattern in enumerate(patterns):
            if not isinstance(pattern, str):
                errors.append(f"cross_window_patterns[{i}] must be string, got {type(pattern)}")

        # ✅ C5 FIX: Check 9 - Cross-reference patterns against input features
        for i, pattern in enumerate(patterns):
            if not isinstance(pattern, str):
                continue

            pattern_lower = pattern.lower()

            # Check against input features
            for feature_name in input_features:
                if (feature_name in pattern_lower or
                    feature_name.replace('_', ' ') in pattern_lower):
                    errors.append(
                        f"cross_window_patterns[{i}] contains raw feature name '{feature_name}'. "
                        f"Must be translated to plain English."
                    )
                    break

            # Snake_case pattern check
            matches = raw_feature_pattern.findall(pattern_lower)
            if matches:
                false_positives = {'data_driven', 'well_defined'}
                suspicious_matches = [m for m in matches if m not in false_positives]
                if suspicious_matches:
                    errors.append(
                        f"cross_window_patterns[{i}] contains snake_case pattern(s): {suspicious_matches}"
                    )

    is_valid = len(errors) == 0
    return is_valid, errors
```

### Retry Logic Integration

Update `run_phase2_synthesis()` in `stage7_llm_analysis.py` after line 550:

```python
# After JSON parsing (line 542)
synthesis = json.loads(response_text)
logger.info(f"✓ JSON parsed successfully")

# ADD VALIDATION WITH RETRY
MAX_RETRIES = 2
retry_count = 0

while retry_count <= MAX_RETRIES:
    # Validate synthesis structure (✅ C5 FIX: Pass rf_video_data for cross-reference)
    is_valid, validation_errors = validate_supplementary_insights_schema(synthesis, rf_video_data)

    if is_valid:
        logger.info("✓ Supplementary insights validation passed")
        break
    else:
        logger.warning(f"Supplementary insights validation failed (attempt {retry_count + 1}/{MAX_RETRIES + 1}):")
        for error in validation_errors:
            logger.warning(f"  - {error}")

        if retry_count >= MAX_RETRIES:
            # Final attempt failed - raise error with details
            error_summary = "\n".join(f"  - {e}" for e in validation_errors)
            raise ValueError(
                f"Phase 2 synthesis validation failed after {MAX_RETRIES + 1} attempts:\n{error_summary}\n\n"
                f"LLM generated invalid supplementary_insights format. "
                f"Check prompt instructions in stage7_prompts.py"
            )

        # Retry: make new API call
        logger.info(f"Retrying Phase 2 API call (attempt {retry_count + 2}/{MAX_RETRIES + 1})...")
        retry_count += 1

        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=PHASE2_MAX_TOKENS,
            temperature=PHASE2_TEMPERATURE,
            timeout=PHASE2_TIMEOUT_SECONDS,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text
        # Strip markdown fences (reuse existing logic from lines 523-535)
        # ... (fence stripping code)

        synthesis = json.loads(response_text)

# Continue with existing code (add metadata, save, etc.)
```

**Retry Strategy**:
- **Max 2 retries** (3 total attempts)
- **No exponential backoff** (Phase 2 is deterministic prompt, not rate-limit issue)
- **Same prompt** (LLM should be consistent with clear instructions)
- **Fail-fast on 3rd attempt** - human intervention needed if LLM can't follow format

**⚠️ L1 FIX - Cost Implications** (Documentation):

Retry logic may result in up to 3× API calls if validation fails:

**Estimated Costs per Bucket** (Claude Sonnet 4: $10/M input, $75/M output):
- Input tokens: ~50k per call (prompt + data)
- Output tokens: ~2k per call (synthesis JSON)
- Cost per call: ~$0.65 ($0.50 input + $0.15 output)

**Retry Cost Scenarios**:
- Success on 1st attempt: $0.65 (expected case)
- Success on 2nd attempt: $1.30 (validation failed once)
- Success on 3rd attempt: $1.95 (validation failed twice)
- Failure after 3 attempts: $1.95 + manual intervention

**Expected Impact**: With robust prompts (C1-C8 fixes), retry rate should be <5%. Expected cost increase: ~$0.03 per bucket.

**Monitoring**: Check logs for retry frequency. If >10% of buckets require retries, investigate prompt clarity.

---

## ✅ Critical & Medium Fixes Applied (2025-10-28)

### Critical Fixes (C1-C8)

All 8 critical issues from structured critique have been integrated into the document:

| Fix ID | Issue | Solution | Status |
|--------|-------|----------|--------|
| **C1** | LLM Feature Name Hallucination | Added 16-entry translation dictionary to Issue #1 prompt | ✅ Applied |
| **C2** | Percentage Calculation Missing | Added step-by-step calculation rules with worked example | ✅ Applied |
| **C3** | Threshold Source Ambiguity | Specified `thresholds.high` as target source with conditions | ✅ Applied |
| **C4** | Contradictory Output Format | Provided 3 conditional formats (A/B/C) with selection logic | ✅ Applied |
| **C5** | Weak Validation - Raw Names | Added cross-reference against input features + snake_case detection | ✅ Applied |
| **C6** | Contradictory Structure Instructions | Clarified "use for analysis" vs "synthesize for output" | ✅ Applied |
| **C7** | No Guidance on Short Videos | Explicit instruction to OMIT middle key (not null/N/A) | ✅ Applied |
| **C8** | Numeric Value Detection Holes | Added 4 regex patterns (parentheses, bare decimals, percentages, technical) | ✅ Applied |

### Implementation Impact

**Before Critique**:
- Ambiguous prompt instructions → LLM hallucination risk
- Weak validation → Raw feature names could pass undetected
- Missing guidance → Inconsistent outputs across buckets

**After Critique Fixes**:
- ✅ Zero ambiguity: Step-by-step instructions with worked examples
- ✅ Robust validation: Cross-references input data, catches all numeric patterns
- ✅ Complete coverage: Handles short/medium/long videos explicitly
- ✅ Fail-safe mechanisms: Multiple regex patterns catch edge cases

**Risk Reduction**:
- LLM hallucination risk: 🔴 High → 🟢 Low
- Validation gaps: 🔴 High → 🟢 Low
- Output inconsistency: 🟡 Medium → 🟢 Low

---

### Medium Fixes (M1-M6)

All 6 medium-priority robustness improvements have been integrated:

| Fix ID | Issue | Solution | Status |
|--------|-------|----------|--------|
| **M1** | Token Explosion from Dict Data | Added token monitoring + fallback strategy (prune if >100k) | ✅ Applied |
| **M2** | Weak Step Template Validation | Added minimum word counts (Hook: 8, Middle: 15, Closing: 8) | ✅ Applied |
| **M3** | Ambiguous Progression Language | Added 4 examples (same/progressive/alternating/complex patterns) | ✅ Applied |
| **M4** | No Guidance on Cluster Repetition | Added explicit pattern detection rules for alternating clusters | ✅ Applied |
| **M5** | Validation Doesn't Check Percentage Accuracy | Added plausibility bounds (0-100%, flag round numbers, flag <5%) | ✅ Applied |
| **M6** | Structure Validation Whitelist | Kept strict {hook, middle, closing} whitelist for consistency | ✅ Applied |

**Robustness Improvements**:
- ✅ Token usage monitoring prevents context overflow
- ✅ Word count validation prevents lazy outputs
- ✅ Pattern examples guide LLM for edge cases
- ✅ Percentage validation catches hallucinations
- ✅ Strict schema ensures downstream compatibility

---

### Low-Priority Fixes (L1-L5)

Selected low-priority improvements have been applied (others skipped as optional):

| Fix ID | Issue | Solution | Status |
|--------|-------|----------|--------|
| **L1** | Missing Retry Token Cost Warning | Added cost documentation ($0.65/call, 3× max) + expected <5% retry rate | ✅ Applied (Option B) |
| **L2** | No Examples for Cross-Window Patterns | Added 3 worked examples with input/output transformations | ✅ Applied (Option A) |
| **L3** | Validation Error Messages Not Creator-Friendly | Skipped - errors are logged for debugging, not shown to creators | ⏭️ Skipped (Option C) |
| **L4** | No Handling for Empty Distribution Data | Added explicit fallback: Use Format B when distribution is null | ✅ Applied (Option A) |
| **L5** | Missing Transition Word List | Skipped - M3/M4 examples already demonstrate pattern sufficiently | ⏭️ Skipped (Option B) |

**Quality-of-Life Improvements**:
- ✅ Cost transparency for monitoring (L1)
- ✅ Complete guidance for cross-window features (L2, L4)
- ⏭️ Developer-facing errors remain technical (L3)
- ⏭️ Transition words implicit in examples (L5)

---

## Appendix: Feature Name Translation Examples

For future reference when implementing translation logic:

| Raw Feature Name | Plain English Translation |
|------------------|---------------------------|
| `hook_energy_max` | "High energy delivery in opening" |
| `closing_energy_variance` | "Consistent energy in closing" |
| `hook_longest_scene` | "Extended opening scene duration" |
| `closing_pitch_scatter_ratio` | "Dynamic vocal variety in closing" |
| `hook_average_face_size` | "Close-up framing in opening" |
| `xwin_eye_contact_consistency` | "Consistent eye contact throughout video" |
| `xwin_middle_to_closing_energy` | "Energy build from middle to closing" |
| `scene_duration_variance` | "Varied scene pacing" |
| `person_count` | "Number of people on screen" |
| `word_count` | "Script length / verbosity" |

---

## Decision Summary

### ✅ Verification Complete (2025-10-28)

All 3 concerns have been investigated:

| Concern | Status | Risk | Mitigation |
|---------|--------|------|------------|
| **#1 Breaking Change** | ✅ Verified | 🟡 Medium | 3 files affected, coordinated update |
| **#2 LLM Reliability** | ✅ Valid | 🔴 High | Add schema validation + retry logic |
| **#3 Distribution Data** | ✅ Confirmed | 🟢 Low | Data exists in Stage 6 output |

### 🎯 Recommended Approach: Option 1 with Validation + All Fixes

**IMPORTANT**: All critical (C1-C8), medium (M1-M6), and selected low-priority (L1, L2, L4) fixes from structured critique have been integrated. See "✅ Critical & Medium & Low Fixes Applied" section for details.

**Total Improvements**: 17 fixes applied
- Critical (C): 8/8 ✅ (100%)
- Medium (M): 6/6 ✅ (100%)
- Low (L): 3/5 ✅ (60% - others skipped as optional)

**Implementation Strategy**:

**Issue #1 (Supplementary Insights)**:
1. Update `generate_universal_principles()` → return `List[dict]` (structured data)
2. Update `generate_cross_window_patterns()` → return `List[dict]` (structured data)
3. Update `build_phase2_prompt()` → pass structured data + transformation instructions (Issue #1)
4. Add `validate_supplementary_insights_schema()` → detect raw data vs natural language
5. Update 3 test files for new return types

**Issue #2 (Granular Middle Structure)**:
6. Update `build_phase2_prompt()` → add structure field guidelines
7. Add `validate_creative_report_structure()` → detect middle_1, middle_2 vs single middle

**Issue #3 (Step-by-Step Template)**:
8. Update `build_phase2_prompt()` → add step_by_step_template guidelines (remove numbers + timings)
9. Add `validate_step_by_step_template()` → detect Middle_1, numbers (0.77), timings (3-6s)

**Combined Changes**:
10. Integrate all 3 validations into retry loop
11. Add retry logic to Phase 2 → 3 attempts with validation between each
12. Test on existing production data to verify natural language output

**Total Files Modified**: 4 source files + 2 test files = 6 files

**Estimated Implementation Time**: 4-5 hours (coordinated changes + testing + validation for 3 issues)

**Risk Level**: 🟡 **Medium** (with validation/retry, risk is manageable)

**Impact on Output**:
- ✅ `supplementary_insights` transforms from raw data → creator-friendly percentages
- ✅ `structure` simplifies from middle_1/middle_2/middle_3 → single "middle" synthesis
- ✅ `step_by_step_template` removes numbers (0.77), timings (3-6s), and granular middle segments
- ✅ All three changes improve creator UX without reducing analytical accuracy
- ✅ Consistent 3-part structure (hook/middle/closing) across all fields

---

---

## 🔄 STRATEGY CHANGE: Python-Only Approach for supplementary_insights (2025-10-29)

### Decision Summary

After thorough critique and discussion, **we are pivoting from LLM transformation to Python-only formatting** for `supplementary_insights`.

**Rationale**:
1. **Task is mechanical, not creative**: Formatting percentages is deterministic, unlike creative_reports which genuinely needs LLM synthesis
2. **Cost savings**: $0/year vs $6,552/year for LLM approach
3. **Reliability**: 100% deterministic vs 95% with LLM retries
4. **Simplicity**: No breaking changes, no complex validation
5. **Feature structure**: Only 26 base features (compositional pattern reduces maintenance)

### What Changes

| Component | Original LLM Plan | New Python Plan |
|-----------|------------------|-----------------|
| **universal_principles** | Python generates `List[dict]` → LLM transforms → `List[str]` | Python generates `List[str]` directly |
| **cross_window_patterns** | Python generates `List[dict]` → LLM transforms → `List[str]` | Python generates `List[str]` directly (template-based) |
| **Breaking changes** | YES (`List[str]` → `List[dict]` → `List[str]`) | NO (keep `List[str]`) |
| **Validation** | 3 complex validation functions + retry logic | Simple string format checks |
| **Files modified** | 7 files | 2-3 files |
| **Implementation time** | 8+ hours | 3 hours |
| **Monthly cost** | $546 | $0 |

---

### Implementation Strategy: Python-Only Approach

#### **Component 1: Feature Name Translation (Compositional Dictionary)**

**26 base features + 5 window prefixes = 31 total entries**

```python
# config/feature_translations.py

BASE_FEATURE_TRANSLATIONS = {
    # Energy/Performance (4)
    'energy_max': 'high energy delivery',
    'energy_level': 'energy intensity',
    'energy_variance': 'consistent energy',
    'emotional_valence': 'emotional tone',

    # Visual Composition (4)
    'average_face_size': 'close-up framing',
    'person_count': 'number of people visible',
    'object_count': 'visual elements present',
    'overlay_unique_count': 'text overlay elements',

    # Eye Contact & Gaze (3)
    'eye_contact_rate': 'direct eye contact',
    'eye_contact_consistency': 'consistent eye contact',
    'gaze_variance': 'consistent gaze direction',

    # Audio/Speech (4)
    'pitch_scatter_ratio': 'dynamic vocal pitch variation',
    'word_count': 'script length',
    'speech_coverage': 'verbal content',
    'word_density_std': 'varied pacing of verbal content',

    # Scene/Pacing (4)
    'scene_count': 'number of scene cuts',
    'scene_duration_variance': 'varied scene pacing',
    'longest_scene': 'extended scene duration',
    'shortest_scene': 'rapid scene cuts',

    # Movement (1)
    'gesture_count': 'hand gestures',

    # Temporal/Progression (3)
    'energy_progression_slope': 'energy trajectory',
    'middle_to_closing_energy': 'energy build from middle to closing',
    'middle_to_closing_delta': 'change from middle to closing',

    # Metadata (3)
    'hour': 'posting time',
    'day_of_week': 'posting day',
    'dominant_emotion_id': 'primary emotion displayed',
}

WINDOW_TRANSLATIONS = {
    'hook': 'in opening',
    'closing': 'in closing',
    'xwin': 'throughout video',
    'middle_aggregate': 'across middle segments',
}

def translate_feature_name(feature: str) -> str:
    """Compositional translation: [window] + [base_feature]"""
    # Parse window prefix and base feature
    # Combine using translations
    # Returns: "High energy delivery in opening"
```

**Maintenance**: 1 minute per new base feature (rare)

---

#### **Component 2: Value Interpretation (Full Semantic Dictionaries)** ✅ **APPROACH C SELECTED**

**Key Decision**: Define semantic interpretations for all 26 base features

**Rationale**: While percentages are useful, semantic descriptions (e.g., "close-up" vs "wide shot") are more actionable for creators than numeric values.

**Challenge**: We need to define what numeric ranges mean semantically for each feature type.

**Examples of the problem**:
- `average_face_size = 0.0446` → Is this "close-up", "medium shot", or "wide shot"?
- `eye_contact_consistency = 0.14` → Is this "consistent", "moderate", or "scattered"?
- `energy_level = 0.057` → Is this "high energy", "moderate", or "low"?

**Solution**: Create semantic interpretation dictionaries for all 26 features.

```python
# config/semantic_interpretations.py

SEMANTIC_INTERPRETATIONS = {
    'average_face_size': {
        'metric_type': 'ratio',  # 0.0-1.0 scale (% of frame)
        'direction': 'higher_is_closer',  # Higher value = closer to camera
        'unit': 'proportion of frame',
        'ranges': [
            (0.0, 0.05, 'wide shot', 'face occupies small portion of frame'),
            (0.05, 0.15, 'medium shot', 'face occupies moderate portion'),
            (0.15, 0.50, 'close-up', 'face fills significant portion of frame'),
            (0.50, 1.0, 'extreme close-up', 'face dominates entire frame')
        ]
    },

    'eye_contact_consistency': {
        'metric_type': 'variance',  # Standard deviation
        'direction': 'lower_is_better',  # Lower variance = more consistent
        'unit': 'variance',
        'ranges': [
            (0.0, 0.10, 'very consistent', 'maintains steady eye contact throughout'),
            (0.10, 0.20, 'moderately consistent', 'occasional variance in gaze'),
            (0.20, 0.40, 'inconsistent', 'significant gaze variance'),
            (0.40, 1.0, 'scattered', 'highly variable eye contact')
        ]
    },

    'energy_level': {
        'metric_type': 'continuous',  # 0.0-1.0 normalized scale
        'direction': 'higher_is_more',
        'unit': 'normalized energy',
        'ranges': [
            (0.0, 0.03, 'low energy', 'calm, subdued delivery'),
            (0.03, 0.07, 'moderate energy', 'balanced, natural delivery'),
            (0.07, 0.12, 'high energy', 'animated, dynamic delivery'),
            (0.12, 1.0, 'very high energy', 'intense, highly animated')
        ]
    },

    # ========================================
    # CATEGORY 1: VISUAL COMPOSITION (4 features) ✅ FINALIZED
    # ========================================

    'average_face_size': {
        'metric_type': 'ratio',
        'direction': 'higher_is_closer',
        'unit': 'proportion of frame occupied by face',
        'data_range': (0.034, 0.142),
        'ranges': [
            (0.0, 0.06, 'wide shot', 'face occupies <6% of frame'),
            (0.06, 0.10, 'medium shot', 'face occupies 6-10% of frame'),
            (0.10, 0.20, 'close-up', 'face occupies 10-20% of frame'),
            (0.20, 1.0, 'extreme close-up', 'face occupies >20% of frame')
        ],
        'notes': 'Methodology: Domain expertise (cinematography standards) + data range. Top performers avg 0.058 (wide shot), bottom avg 0.084 (medium shot). Thresholds based on standard shot classifications adjusted for observed data.'
    },

    'person_count': {
        'metric_type': 'count',
        'direction': 'neutral',
        'unit': 'number of people visible in frame',
        'data_range': (1.0, 5.0),
        'ranges': [
            (0, 1.5, 'solo', 'single person on screen'),
            (1.5, 2.5, 'duo', 'two people visible'),
            (2.5, 5.0, 'small group', '3-5 people visible'),
            (5.0, 100, 'large group', 'more than 5 people')
        ],
        'notes': 'Methodology: Semantic categories (culturally obvious). Top performers avg 3.6 (small group). Count-based with logical thresholds for solo/duo/group distinction.'
    },

    'object_count': {
        'metric_type': 'count',
        'direction': 'neutral',
        'unit': 'number of detected objects/props',
        'data_range': (2.28, 7.68),
        'ranges': [
            (0, 3.0, 'minimal objects', 'very few objects/props visible'),
            (3.0, 6.0, 'moderate objects', 'balanced visual elements'),
            (6.0, 10.0, 'many objects', 'rich visual environment'),
            (10.0, 100, 'cluttered', 'visually dense/busy composition')
        ],
        'notes': 'Methodology: Data range estimation. YOLO object detection counts. Top performers avg 6.24 (many objects). Thresholds approximate quartiles but could be refined.'
    },

    'overlay_unique_count': {
        'metric_type': 'count',
        'direction': 'neutral',
        'unit': 'number of unique text overlay elements',
        'data_range': (1.0, 5.08),
        'ranges': [
            (0, 0.5, 'no text', 'no text overlays present'),
            (0.5, 2.5, 'minimal text', '1-2 text elements'),
            (2.5, 4.5, 'moderate text', '3-4 text elements'),
            (4.5, 20, 'heavy text', '5+ text elements')
        ],
        'notes': 'Methodology: Data range estimation. OCR-detected text overlays. Top performers avg 2.83 (moderate text), bottom avg 5.08 (heavy text). Suggests less text may perform better.'
    },

    # ========================================
    # REMAINING CATEGORIES (22 features) - TODO
    # ========================================
    # CATEGORY 2: Energy/Performance (4 features)
    # CATEGORY 3: Audio/Speech (4 features)
    # CATEGORY 4: Eye Contact/Gaze (3 features)
    # CATEGORY 5: Scene/Pacing (4 features)
    # CATEGORY 6: Movement/Temporal/Metadata (7 features)
}

def interpret_value(feature: str, value: float) -> tuple[str, str]:
    """
    Convert numeric value to semantic label and description.

    Returns:
        tuple[str, str]: (label, description)
        Example: ('close-up', 'face fills significant portion of frame')
    """
    if feature not in SEMANTIC_INTERPRETATIONS:
        return ('unknown', f'value: {value:.2f}')

    interp = SEMANTIC_INTERPRETATIONS[feature]

    # Find matching range
    for min_val, max_val, label, description in interp['ranges']:
        if min_val <= value < max_val:
            return (label, description)

    # Edge case: value at max boundary
    if value >= interp['ranges'][-1][1]:
        return (interp['ranges'][-1][2], interp['ranges'][-1][3])

    return ('unknown', f'value: {value:.2f}')


def format_universal_principle(feature_data: dict) -> str:
    """
    Format using semantic interpretations + percentages.

    Output: "Close-up framing in opening: 72% of top performers use close-ups vs 15% use medium shots"
    """
    feature = feature_data['feature']
    base_feature = extract_base_feature(feature)  # Remove window prefix

    translated = translate_feature_name(feature)
    dist = feature_data.get('distribution')

    if dist:
        top_pct = dist['top_performers']['high_percentage'] * 100
        bottom_pct = dist['bottom_performers']['high_percentage'] * 100

        # Interpret what "high" means semantically
        threshold_high = dist['thresholds']['high']
        label_high, desc_high = interpret_value(base_feature, threshold_high)

        # Interpret top/bottom averages
        top_avg = feature_data['top_performer_avg']
        bottom_avg = feature_data['bottom_performer_avg']
        label_top, _ = interpret_value(base_feature, top_avg)
        label_bottom, _ = interpret_value(base_feature, bottom_avg)

        return (
            f"{translated}: {top_pct:.0f}% of top performers use {label_top} "
            f"vs {bottom_pct:.0f}% of bottom (avg: {label_bottom})"
        )
    else:
        # Fallback when no distribution
        top_avg = feature_data['top_performer_avg']
        bottom_avg = feature_data['bottom_performer_avg']
        label_top, _ = interpret_value(base_feature, top_avg)
        label_bottom, _ = interpret_value(base_feature, bottom_avg)

        return (
            f"{translated}: Top performers use {label_top} (avg {top_avg:.2f}) "
            f"vs bottom use {label_bottom} (avg {bottom_avg:.2f})"
        )

# Example outputs:
# "Face size in opening: 72% of top performers use wide shots vs 15% of bottom (avg: medium shot)"
# "Eye contact consistency in opening: 68% of top performers maintain very consistent contact vs 29% of bottom (avg: moderately consistent)"
# "Energy level in closing: 75% of top performers deliver high energy vs 20% of bottom (avg: moderate energy)"
```

**Why This Approach**:
- ✅ **Most descriptive**: Uses videography/creator terminology ("close-up", "wide shot")
- ✅ **Actionable**: Creators know exactly what to aim for
- ✅ **Professional**: Matches industry language
- ✅ **Comprehensive**: All 26 features get semantic labels

**Trade-offs Accepted**:
- 🟡 **8-12 hours research**: Need to define ranges for all 26 features
- 🟡 **Subjective**: Thresholds require domain expertise and judgment
- 🟡 **Maintenance**: New features need semantic definitions
- 🟡 **Validation**: Need to verify ranges match reality using production data

---

##### **Research Guide: How to Define Semantic Ranges**

**Step 1: Examine Production Data for Each Feature**

```bash
# For each feature, find actual value ranges in production data
find /home/jorge/rumiaifinal/data -name "rf_video_analysis.json" -exec jq -r \
  '.feature_importance[] | select(.feature | contains("average_face_size")) |
  "\(.feature)|\(.top_performer_avg)|\(.bottom_performer_avg)|\(.distribution.thresholds.high)|\(.distribution.thresholds.low)"' {} \;

# Output example:
# hook_average_face_size|0.058|0.084|0.065|0.045
# closing_average_face_size|0.057|0.117|0.068|0.040
```

**Step 2: Determine Metric Type**

| Type | Description | Example Features |
|------|-------------|------------------|
| **ratio** | 0.0-1.0 scale, represents proportion | average_face_size (% of frame) |
| **variance** | Standard deviation, lower = more consistent | energy_variance, gaze_variance |
| **count** | Discrete integers | word_count, person_count, scene_count |
| **continuous** | Normalized scale, higher = more | energy_level, pitch_scatter_ratio |
| **duration** | Seconds/time | longest_scene, shortest_scene |

**Step 3: Determine Direction**

| Direction | Meaning | Example |
|-----------|---------|---------|
| `higher_is_more` | Higher value = more of trait | energy_level, word_count |
| `lower_is_better` | Lower value = better (variance) | energy_variance, gaze_variance |
| `higher_is_closer` | Higher value = closer proximity | average_face_size |
| `neutral` | No clear better/worse | hour, day_of_week |

**Step 4: Define Semantic Ranges (Data-Driven)**

Use **quartile analysis** from production data:

```python
# Example: average_face_size analysis
# Production data shows:
# - Top performers: avg=0.058, range=[0.02, 0.15]
# - Bottom performers: avg=0.084, range=[0.04, 0.20]

# Define ranges based on data distribution:
# P0-P25 (0.0-0.05): wide shot (small face)
# P25-P50 (0.05-0.10): medium shot
# P50-P75 (0.10-0.20): close-up
# P75-P100 (0.20-1.0): extreme close-up
```

**Step 5: Validate with Domain Expertise**

Cross-reference with industry standards:
- **Videography**: Cinematography textbooks for shot types
- **Audio Engineering**: Standard practices for vocal variation, energy levels
- **Content Creation**: TikTok/creator best practices

---

##### **26 Features to Define (Organized by Category)**

**Category 1: Visual Composition (4 features)** [~2 hours research]

| Feature | Current Range (from data) | Semantic Labels Needed | Priority |
|---------|---------------------------|------------------------|----------|
| `average_face_size` | 0.02-0.20 | wide shot, medium shot, close-up, extreme close-up | High |
| `person_count` | 1-5 | solo, duo, small group, large group | High |
| `object_count` | 0-20 | minimal, few objects, moderate, cluttered | Medium |
| `overlay_unique_count` | 0-10 | none, minimal text, moderate, heavy text | Medium |

**Research approach**:
- average_face_size: Use cinematography shot classifications
- person_count: Straightforward (1=solo, 2=duo, 3-4=small group, 5+=large group)
- object_count: Use visual complexity research
- overlay_unique_count: Count-based (0=none, 1-2=minimal, 3-5=moderate, 6+=heavy)

---

**Category 2: Energy/Performance (4 features)** [~2 hours research]

| Feature | Current Range | Semantic Labels Needed | Priority |
|---------|---------------|------------------------|----------|
| `energy_max` | 0.03-0.15 | low, moderate, high, very high | High |
| `energy_level` | 0.02-0.12 | calm, balanced, animated, intense | High |
| `energy_variance` | 0.0-0.05 | very consistent, consistent, variable | Medium |
| `emotional_valence` | -1.0 to 1.0 | negative, neutral, positive, very positive | Medium |

**Research approach**:
- energy_max/level: Reference FEAT documentation for calibrated ranges
- energy_variance: Use standard deviation interpretation (low=consistent)
- emotional_valence: Already semantic (-1 to 1 scale, map to labels)

---

**Category 3: Audio/Speech (4 features)** [~2 hours research]

| Feature | Current Range | Semantic Labels Needed | Priority |
|---------|---------------|------------------------|----------|
| `pitch_scatter_ratio` | 0.5-0.9 | monotone, moderate variation, dynamic, highly dynamic | High |
| `word_count` | 0-60 | silent, brief, moderate, verbose | High |
| `speech_coverage` | 0.0-1.0 | silent, sparse speech, balanced, continuous speech | Medium |
| `word_density_std` | 0.0-0.5 | steady pacing, variable pacing, highly variable | Low |

**Research approach**:
- pitch_scatter_ratio: Audio engineering standards for vocal dynamics
- word_count: TikTok average (30-40 words/30sec video)
- speech_coverage: % of video with speech (0=silent, 1=talking entire time)

---

**Category 4: Eye Contact/Gaze (3 features)** [~1.5 hours research]

| Feature | Current Range | Semantic Labels Needed | Priority |
|---------|---------------|------------------------|----------|
| `eye_contact_rate` | 0.0-1.0 | no contact, occasional, frequent, constant | High |
| `eye_contact_consistency` | 0.0-0.5 | very consistent, consistent, variable, scattered | High |
| `gaze_variance` | 0.0-0.3 | steady gaze, moderate variance, wandering | Medium |

**Research approach**:
- eye_contact_rate: % of frames with eye contact (straightforward percentages)
- eye_contact_consistency: Variance measure (lower=more consistent)
- gaze_variance: Similar to consistency

---

**Category 5: Scene/Pacing (4 features)** [~2 hours research]

| Feature | Current Range | Semantic Labels Needed | Priority |
|---------|---------------|------------------------|----------|
| `scene_count` | 1-30 | static, few cuts, moderate cuts, rapid cuts | High |
| `scene_duration_variance` | 0.0-2.0 | consistent pacing, varied pacing, chaotic | Medium |
| `longest_scene` | 1.0-10.0s | quick cuts only, mixed, extended scenes | Medium |
| `shortest_scene` | 0.1-2.0s | flash cuts, brief cuts, standard cuts | Low |

**Research approach**:
- scene_count: Cuts per video (normalize by duration: cuts/second)
- Duration variance: Standard deviation of scene lengths
- longest/shortest: Absolute values in seconds

---

**Category 6: Movement/Temporal/Metadata (7 features)** [~2 hours research]

| Feature | Current Range | Semantic Labels Needed | Priority |
|---------|---------------|------------------------|----------|
| `gesture_count` | 0-50 | still, minimal gestures, moderate, highly animated | Medium |
| `energy_progression_slope` | -0.2 to 0.2 | declining energy, steady, building energy | Medium |
| `middle_to_closing_energy` | -0.1 to 0.1 | drops, maintains, builds | Medium |
| `middle_to_closing_delta` | varies | (depends on feature) | Low |
| `hour` | 0-23 | early morning, morning, afternoon, evening, night | Low |
| `day_of_week` | 0-6 | Monday, Tuesday, ... Sunday | Low |
| `dominant_emotion_id` | 0-7 | (FEAT emotion IDs) | Low |

**Research approach**:
- gesture_count: Gestures per second, reference body language research
- energy_progression_slope: Positive=building, negative=declining
- hour/day_of_week: Straightforward mapping
- dominant_emotion_id: Map to FEAT emotion labels

---

##### **Data-Driven Range Definition Process**

**For each feature, execute this workflow**:

1. **Extract value distribution**:
```bash
find data -name "rf_video_analysis.json" -exec jq -r \
  '.feature_importance[] | select(.feature=="hook_energy_level") |
  .top_performer_avg, .bottom_performer_avg' {} \; | sort -n
```

2. **Calculate quartiles**:
```python
import numpy as np
values = [0.02, 0.03, 0.05, 0.06, 0.08, 0.10, 0.12]  # From production data
q25, q50, q75 = np.percentile(values, [25, 50, 75])
# Define ranges: [min, q25), [q25, q50), [q50, q75), [q75, max]
```

3. **Apply domain labels**:
```python
# Example: energy_level
# q25=0.03, q50=0.06, q75=0.09
ranges = [
    (0.0, 0.03, 'low energy', 'calm delivery'),
    (0.03, 0.06, 'moderate energy', 'balanced delivery'),
    (0.06, 0.09, 'high energy', 'animated delivery'),
    (0.09, 1.0, 'very high energy', 'intense delivery')
]
```

4. **Validate with test data**:
```python
# Check if ranges make sense
test_values = [0.025, 0.055, 0.075, 0.11]
for val in test_values:
    label, desc = interpret_value('energy_level', val)
    print(f"{val:.3f} → {label}")
# Output should match intuition
```

5. **Document in semantic_interpretations.py**

---

##### **Template for Documenting Each Feature**

```python
'feature_name': {
    'metric_type': 'ratio|variance|count|continuous|duration',
    'direction': 'higher_is_more|lower_is_better|higher_is_closer|neutral',
    'unit': 'descriptive unit (e.g., "proportion of frame", "variance", "count")',
    'data_range': (min_observed, max_observed),  # From production data
    'ranges': [
        (min1, max1, 'semantic_label_1', 'creator-friendly description'),
        (min2, max2, 'semantic_label_2', 'creator-friendly description'),
        (min3, max3, 'semantic_label_3', 'creator-friendly description'),
        (min4, max4, 'semantic_label_4', 'creator-friendly description'),
    ],
    'notes': 'Any special considerations or context'
},
```

**Example: Complete definition**
```python
'average_face_size': {
    'metric_type': 'ratio',
    'direction': 'higher_is_closer',
    'unit': 'proportion of frame occupied by face',
    'data_range': (0.02, 0.20),  # From production analysis
    'ranges': [
        (0.0, 0.05, 'wide shot', 'face occupies <5% of frame, shows full body/environment'),
        (0.05, 0.15, 'medium shot', 'face occupies 5-15% of frame, upper body visible'),
        (0.15, 0.50, 'close-up', 'face occupies 15-50% of frame, fills significant portion'),
        (0.50, 1.0, 'extreme close-up', 'face occupies >50% of frame, dominates screen')
    ],
    'notes': 'Based on standard cinematography shot classifications. Thresholds calibrated from top performer data (avg=0.058, mostly wide/medium shots).'
},
```

---

#### **Component 3: cross_window_patterns (Template-Based)**

**Approach**: Simple template-based formatting (no LLM)

```python
def format_cross_window_pattern(pattern_data: dict) -> str:
    """
    Template-based formatting for temporal progressions.

    Deterministic, no LLM needed.
    """
    feature = translate_feature_name(pattern_data['feature'])
    direction = pattern_data['direction']
    top_avg = pattern_data['top_avg']
    bottom_avg = pattern_data['bottom_avg']

    # Template based on direction
    if direction == 'increase':
        return (
            f"{feature}: Top performers show upward trend "
            f"({bottom_avg:.2f} → {top_avg:.2f})"
        )
    elif direction == 'decrease':
        return (
            f"{feature}: Top performers maintain lower variance "
            f"({top_avg:.2f}) vs bottom ({bottom_avg:.2f}) for consistency"
        )
    else:
        return f"{feature}: Top performers average {top_avg:.2f} vs bottom {bottom_avg:.2f}"

# Example output:
# "Consistent eye contact throughout video: Top performers maintain lower variance (0.14) vs bottom (0.22) for consistency"
```

**Alternative Considered**: Use LLM just for cross_window_patterns
- **Cost**: $0.02 per call (vs $0 for templates)
- **Benefit**: Slightly more natural phrasing
- **Decision**: Templates sufficient, save the $0.02

---

### Updated Implementation Checklist (Approach C)

**Phase 1: Research & Define Semantic Ranges** [8-12 hours]
- [ ] **Category 1: Visual Composition** (4 features) [~2 hours]
  - [ ] Define ranges for `average_face_size` (wide shot, medium, close-up, extreme close-up)
  - [ ] Define ranges for `person_count` (solo, duo, small group, large group)
  - [ ] Define ranges for `object_count` (minimal, few, moderate, cluttered)
  - [ ] Define ranges for `overlay_unique_count` (none, minimal, moderate, heavy)
- [ ] **Category 2: Energy/Performance** (4 features) [~2 hours]
  - [ ] Define ranges for `energy_max` (low, moderate, high, very high)
  - [ ] Define ranges for `energy_level` (calm, balanced, animated, intense)
  - [ ] Define ranges for `energy_variance` (very consistent, consistent, variable)
  - [ ] Define ranges for `emotional_valence` (negative, neutral, positive, very positive)
- [ ] **Category 3: Audio/Speech** (4 features) [~2 hours]
  - [ ] Define ranges for `pitch_scatter_ratio` (monotone, moderate, dynamic, highly dynamic)
  - [ ] Define ranges for `word_count` (silent, brief, moderate, verbose)
  - [ ] Define ranges for `speech_coverage` (silent, sparse, balanced, continuous)
  - [ ] Define ranges for `word_density_std` (steady, variable, highly variable)
- [ ] **Category 4: Eye Contact/Gaze** (3 features) [~1.5 hours]
  - [ ] Define ranges for `eye_contact_rate` (none, occasional, frequent, constant)
  - [ ] Define ranges for `eye_contact_consistency` (very consistent, consistent, variable, scattered)
  - [ ] Define ranges for `gaze_variance` (steady, moderate, wandering)
- [ ] **Category 5: Scene/Pacing** (4 features) [~2 hours]
  - [ ] Define ranges for `scene_count` (static, few cuts, moderate, rapid)
  - [ ] Define ranges for `scene_duration_variance` (consistent, varied, chaotic)
  - [ ] Define ranges for `longest_scene` (quick cuts, mixed, extended)
  - [ ] Define ranges for `shortest_scene` (flash, brief, standard)
- [ ] **Category 6: Movement/Temporal/Metadata** (7 features) [~2 hours]
  - [ ] Define ranges for `gesture_count` (still, minimal, moderate, highly animated)
  - [ ] Define ranges for `energy_progression_slope` (declining, steady, building)
  - [ ] Define ranges for `middle_to_closing_energy` (drops, maintains, builds)
  - [ ] Define ranges for `middle_to_closing_delta` (varies by feature)
  - [ ] Define mappings for `hour` (early morning, morning, afternoon, evening, night)
  - [ ] Define mappings for `day_of_week` (Monday-Sunday)
  - [ ] Define mappings for `dominant_emotion_id` (FEAT emotion labels)

**Phase 2: Backend Implementation** [3-4 hours]
- [ ] Create `config/feature_translations.py` with 31 compositional entries
- [ ] Create `config/semantic_interpretations.py` with all 26 feature definitions
- [ ] Implement `interpret_value()` function with range lookup logic
- [ ] Implement `extract_base_feature()` helper (removes window prefix)
- [ ] Update `generate_universal_principles()` in `stage7_preprocessing.py`
  - [ ] Add gap filtering (threshold: 0.05)
  - [ ] Use compositional translation for feature names
  - [ ] Use semantic interpretation for value labels
  - [ ] Format: "X% of top use {semantic_label} vs Y% of bottom (avg: {semantic_label})"
  - [ ] Return `List[str]` (NO breaking change)
- [ ] Update `generate_cross_window_patterns()` in `stage7_preprocessing.py`
  - [ ] Use template-based formatting with semantic labels
  - [ ] Return `List[str]` (NO breaking change)

**Phase 3: Testing** [2-3 hours]
- [ ] Create test suite for semantic interpretations
  - [ ] Test boundary conditions for all 26 features
  - [ ] Test edge cases (min/max values, boundary overlap)
  - [ ] Test missing features (fallback behavior)
- [ ] Update existing test cases for new output format
  - [ ] Update `test_phase2_preprocessing.py` (2-3 test cases)
  - [ ] Verify semantic labels appear in output
  - [ ] Verify no raw numeric values in final strings
- [ ] Add integration tests with real production data
  - [ ] Validate ranges match actual data distributions
  - [ ] Check for any values falling outside defined ranges

**Phase 4: Validation & Integration** [1-2 hours]
- [ ] Validate semantic ranges against real production data
  - [ ] Run quartile analysis on all features
  - [ ] Verify semantic labels match data distribution
  - [ ] Adjust ranges if needed based on data
- [ ] Run Stage 7 on test bucket with semantic interpretations
- [ ] Manual quality review of output (creator-friendliness)
- [ ] Compare against current raw output (improvement verification)

**Total Effort**: **14-21 hours** (8-12 research + 6-9 implementation/testing)

**Files Modified**: 5 files
- `config/feature_translations.py` (NEW)
- `config/semantic_interpretations.py` (NEW)
- `stage7_preprocessing.py` (UPDATE)
- `test_phase2_preprocessing.py` (UPDATE)
- `test_semantic_interpretations.py` (NEW)

**Cost**: $0/month (vs $546/month for LLM approach)

---

### What We ARE Doing (Approach C)

1. ✅ **Python-only for supplementary_insights** (no LLM)
2. ✅ **Keeping return types** (`List[str]` stays `List[str]` - no breaking changes)
3. ✅ **Creating full semantic interpretation dictionaries** (26 features × 4-5 ranges each)
4. ✅ **Using compositional feature name translation** (31 entries: 26 base + 5 windows)
5. ✅ **Combining percentages + semantic labels** ("72% use close-ups vs 15% use medium shots")
6. ❌ **NOT modifying stage7_prompts.py** (no LLM prompt changes)
7. ❌ **NOT modifying stage7_llm_analysis.py** (no validation changes)
8. ❌ **NOT adding retry logic** (deterministic = no failures)

---

**Status**: ✅ Strategy Pivot Complete - Python-Only Approach for supplementary_insights
