# Semantic Ranges Update Guide - Metaprompt for Fresh CLI Instances

## 🎯 Purpose
This guide explains how to update semantic range definitions for RumiAI Stage 7 ML features across the two source-of-truth files.

---

## 📁 Files to Update (Source of Truth)

### **1. LLMOutputFix.md**
- **Location**: `/home/jorge/rumiaifinal/LLMOutputFix.md`
- **Purpose**: Final implementation decisions document (human-readable)
- **Section to update**: Search for `SEMANTIC_INTERPRETATIONS = {` (around line 2240+)

### **2. config/semantic_interpretations.py**
- **Location**: `/home/jorge/rumiaifinal/config/semantic_interpretations.py`
- **Purpose**: Actual Python code used by Stage 7 at runtime
- **Section to update**: The `SEMANTIC_INTERPRETATIONS` dictionary (starts around line 19)

**Critical Rule**: These two files MUST stay in sync. Any change to one requires the same change to the other.

---

## 📋 What Gets Updated

### **Category Structure**
Features are organized into 6 categories:

```
CATEGORY 1: Visual Composition (4 features) ✅ COMPLETE
  - average_face_size, person_count, object_count, overlay_unique_count

CATEGORY 2: Energy/Performance (4 features) ✅ COMPLETE
  - energy_max, energy_level, energy_variance, emotional_valence

CATEGORY 3: Audio/Speech (4 features) ✅ COMPLETE
  - pitch_scatter_ratio, word_count, speech_coverage, word_density_std

CATEGORY 4: Eye Contact/Gaze (3 features) ✅ COMPLETE
  - eye_contact_rate, eye_contact_consistency, gaze_variance

CATEGORY 5: Scene/Pacing (4 features) ⏳ IN PROGRESS
  - scene_count, scene_duration_variance, longest_scene, shortest_scene

CATEGORY 6: Movement/Temporal/Metadata (7 features) ⏳ PENDING
  - gesture_count, energy_progression_slope, middle_to_closing_energy,
    middle_to_closing_delta, hour, day_of_week, dominant_emotion_id
```

---

## 🔧 Update Process (Step-by-Step)

### **Step 1: Locate the Category Section**

In **both files**, find the category comment block:

**LLMOutputFix.md example:**
```python
# ========================================
# CATEGORY 5: SCENE/PACING (4 features) - TODO
# ========================================
# TODO: scene_count, scene_duration_variance, longest_scene, shortest_scene
```

**config/semantic_interpretations.py example:**
```python
# ========================================
# CATEGORY 5: SCENE/PACING (4 features) - TODO
# ========================================
# CATEGORY 5: Scene/Pacing (4 features)
#   - scene_count, scene_duration_variance, longest_scene, shortest_scene
```

### **Step 2: Replace TODO Comment with Feature Definition**

Replace the TODO line with the actual feature dictionary:

```python
'feature_name': {
    'metric_type': 'ratio|variance|count|continuous|duration',
    'direction': 'higher_is_more|lower_is_better|neutral',
    'unit': 'descriptive unit string',
    'data_range': (min, max),  # From production data analysis
    'ranges': [
        (min1, max1, 'label1', 'description1'),
        (min2, max2, 'label2', 'description2'),
        # 3-5 ranges per feature
    ],
    'notes': 'Methodology: [method used]. [Key findings]. [Production data insights].'
},
```

### **Step 3: Update Category Status**

After adding all features in a category, update the header:

**Before:**
```python
# CATEGORY 5: SCENE/PACING (4 features) - TODO
```

**After:**
```python
# CATEGORY 5: SCENE/PACING (4 features) ✅ FINALIZED
```

### **Step 4: Update Remaining Count**

Update the "REMAINING CATEGORIES" comment to reflect progress:

**Before:**
```python
# REMAINING CATEGORIES (12 features) - TODO
```

**After (when Category 5 complete):**
```python
# REMAINING CATEGORIES (8 features) - TODO
```

---

## 📝 Feature Definition Template

When adding a new feature, use this exact structure:

```python
'feature_name': {
    # REQUIRED FIELDS
    'metric_type': 'ratio',  # Options: ratio, variance, count, continuous, duration
    'direction': 'neutral',  # Options: higher_is_more, lower_is_better, neutral
    'unit': 'descriptive unit',  # What the feature measures
    'data_range': (0.0, 1.0),  # (min, max) from production data

    # RANGES (3-5 threshold-based categories)
    'ranges': [
        (0.0, 0.3, 'low', 'descriptive text'),
        (0.3, 0.6, 'medium', 'descriptive text'),
        (0.6, 1.0, 'high', 'descriptive text')
    ],

    # NOTES (methodology, production insights, calibration notes)
    'notes': 'Methodology: [Quartile-based|Semantic|Domain expertise]. [Key production findings]. [Any recalibration notes].'
},
```

---

## 🔍 Example Update (Complete)

### **Before Update:**

**LLMOutputFix.md (line 2480):**
```python
    # ========================================
    # CATEGORY 5: SCENE/PACING (4 features) - TODO
    # ========================================
    # TODO: scene_count, scene_duration_variance, longest_scene, shortest_scene
```

**config/semantic_interpretations.py (line 253):**
```python
    # ========================================
    # CATEGORY 5: SCENE/PACING (4 features) - TODO
    # ========================================
    # CATEGORY 5: Scene/Pacing (4 features)
    #   - scene_count, scene_duration_variance, longest_scene, shortest_scene
```

### **After Update:**

**Both files (replace TODO with):**
```python
    # ========================================
    # CATEGORY 5: SCENE/PACING (4 features) ✅ FINALIZED
    # ========================================

    'scene_count': {
        'metric_type': 'count',
        'direction': 'neutral',
        'unit': 'number of scene cuts',
        'data_range': (0, 15),
        'ranges': [
            (0, 2, 'no cuts', 'single continuous scene'),
            (2, 5, 'few cuts', 'minimal scene changes'),
            (5, 10, 'moderate cuts', 'regular scene changes'),
            (10, 20, 'many cuts', 'frequent scene changes')
        ],
        'notes': 'Methodology: Semantic categories. Scene detection counts from PySceneDetect. Production range: 0-15 cuts per window.'
    },

    'scene_duration_variance': {
        'metric_type': 'variance',
        'direction': 'neutral',
        'unit': 'variance in scene durations',
        'data_range': (0.0, 5.0),
        'ranges': [
            (0.0, 0.5, 'very consistent', 'uniform scene lengths'),
            (0.5, 1.5, 'consistent', 'similar scene lengths'),
            (1.5, 3.0, 'varied', 'mixed scene lengths'),
            (3.0, 10.0, 'very varied', 'highly irregular scene lengths')
        ],
        'notes': 'Methodology: Quartile-based. Measures variance in scene durations within window.'
    },

    'longest_scene': {
        'metric_type': 'duration',
        'direction': 'neutral',
        'unit': 'seconds',
        'data_range': (0.5, 10.0),
        'ranges': [
            (0.0, 1.0, 'very short', 'under 1 second'),
            (1.0, 3.0, 'short', '1-3 seconds'),
            (3.0, 6.0, 'medium', '3-6 seconds'),
            (6.0, 15.0, 'long', 'over 6 seconds')
        ],
        'notes': 'Methodology: Domain expertise (editing standards). Longest single scene duration within window.'
    },

    'shortest_scene': {
        'metric_type': 'duration',
        'direction': 'neutral',
        'unit': 'seconds',
        'data_range': (0.1, 5.0),
        'ranges': [
            (0.0, 0.5, 'flash cut', 'under 0.5 seconds'),
            (0.5, 1.5, 'quick cut', '0.5-1.5 seconds'),
            (1.5, 3.0, 'standard', '1.5-3 seconds'),
            (3.0, 10.0, 'long hold', 'over 3 seconds')
        ],
        'notes': 'Methodology: Domain expertise (editing standards). Shortest single scene duration within window.'
    },
```

---

## ✅ Validation Checklist

After updating both files, verify:

- [ ] Both files have identical feature definitions (same ranges, same notes)
- [ ] Category header updated from "TODO" to "✅ FINALIZED"
- [ ] TODO comment replaced with actual dictionary entries
- [ ] All 4 required fields present: metric_type, direction, unit, data_range
- [ ] ranges list has 3-5 entries with proper tuple format: (min, max, 'label', 'description')
- [ ] notes field documents methodology and production insights
- [ ] Remaining category count updated (if category completed)
- [ ] Python syntax valid (commas, quotes, brackets)

---

## 🚫 Common Mistakes to Avoid

1. **Updating only one file** - ALWAYS update both LLMOutputFix.md AND semantic_interpretations.py
2. **Wrong section** - Make sure you're updating the SEMANTIC_INTERPRETATIONS dictionary, not example code elsewhere
3. **Inconsistent ranges** - The two files must have identical range definitions
4. **Missing commas** - Each feature dictionary needs a trailing comma (except the last one)
5. **Wrong quotes** - Use single quotes for Python strings, consistent with existing code
6. **Forgetting to update status** - Change "TODO" to "✅ FINALIZED" when done

---

## 📍 Quick Reference: Line Numbers

**As of 2025-01-30:**

### LLMOutputFix.md
- Category 1 (Visual): Line ~2247-2305 ✅ COMPLETE
- Category 2 (Energy): Line ~2307-2367 ✅ COMPLETE
- Category 3 (Audio): Line ~2370-2429 ✅ COMPLETE
- Category 4 (Eye Contact): Line ~2432-2476 ✅ COMPLETE
- Category 5 (Scene): Line ~2478-2482 ⏳ TODO
- Category 6 (Movement): Line ~2482 ⏳ TODO

### config/semantic_interpretations.py
- Category 1 (Visual): Line ~21-79 ✅ COMPLETE
- Category 2 (Energy): Line ~81-141 ✅ COMPLETE
- Category 3 (Audio): Line ~144-203 ✅ COMPLETE
- Category 4 (Eye Contact): Line ~205-251 ✅ COMPLETE
- Category 5 (Scene): Line ~253-259 ⏳ TODO
- Category 6 (Movement): Line ~259 ⏳ TODO

**Note**: Line numbers may shift as features are added. Use section comments to locate categories.

---

## 💡 Pro Tips

1. **Read before editing**: Always use the Read tool on both files before making edits
2. **Edit one category at a time**: Complete all features in a category before moving to the next
3. **Copy-paste carefully**: When copying ranges between files, ensure formatting stays consistent
4. **Test syntax**: After editing Python file, consider basic syntax validation if possible
5. **Document methodology**: Always include "Methodology: [method]" in notes field for transparency

---

## 🎓 Summary for Fresh CLI Instance

**Your task**: Update semantic range definitions for RumiAI ML features

**The two files**:
1. `/home/jorge/rumiaifinal/LLMOutputFix.md`
2. `/home/jorge/rumiaifinal/config/semantic_interpretations.py`

**What to update**: Replace `# TODO: feature_name` comments with full feature dictionary definitions inside the `SEMANTIC_INTERPRETATIONS` dictionary

**Keep in sync**: Changes to one file MUST be mirrored exactly in the other

**When done**: Update category status from "TODO" to "✅ FINALIZED"

**Validation**: Both files must have identical ranges for each feature
