# K-Means Transform Validation Checklist

**Purpose**: Track features whose K-Means transformations need validation after first training data collection

**When to Use**: After scraping the first 60 videos for any bucket, validate these features before training

**Parent Document**: [FeatureTransformation.md](./FeatureTransformation.md)

**Last Updated**: 2025-10-06

---

## Features Requiring Validation

### 1. overlay_unique_count

**Current Transform**: Log + Scale

**Assumption**: Right-skewed distribution (most videos have 0-3 overlays, some have 10+)

**Validation Method**:

```python
# Check distribution
import matplotlib.pyplot as plt
import pandas as pd

# Plot histogram
df['overlay_unique_count'].hist(bins=20)
plt.xlabel('Overlay Unique Count')
plt.ylabel('Frequency')
plt.title('Distribution of Overlay Unique Count')
plt.show()

# Get value counts
print(df['overlay_unique_count'].value_counts().sort_index())

# Calculate skewness
print(f"Skewness: {df['overlay_unique_count'].skew()}")
```

**Decision Criteria**:

✅ **Keep Log + Scale if RIGHT-SKEWED:**
- **Skewness > 1.0** (positively skewed)
- **Pattern**: Long tail to the right
- **Example distribution**:
  ```
  Value | Count | %
  ------|-------|----
  0     | 25    | 42%
  1     | 18    | 30%
  2     | 10    | 17%
  3-5   | 5     | 8%
  6-10  | 2     | 3%
  ```
- **Visual**: Most values clustered at left, long tail extending right
- **Interpretation**: Few videos use many overlays (outliers need compression)

❌ **Change to Scale [0-1] if UNIFORM/NORMAL:**
- **Skewness < 1.0** (not strongly skewed)
- **Pattern**: Relatively balanced distribution
- **Example distribution**:
  ```
  Value | Count | %
  ------|-------|----
  0-2   | 20    | 33%
  3-5   | 20    | 33%
  6-8   | 20    | 33%
  ```
- **Visual**: Values spread relatively evenly across range
- **Interpretation**: No extreme outliers, log transform unnecessary

**If Change Needed**:
1. Update [FeatureTransformation.md](./FeatureTransformation.md) line 65:
   - Change KM Transform: `Log + scale` → `Scale [0-1]`
2. Update implementation in feature transformation script
3. Re-document in this file

**Status**: ⏳ Pending first data collection

### 2. scene_count

**Current Transform**: Log + Scale

**Assumption**: Highly right-skewed distribution (most videos have 2-10 scene cuts, viral fast-cut videos have 30-50+)

### 3. shortest_scene

**Current Transform**: Log + Scale

**Assumption**: Right-skewed distribution (most videos have shortest scene 0.5-2s, fast-paced videos have flash cuts 0.1-0.3s)

### 4. longest_scene

**Current Transform**: Log + Scale

**Assumption**: Heavily right-skewed distribution (most videos have longest scene 3-10s, one-shot videos have longest scene = entire duration)

### 5. scene_duration_variance

**Current Transform**: Log + Scale

**Assumption**: Right-skewed distribution (most videos have uniform pacing variance 0.5-3.0, chaotic editing can have variance 20+)

### 6. object_count

**Current Transform**: Log + Scale

**Assumption**: Right-skewed distribution (most videos have 3-15 objects, product showcase videos have 20-50+ objects)

### 7. person_count

**Current Transform**: Log + Scale

**Assumption**: Right-skewed distribution with moderate skew (most videos have 1-3 people, crowd/event videos have 20-100+ people)

### 8. word_count

**Current Transform**: Log + Scale

**Assumption**: Right-skewed distribution (silent videos 0 words, normal talking 100-200 words, fast talkers 300-500+ words)

### 9. energy_variance

**Current Transform**: Log + Scale

**Assumption**: Right-skewed distribution (monotone content 0.01-0.1 variance, dynamic content with loud/quiet contrasts 0.5-2.0+ variance)

### 10. gesture_count

**Current Transform**: Log + Scale

**Assumption**: Right-skewed distribution (static/B-roll videos 0-5 gestures, normal talking head 10-30 gestures, expressive presenters/fitness/dance 50-100+ gestures)

### 11. gaze_variance

**Current Transform**: Log + Scale

**Assumption**: Right-skewed distribution (steady gaze/teleprompter 0.01-0.1 variance, natural gaze shifts 0.1-0.5 variance, erratic gaze 0.5-2.0+ variance)

---

## Validation Workflow

**Step 1: Collect Training Data**
- Scrape 60 videos for first bucket (e.g., 18-33s)
- Process through RumiAI pipeline
- Load `temporal_windows_updated.json` files into DataFrame

**Step 2: Aggregate Features**
- Aggregate temporal windows to video level (hook + avg_middle + closing)
- Extract features requiring validation

**Step 3: Run Validation**
- For each feature in this checklist:
  - Plot distribution histogram
  - Calculate skewness metric
  - Review decision criteria
  - Make transform decision

**Step 4: Update Documentation**
- Update FeatureTransformation.md if transforms change
- Mark feature as ✅ Validated in this document
- Note final decision and skewness value

**Step 5: Proceed to Training**
- Once all features validated → proceed with scaler fitting and model training

---

## Validation History

| Feature | Bucket Tested | Skewness | Decision | Date | Notes |
|---------|---------------|----------|----------|------|-------|
| overlay_unique_count | - | - | Pending | - | Awaiting first data collection |
| scene_count | - | - | Pending | - | Awaiting first data collection |
| shortest_scene | - | - | Pending | - | Awaiting first data collection |
| longest_scene | - | - | Pending | - | Awaiting first data collection |
| scene_duration_variance | - | - | Pending | - | Awaiting first data collection |
| object_count | - | - | Pending | - | Awaiting first data collection |
| person_count | - | - | Pending | - | Awaiting first data collection |
| word_count | - | - | Pending | - | Awaiting first data collection |
| energy_variance | - | - | Pending | - | Awaiting first data collection |
| gesture_count | - | - | Pending | - | Awaiting first data collection |
| gaze_variance | - | - | Pending | - | Awaiting first data collection |

---

## Notes

- **Validation is per-hashtag**: Different hashtags may have different distributions
- **One-time check**: Only need to validate once per feature (not per bucket)
- **Low risk**: If wrong, it's a one-line fix in FeatureTransformation.md
- **Skewness threshold**: Using 1.0 as cutoff (moderate skew)
  - Skewness > 1.0: Right-skewed (use log)
  - Skewness < 1.0: Not strongly skewed (use direct scale)
