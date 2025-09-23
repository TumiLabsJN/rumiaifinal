# Feature Engineering Insights: Lessons from RumiAI Implementation

## Executive Summary
Key learnings from implementing temporal and global features in video analysis, with practical guidance on feature selection, multicollinearity, and ML model considerations.

---

## 1. Global vs Temporal Features: Design Philosophy

### The Core Principle
Not all video features need temporal windowing. Forcing everything into temporal windows can actually **lose signal** rather than enhance it.

### Feature Classification Framework

#### Inherently Global Features (Legitimate Exceptions)
These features should remain global - attempting to make them temporal adds no value:

| Feature | Type | Rationale |
|---------|------|-----------|
| `duration_sec` | Technical | Video length needed for rate calculations |
| `uniqueOverlayCount` | Cross-window | Requires content comparison across entire video |
| `uniqueOverlayRatio` | Derived | Calculated from unique count |
| `aspectRatio` | Technical | Video format property |
| `resolution` | Technical | Fixed video property |
| `hashtagCount` | Metadata | Not temporally distributed |

### Why Mixed Features Work

1. **ML Algorithms Handle It Well**
   - **Random Forest**: Treats each feature independently
   - **K-means**: After scaling, all features are equal
   - **Both**: Can process global + temporal features together effectively

2. **Reflects Reality**
   - Some patterns ARE global (video quality, format)
   - Some patterns ARE temporal (pacing, emotion flow)
   - Forcing everything into windows loses important signals

3. **Implementation Principle**
   ```
   Temporal events → Windows (e.g., overlays appearing at timestamp X)
   Cross-window properties → Global (e.g., uniqueness across video)
   Technical properties → Global (e.g., resolution, codec)
   ```
   ⚠️ **Critical**: Never store both temporal and global versions of the same feature

---

## 2. The Peak Detection Problem

### Why Peaks Are Problematic Features

Peaks are **interpretive schema** - they require subjective decisions that may not align with actual patterns:

1. **Threshold Arbitrariness**: What constitutes a "peak"? (>2σ? >3σ? Top 10%?)
2. **Window Size Assumptions**: Peak over 0.5s vs 1s vs 2s - which matters?
3. **Direction Bias**: Assuming peaks matter more than valleys
4. **Context Ignorance**: A "peak" in a calm video ≠ peak in intense video

### Features to Deprecate
- `climaxTiming` - Who decides what's a climax?
- `emotionalPeaks` - Subjective threshold problem
- `accelerationPoints` - Arbitrary derivative thresholds
- `hooks` - Human-defined concept

### The Exception
**Audio energy peaks** - These have physical meaning (loudness) and industry-standard measurement (dB).

---

## 3. Temporal Data ≠ Temporal Feature

### The Key Distinction

Using temporal data to calculate a feature doesn't make the feature itself temporal.

#### Analogy
```
Temperature readings every hour = Temporal (24 values)
"Today was warmer than yesterday" = Global (1 comparison result)
```

#### In Practice
```python
# Temporal: Store per window
overlay_count_per_window = [3, 5, 2, 8, 1]  # 5 values

# Global: Derived from temporal but stored once
overlay_variance = calculate_variance(overlay_count_per_window)  # 1 value
```

The feature `overlay_variance` uses temporal data but is itself a global feature.

---

## 4. Multicollinearity: The Hidden Feature Killer

### Definition
Multicollinearity occurs when features are highly correlated - they contain the same information expressed differently.

### Real Example from RumiAI
```python
# These are multicollinear:
features = {
    'joy_ratio': 0.75,           # Joy frames / total frames
    'emotional_diversity': 0.25,  # 1 - joy_ratio
    'non_joy_ratio': 0.25        # Same as emotional_diversity!
}
```

### Impact by Algorithm

#### Linear Models (Regression)
- **Coefficients become unstable** - small data changes → huge coefficient swings
- **Can't determine true importance** - which feature really matters?
- **Standard errors inflate** - features appear less significant than they are

#### K-means Clustering
- **Double-counts signals** - same information gets weighted twice
- **Distance calculations skewed** - cluster assignments biased toward duplicated info
- **Example**: If height_cm and height_inches both included, tall people appear "further" from average

#### Random Forest
- **Less problematic** - RF randomly selects features at each split
- **Wasted splits** - might use redundant features instead of informative ones
- **Diluted importance** - feature importance spread across correlated features

### Detection Methods

1. **Correlation Matrix**
   ```python
   correlation_matrix = df.corr()
   high_corr = correlation_matrix[abs(correlation_matrix) > 0.9]
   ```

2. **Variance Inflation Factor (VIF)**
   ```python
   from statsmodels.stats.outliers_influence import variance_inflation_factor
   # VIF > 10 indicates problematic multicollinearity
   ```

3. **Domain Knowledge**
   - `total_time` and `duration_sec` - same thing
   - `joy_count` and `joy_ratio` with fixed frame count - perfectly correlated

### Solutions

1. **Remove Redundant Features** - Keep the most interpretable one
2. **Create Composite Features** - PCA or domain-specific combinations
3. **Use Regularization** - L1 (Lasso) naturally selects among correlated features
4. **Feature Engineering** - Replace correlated pairs with their ratio or difference

---

## Practical Recommendations

### DO ✅
- Keep technical properties global (resolution, duration)
- Use temporal windows for events (object appearances, emotions)
- Calculate cross-window metrics globally (uniqueness, variance)
- Check correlation matrix before model training
- Document why each feature is global vs temporal

### DON'T ❌
- Force all features into temporal windows
- Create both temporal and global versions of same metric
- Use arbitrary peak detection thresholds
- Ignore multicollinearity in clustering tasks
- Assume more features = better model

---

## Next Steps for RumiAI

1. **Audit Current Features**
   - Run correlation analysis on existing feature set
   - Identify and remove redundant features
   - Document global vs temporal decisions

2. **Refactor Peak Detection**
   - Replace with percentile-based metrics
   - Use rolling statistics instead of peak counts
   - Keep only physically-meaningful peaks (audio)

3. **Optimize Feature Set**
   - Aim for <50 high-quality features
   - Ensure each feature adds unique information
   - Validate with ablation studies

---

*Last Updated: 2025-09-23*
*Based on production experience with 213,000+ video analyses*