
## Brainstorm
- Kickstart Process has to include differentiation between handle or hashtag flow


### 📦 Storage Strategy & Data Management

#### Storage Requirements
- **Per Video**: ~100MB (video file + ML analysis + reports)
- **Per Hashtag**: ~30GB (300 videos × 100MB)
- **Per Client**: ~150-300GB (5-10 hashtags)
- **Total System (6 clients)**: ~2TB by end of 2026
- **Recommended Initial**: 4TB to allow for growth and redundancy

#### Data Organization & Isolation
```
/data/
├── clients/                      # Client data isolation
│   ├── {client_id}/
│   │   ├── raw_videos/           # Original TikTok videos (30-day retention)
│   │   ├── ml_analysis/          # RumiAI analysis outputs
│   │   ├── ml_models/            # Client-specific trained models
│   │   ├── reports/              # Generated PDF reports
│   │   └── checkpoints/          # Processing checkpoints
├── shared/                       # Non-sensitive shared resources
│   └── ml_base_models/           # Pre-trained model weights
└── temp/                         # Temporary processing files
```

#### Data Retention Policy
- **Raw Videos**: 30 days (then delete to save space, can re-download if needed)
- **ML Analysis**: 6 months (compressed after 30 days)
- **ML Models**: Keep latest 3 versions per client/hashtag
- **Reports**: Indefinite (small size, high value)
- **Checkpoints**: 7 days after successful completion

#### Backup & Disaster Recovery
- **Critical Data** (ML models, reports): Daily backup
- **Analysis Data**: Weekly backup (can be regenerated if needed)
- **Raw Videos**: No backup (can re-download from TikTok)
- **Recovery Time Objective (RTO)**: 4 hours
- **Recovery Point Objective (RPO)**: 24 hours for critical data

#### Storage Cost Optimization
- **Compression**: Compress ML analysis JSON after 30 days (70% reduction)
- **Video Deletion**: Remove raw videos after 30 days (saves ~80% of storage)
- **Deduplication**: Share common videos across hashtags when applicable

---

## A. System Architecture

### A.1 Goals - Core Functionalities

#### Primary Goals
1. **Batch Video Analysis**
   - Process up to 300 videos sequentially through `rumiai_runner.py`
   - Implement checkpoint/resume system for failure recovery

2. **Client-Centric Data Organization**
   - Multi-tenant data structure: Client → Hashtags → Duration Buckets → Videos
   - Bucket-specific analysis within client/hashtag boundaries
   - Persistent client/hashtag/duration configuration management

3. **Duration-Specific ML Pattern Recognition**
   - Train **separate ML models for each duration bucket**
   - Recognize that 15-second patterns differ completely from 60-second patterns
   - Generate bucket-specific insights (no universal patterns across durations)

4. **Creative Report Generation**
   - Output 5 creative strategy reports per bucket, or 20 total per Hashtag 
   - Multiple perspectives and strategies for content creators
   - Format: "What works for 15-second #nutrition videos" (not generic advice)
   - Include bucket performance metrics for strategic content planning

#### Success Criteria
- 100% completion rate with checkpoint recovery
- < 2 hours for 200 video batch processing
- Actionable insights with confidence scores > 0.8
- Creative reports readable by non-technical users


### A.2 Non-Goals (Out of Scope)

- ❌ Parallel video processing (maintain sequential for stability)
- ❌ Videos over 120 seconds (TikTok long-form content)
- ❌ Real-time analysis (batch processing only)
- ❌ Cross-client pattern analysis (privacy/competitive isolation)
- ❌ Automatic content generation (insights only, not creation)

---

## B. System Components & Data Flow

### B.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERACTION LAYER                   │
│  Client Selection → Hashtag Config → Batch Parameters       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO ACQUISITION LAYER                  │
│  Apify API → Filter by Date/Duration → Download Queue       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 RUMIAI ANALYSIS PIPELINE                    │
│  rumiai_runner.py → ML Services → Python Compute → JSON     │
│  (>50 features per video)  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    ML TRAINING LAYER                        │
│  Feature Engineering → Model Training → Pattern Detection   │
│  Segmented by: Client × Hashtag × Duration                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 INSIGHT GENERATION LAYER                    │
│  Aggregate Analysis → Claude API → Creative Reports         │
│  Top 5 Creative Combinations per Segment                    │
└─────────────────────────────────────────────────────────────┘
```

### B.2 Detailed Component Flow

#### Step 1: Configuration Setup
```python
{
  "mode": "ml_training",
  "client": {
    "name": "Stateside Grower",
    "is_new": false,
    "industry": "nutritional_supplements"  // Current: supplements, functional_drinks (expanding as needed)
  },
  "hashtags": [
    {
      "name": "#nutrition",
      "url": "https://www.tiktok.com/search?q=%23nutrition",
      "videos_per_segment": 30,
      "date_filter": "2025-01-05",  // User-defined cutoff: only videos after this date
      "segments": ["0-15s", "16-30s", "31-60s", "61-120s"]
    }
  ]
}
```

#### Step 2: Video Acquisition (Enhanced Apify Integration)

#### Step 3: Sequential Processing with Checkpointing

#### Step 4: Bucket-Specific ML Training Architecture

#### Step 5: Bucket Performance Intelligence Report


## E. Success Metrics & KPIs

### Business Value Metrics - Human-Actionable Output Quality
- **Primary Goal**: Creative reports must provide implementable insights that video creators can execute
- **Good Output Example**: "Use text overlays at 3-second intervals with bounce animations, synchronized with gesture changes. Start with question hook in first 2 seconds."
- **Bad Output Example**: "textOverlayDensity: 0.847, gestureCoordination: 0.923"
- **Success Criteria**: Reports contain specific, actionable creative directions, not just statistics
- **Validation**: Manual review of report quality and implementability

### Industry Segmentation Strategy

#### Current Industries (MVP Phase)
- **Nutritional Supplements**: Our primary industry with established patterns
- **Functional Drinks**: Coming soon, high overlap with supplements

**Simplification Decisions**:
- No sub-categories needed (protein vs vitamins) - treating supplements holistically
- No multi-category clients yet - one industry per client for MVP
- Focus: Perfect execution for single-industry clients first

### Technical Validation Metrics

#### 1. Statistical Significance (p-value < 0.05)
- **Definition**: Pattern is NOT due to random chance
- **Threshold**: p-value < 0.05 (95% confidence the pattern is real)
- **Example**: "Text in first 3 seconds" pattern must show statistical difference between top/bottom performers
- **Implementation**: Chi-square or t-tests comparing pattern presence

#### 2. Pattern Consistency (>30% frequency)
- **Definition**: Pattern appears frequently enough to be reliable
- **Threshold**: Must appear in ≥30% of top-performing videos
- **Example**: In 50 top videos, pattern must appear in at least 15
- **Rationale**: Balance between too rare (unreliable) and too common (not differentiating)

#### 3. Silhouette Scores (for clustering approaches)
- **Definition**: How well-separated the pattern groups are
- **Score Ranges**:
  - +0.7 to +1.0: Strong, distinct patterns (videos in group very similar)
  - +0.3 to +0.7: Moderate patterns (some overlap between groups)
  - < 0.3: Weak patterns (groups blend together, not reliable)
- **Example**: Clustering finds "high-energy" vs "educational" styles - score shows how distinct these are

### Performance Baseline: Relative Performance Tiers

#### Tier-Based Pattern Extraction
Instead of comparing against a single baseline, segment videos into performance tiers:

```python
# Segment videos by engagement percentiles within hashtag
tier_1_top_10_percent = videos[:10%]       # Viral/Exceptional - Extract success patterns
tier_2_next_20_percent = videos[10:30%]    # High Performers - Secondary patterns
tier_3_middle_40_percent = videos[30:70%]  # Average/Baseline - Compare against
tier_4_bottom_30_percent = videos[70:100%] # Underperformers - Identify patterns to avoid
```

#### Implementation Strategy
- **Success Patterns**: Extract from Tier 1 (top 10%) that DON'T appear in Tier 3 (middle 40%)
- **Differentiating Factors**: Features that separate top performers from average
- **Anti-Patterns**: Elements common in Tier 4 but rare in Tier 1
- **Context-Aware**: Always comparing within same hashtag/duration segment

#### Example
- If 80% of Tier 1 videos have "question hook in first 2 seconds"
- But only 20% of Tier 3 videos have this
- → This is a strong success pattern to recommend

### Creative Element Engagement Validation

#### Sequential Multi-Method Validation Approach
Creative elements are validated through three sequential layers to ensure reliability:

```python
# Step 1: PRIMARY - Differential Analysis (Quick Filter)
element_in_top_tier = 80%  # Element appears in 80% of top 10% videos
element_in_mid_tier = 20%  # Element appears in 20% of middle 40% videos
differential = 60%
if differential > 30%:  # Threshold for interesting pattern
    → Proceed to Step 2

# Step 2: SECONDARY - Feature Importance from ML Model (Validation)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_features, y_engagement)
importance_score = model.feature_importances_[element_index]  # e.g., 0.15
if importance_score > 0.05:  # Top features threshold
    → Proceed to Step 3

# Step 3: SUPPORTING - Lift Score (Client Communication)
P_high_engagement_with_element = 0.6
P_high_engagement_baseline = 0.2
lift_score = 3.0  # Element triples engagement probability
→ Include in final report
```

#### Confidence Levels Based on Agreement
- **HIGH Confidence**: All three methods agree (differential >30%, importance >0.05, lift >2.0)
- **MEDIUM Confidence**: Two methods agree
- **EXCLUDED**: Only one method shows significance (likely spurious)

#### Example Output for Clients
```
"Question hook in first 2 seconds":
✅ 60% more common in viral videos (Differential Analysis)
✅ Top 5 importance feature in ML model (15% predictive power)
✅ 3.2x more likely to achieve high engagement (Lift Score)
→ CONFIDENCE: HIGH - Strongly recommend implementation
```

### Minimum Confidence Threshold for Pattern Recommendations

#### Analysis of Threshold Approaches Considered

**1. Single Fixed Threshold (e.g., 75% confidence required)**
- *Pros*: Simple, consistent, easy to implement
- *Cons*: Treats all patterns equally, arbitrary cutoff, might miss valuable insights

**2. Tiered Thresholds (80% = "Strongly Recommend", 65% = "Test", 50% = "Consider")**
- *Pros*: Nuanced recommendations, client choice flexibility
- *Cons*: Creates false precision, arbitrary tier boundaries, over-engineering without evidence

**3. Context-Dependent Thresholds (adjust based on data volume)**
- *Pros*: Statistically sound adaptation
- *Cons*: Inconsistent, harder to explain, complex implementation

**4. Top-K Patterns (always report top 5 regardless of confidence)**
- *Pros*: Always provides insights, comparative view
- *Cons*: Might report weak patterns if all are weak

**5. Statistical + Confidence Combined Gates**
- *Critical Flaw*: Over-engineering with redundant filters, risk of excluding valuable patterns

#### Why We Rejected Confidence Thresholds

**Core Problem**: We don't have empirical data on what confidence levels actually correlate with client success:
- No evidence that 80% confidence patterns outperform 65% patterns
- Unknown data distribution (all patterns might cluster at 40-60% or 80-90%)
- Business value undefined (is 3.2x engagement better than 1.5x consistent lift?)
- Confidence is composite of uncertain metrics, creating false precision

#### Selected Approach: Strength of Evidence Labels

**Implementation**:
```python
def classify_pattern_evidence(pattern):
    """
    Score based on multiple independent validation signals
    Each signal represents a different type of evidence
    """
    score = 0
    score += 1 if pattern.differential > 40 else 0           # Strong gap vs average
    score += 1 if pattern.frequency > 50 else 0             # Common in top performers  
    score += 1 if pattern.p_value < 0.01 else 0            # Statistically significant
    score += 1 if pattern.appears_in_multiple_segments else 0  # Consistent across durations
    score += 1 if pattern.lift > 2.0 else 0                # Strong engagement lift
    
    # Classification based on evidence strength
    if score >= 4:
        return "STRONG EVIDENCE"      # 4+ signals agree
    elif score >= 2:
        return "MODERATE EVIDENCE"    # 2-3 signals agree  
    else:
        return "EMERGING PATTERN"     # 1-2 signals, worth monitoring
```

**Why This Works**:
1. **Honest About Uncertainty**: No false precision with confidence percentages
2. **Multiple Validation**: Requires consensus across independent signals
3. **Client-Friendly**: "STRONG EVIDENCE" is immediately understandable
4. **Always Actionable**: Even weak patterns get labeled as "EMERGING" (something to try)
5. **Flexible**: Can add/adjust signals without changing client interface

#### Example Client Report Output
```
═══ STRONG EVIDENCE (4+ validation signals) ═══
✓ Question hooks in first 2 seconds
  • 75% of viral videos use this vs 15% average (Strong Differential)
  • 3.2x engagement lift (Strong Lift)
  • Works across all duration segments (Consistency)
  • Statistically significant (p < 0.001)
  • 60% frequency in top performers (Common Usage)

═══ MODERATE EVIDENCE (2-3 validation signals) ═══
⟳ Text-gesture synchronization  
  • 45% of viral videos vs 25% average (Moderate Differential)
  • 1.8x engagement lift (Moderate Lift)
  • Most effective in 15-30s videos (Limited Consistency)

═══ EMERGING PATTERNS (monitor these trends) ═══
○ Warm color grading
  • 35% of viral videos vs 20% average (Weak Differential)
  • Needs more data across segments to confirm reliability
```

