# Stage 7: LLM Analysis - Hybrid Two-Phase Approach with Dual RF Validation

> **Parent Document**: MLPlanningv2.md Stage 7 "LLM Report Generation"
> **Date**: 2025-01-10
> **Status**: APPROVED - Dual RF LLM Integration
> **Depends On**: MLModelArchitectureStage6.md (Stage 6: Dual RF + K-Means model outputs)

---

## Overview

Stage 7 uses a **two-phase hybrid approach** to generate creative insights from K-Means clustering results, validated by **dual Random Forest analysis**:

- **Phase 1**: Analyze each window type independently with window-level RF validation (6-7 parallel API calls)
- **Phase 2**: Synthesize cross-window patterns and winning formulas with video-level RF validation (1 API call)

**Key Principle**: Minimize hallucination risk with small, focused contexts in Phase 1, then combine insights in Phase 2. Random Forest provides both within-window validation (Phase 1) and cross-window pattern detection (Phase 2).

---

## Stage 6 Dependencies

Stage 7 consumes JSON outputs from Stage 6. For complete model architecture details, see **MLModelArchitectureStage6.md**.

### Stage 6 Outputs (Per Bucket)

**Video-Level RF** (1 JSON):
- `rf_video_analysis.json` - Cross-window feature importance (~30KB)
- Contains: `hook_to_middle_energy_delta`, `middle_to_closing_contrast`, `eye_contact_consistency`
- Used in: Phase 2 synthesis

**Window-Level RF** (6 JSONs):
- `hook_rf_analysis.json`, `middle_1_rf_analysis.json`, ..., `closing_rf_analysis.json`
- Each contains: Feature importance for that specific window (~5KB each)
- Used in: Phase 1 window analyses

**Window-Level K-Means** (6 JSONs):
- `hook_kmeans_analysis.json`, `middle_1_kmeans_analysis.json`, ..., `closing_kmeans_analysis.json`
- Each contains: 3 clusters with 21-dimensional centroids (~5KB each)
- Used in: Phase 1 window analyses

**Total Stage 6 Outputs**: 13 JSON files (~95KB total) per bucket

---

## Architecture

### Two-Phase Strategy with Dual RF Integration

```
Stage 6 Outputs (per bucket)
├── Video-Level RF (1 JSON)     → Cross-window feature importance
├── Window-Level RF (6 JSONs)   → Per-window feature importance
└── K-Means Clustering (6 JSONs) → Per-window cluster centroids
    ↓
┌─────────────────────────────────────────────────────┐
│  Phase 1: Per-Window Analysis (Parallel)            │
├─────────────────────────────────────────────────────┤
│  ├─ Hook API Call        → hook_analysis.json       │
│  ├─ Middle_1 API Call    → middle_1_analysis.json   │
│  ├─ Middle_2 API Call    → middle_2_analysis.json   │
│  ├─ Middle_3 API Call    → middle_3_analysis.json   │
│  ├─ Middle_4 API Call    → middle_4_analysis.json   │
│  └─ Closing API Call     → closing_analysis.json    │
│     (6 calls run in parallel, ~30 seconds total)    │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  Phase 2: Cross-Window Synthesis (Single Call)      │
├─────────────────────────────────────────────────────┤
│  Input: All Phase 1 outputs + video cluster paths   │
│  Output: winning_formulas.json                      │
│     (1 call, ~20 seconds)                           │
└─────────────────────────────────────────────────────┘
    ↓
Final Creative Report (combines Phase 1 + Phase 2)
```

---

## Phase 1: Per-Window Analysis (with Window-Level RF Validation)

### Input Per Window

**Sources**:
1. Stage 6 K-Means outputs (e.g., `hook_kmeans_analysis.json`)
2. Stage 6 Window-Level RF outputs (e.g., `hook_rf_analysis.json`)

**Example Input 1 - K-Means** (`hook_kmeans_analysis.json`):
```json
{
  "window_type": "hook",
  "bucket": "18-33s",
  "total_videos": 100,
  "clusters": [
    {
      "cluster_id": 0,
      "size": 35,
      "centroid": {
        "scene_count": 2.5,
        "eye_contact_rate": 0.87,
        "word_count": 14.2,
        "speech_coverage": 0.45,
        "energy_level": 0.55,
        "gesture_count": 3.2,
        "emotional_valence": 0.6,
        "emotion_consistency": 0.75,
        "average_face_size": 0.42,
        "overlay_unique_count": 1.8,
        "has_captions": 0.8,
        "shortest_scene": 0.8,
        "longest_scene": 2.1,
        "scene_duration_variance": 0.3,
        "object_count": 2.1,
        "person_count": 1.0,
        "energy_variance": 0.15,
        "energy_max": 0.65,
        "pitch_scatter_ratio": 0.35,
        "gaze_variance": 0.12,
        "dominant_emotion_id": 1
      }
    },
    {
      "cluster_id": 1,
      "size": 42,
      "centroid": {
        "scene_count": 4.1,
        "eye_contact_rate": 0.28,
        "word_count": 48.5,
        // ... 18 more features
      }
    },
    {
      "cluster_id": 2,
      "size": 23,
      "centroid": {
        "scene_count": 1.9,
        "eye_contact_rate": 0.65,
        "word_count": 28.3,
        // ... 18 more features
      }
    }
  ]
}
```

**Context Size**: 3 clusters × 21 features = **63 numbers** + metadata

**Example Input 2 - Window-Level RF** (`hook_rf_analysis.json`):
```json
{
  "window_type": "hook",
  "bucket": "18-33s",
  "total_videos": 100,
  "model_performance": {
    "accuracy": 0.82,
    "precision": 0.85,
    "recall": 0.78
  },
  "feature_importance": [
    {
      "feature": "eye_contact_rate",
      "importance": 0.35,
      "top_performer_avg": 0.88,
      "bottom_performer_avg": 0.45,
      "gap": 0.43,
      "rank": 1
    },
    {
      "feature": "energy_level",
      "importance": 0.22,
      "top_performer_avg": 0.82,
      "bottom_performer_avg": 0.54,
      "gap": 0.28,
      "rank": 2
    },
    {
      "feature": "word_count",
      "importance": 0.18,
      "top_performer_avg": 52,
      "bottom_performer_avg": 28,
      "gap": 24,
      "rank": 3
    },
    {
      "feature": "emotional_valence",
      "importance": 0.12,
      "top_performer_avg": 0.65,
      "bottom_performer_avg": 0.42,
      "gap": 0.23,
      "rank": 4
    },
    {
      "feature": "gesture_count",
      "importance": 0.08,
      "top_performer_avg": 6.2,
      "bottom_performer_avg": 3.1,
      "gap": 3.1,
      "rank": 5
    }
    // ... top 10-15 features by importance
  ]
}
```

**Context Size Addition**: Top 10 features × 5 metrics = **50 additional numbers**

**Combined Phase 1 Context**: 63 (K-Means) + 50 (RF) = **113 numbers total** (still manageable)

---

### LLM Prompt Template (Phase 1 with RF Validation)

```
You are analyzing {window_type} segments from 100 viral videos in the {bucket} duration bucket for the #{hashtag} hashtag.

Context:
- These are all TOP-PERFORMING videos (high engagement)
- You are identifying DIFFERENT STRATEGIES that all lead to success
- Focus on what makes each cluster DISTINCT from the others

K-Means clustering has identified 3 distinct {window_type} patterns:

CLUSTER 0 ({size} videos):
{centroid features formatted as bullet list}

CLUSTER 1 ({size} videos):
{centroid features formatted as bullet list}

CLUSTER 2 ({size} videos):
{centroid features formatted as bullet list}

Random Forest Feature Importance ({window_type}-specific predictive power):

The features that BEST PREDICT viral success within {window_type} segments:

1. {feature_1} (importance: {importance}, top avg: {top_avg}, bottom avg: {bottom_avg}, gap: {gap})
2. {feature_2} (importance: {importance}, top avg: {top_avg}, bottom avg: {bottom_avg}, gap: {gap})
3. {feature_3} (importance: {importance}, top avg: {top_avg}, bottom avg: {bottom_avg}, gap: {gap})
4. {feature_4} (importance: {importance}, top avg: {top_avg}, bottom avg: {bottom_avg}, gap: {gap})
5. {feature_5} (importance: {importance}, top avg: {top_avg}, bottom avg: {bottom_avg}, gap: {gap})
... (top 10 features)

Your task:
1. **Name each cluster** with a memorable, creator-friendly label (e.g., "The Direct Eye Contact Hook")
2. **Identify 3-5 defining features** per cluster that differentiate it from the others
   - PRIORITIZE features with high RF importance scores (these are most predictive of viral success)
   - Emphasize features with large top/bottom gaps (biggest performance differentiators)
3. **Describe the strategy** each cluster represents (what creative approach does it use?)
4. **Generate actionable recommendations** - what should creators DO to replicate this pattern?
   - Focus on high-importance RF features first
   - Include target values based on top_performer_avg from RF data

Output format: JSON
{
  "window_type": "{window_type}",
  "clusters": [
    {
      "cluster_id": 0,
      "name": "Creative strategy name",
      "defining_features": [
        "feature_name: value (interpretation)"
      ],
      "strategy_description": "What makes this cluster unique",
      "creator_recommendations": [
        "Specific actionable step 1",
        "Specific actionable step 2",
        "Specific actionable step 3"
      ]
    },
    // ... clusters 1 and 2
  ]
}

Important:
- Be specific and concrete (not generic advice)
- Focus on DIFFERENCES between clusters (not universal best practices)
- Recommendations should be replicable creative techniques
```

---

### Phase 1 Output

**File**: `ml_analysis/llm/{window_type}_analysis.json`

**Example** (`hook_analysis.json`):
```json
{
  "window_type": "hook",
  "bucket": "18-33s",
  "hashtag": "#nutrition",
  "total_videos": 100,
  "clusters": [
    {
      "cluster_id": 0,
      "size": 35,
      "name": "The Direct Eye Contact Hook",
      "defining_features": [
        "eye_contact_rate: 0.87 (RF rank #1, importance 0.35, gap 0.43 - HIGHEST PREDICTOR)",
        "word_count: 14 (RF rank #3, importance 0.18, low count strategy)",
        "energy_level: 0.55 (RF rank #2, importance 0.22, moderate-calm approach)",
        "emotional_valence: 0.6 (RF rank #4, importance 0.12, positive tone)",
        "gesture_count: 3.2 (RF rank #5, importance 0.08, subtle movements)"
      ],
      "rf_validation": {
        "top_predictive_features_in_cluster": [
          "eye_contact_rate: Cluster value 0.87 matches top performer avg 0.88 (RF validated)",
          "energy_level: Cluster value 0.55 near top performer avg 0.82 (moderate variant works)"
        ],
        "insight": "This cluster leverages the #1 most predictive hook feature (eye_contact_rate) at optimal levels. RF confirms high eye contact is the strongest viral predictor for hooks."
      },
      "strategy_description": "Creator looks directly at camera with minimal speech, establishing immediate connection through eye contact rather than information density. Calm, confident presence with positive emotional tone. RF validates this cluster uses the most predictive hook feature (eye contact) at peak performance levels.",
      "creator_recommendations": [
        "PRIORITY: Maintain 85-90% eye contact (RF #1 predictor, importance 0.35, gap 0.43)",
        "Keep opening statement under 15 words (RF #3 predictor, target: 52 words for top performers)",
        "Target moderate energy 0.55-0.60 (RF #2 predictor, this variant succeeds with calm approach)",
        "Use 3-6 subtle gestures (RF #5 predictor, gap 3.1)",
        "Maintain positive emotional tone 0.6+ (RF #4 predictor, gap 0.23)"
      ]
    },
    {
      "cluster_id": 1,
      "size": 42,
      "name": "The Text Overlay Hook",
      "defining_features": [
        "overlay_unique_count: 3.5 (high - multiple text overlays)",
        "eye_contact_rate: 0.28 (low - looking away or at product)",
        "word_count: 48 (very high - talking while showing text)",
        "scene_count: 4.1 (high - dynamic cuts)",
        "has_captions: 0.95 (nearly all videos have captions)"
      ],
      "strategy_description": "Fast-paced, text-heavy opening with multiple scene cuts. Creator speaks rapidly while text overlays reinforce key points. Lower eye contact as focus shifts to product/text.",
      "creator_recommendations": [
        "Add 2-3 text overlays in first 3 seconds (e.g., 'Wait for it...', 'This changes everything')",
        "Use dynamic cuts (3-4 scenes in hook) to maintain attention",
        "Speak quickly - aim for 45-50 words in 3 seconds (natural, not rushed)",
        "Look at product/action rather than camera",
        "Add captions for accessibility and reinforcement"
      ]
    },
    {
      "cluster_id": 2,
      "size": 23,
      "name": "The Action-Driven Hook",
      "defining_features": [
        "object_count: 4.8 (high - multiple props/products visible)",
        "gesture_count: 7.5 (very high - active hand movements)",
        "scene_count: 1.9 (low - single continuous shot)",
        "energy_level: 0.75 (high - dynamic movement)",
        "word_count: 28 (moderate - balanced talking)"
      ],
      "strategy_description": "Single continuous shot with high-energy physical action. Creator actively demonstrates/handles products with frequent gestures. Moderate talking accompanies the visual action.",
      "creator_recommendations": [
        "Film in one continuous take - avoid cuts in first 3 seconds",
        "Use 6-8 hand gestures (pointing, grabbing, showing products)",
        "Show 4-5 different objects/products early (visual density)",
        "Maintain high energy through movement (not just talking)",
        "Balance talking with visual action - don't over-narrate"
      ]
    }
  ],
  "analysis_metadata": {
    "llm_model": "claude-sonnet-4",
    "timestamp": "2025-01-10T14:30:00Z",
    "api_latency_seconds": 4.2
  }
}
```

---

### Phase 1 Execution (Parallel)

**Implementation**:
```python
import concurrent.futures
import json
from anthropic import Anthropic

def analyze_window(window_type: str, kmeans_data: dict, bucket: str, hashtag: str) -> dict:
    """
    Analyze one window type's K-Means clusters.

    Args:
        window_type: 'hook', 'middle_1', 'middle_2', etc.
        kmeans_data: Loaded from {window_type}_kmeans_analysis.json
        bucket: '18-33s'
        hashtag: '#nutrition'

    Returns:
        Phase 1 analysis JSON
    """
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Build prompt from template
    prompt = build_phase1_prompt(window_type, kmeans_data, bucket, hashtag)

    # Call LLM
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        temperature=0.3,  # Lower temperature for consistency
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse JSON response
    analysis = json.loads(response.content[0].text)

    # Add metadata
    analysis['analysis_metadata'] = {
        'llm_model': 'claude-sonnet-4',
        'timestamp': datetime.now().isoformat(),
        'api_latency_seconds': response.usage.total_time_seconds
    }

    return analysis


def run_phase1_parallel(bucket: str, hashtag: str, window_types: list) -> dict:
    """
    Run Phase 1 analysis for all windows in parallel.

    Args:
        bucket: '18-33s'
        hashtag: '#nutrition'
        window_types: ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']

    Returns:
        {
            'hook': {...},
            'middle_1': {...},
            ...
        }
    """
    window_analyses = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # Submit all window analyses
        futures = {}
        for window_type in window_types:
            # Load K-Means data for this window
            kmeans_data = load_json(f'ml_analysis/{window_type}_kmeans_analysis.json')

            # Submit analysis task
            future = executor.submit(
                analyze_window,
                window_type=window_type,
                kmeans_data=kmeans_data,
                bucket=bucket,
                hashtag=hashtag
            )
            futures[window_type] = future

        # Collect results
        for window_type, future in futures.items():
            try:
                analysis = future.result(timeout=60)  # 60s timeout per call
                window_analyses[window_type] = analysis

                # Save individual window analysis
                save_json(f'ml_analysis/llm/{window_type}_analysis.json', analysis)

            except Exception as e:
                logging.error(f"Phase 1 failed for {window_type}: {e}")
                # Continue with other windows even if one fails

    return window_analyses
```

**Execution Time**: ~5-10 seconds (all 6 calls run in parallel)

---

## Phase 2: Cross-Window Synthesis (with Video-Level RF Validation)

### Input Preparation

**Sources**:
1. All Phase 1 window analyses (6-7 JSONs with K-Means + window-level RF)
2. Video cluster assignments across windows (extracted from K-Means outputs)
3. **Video-Level RF cross-window feature importance** (NEW)

**Video Cluster Paths**:
```python
def extract_cluster_paths(window_analyses: dict, kmeans_outputs: dict) -> list:
    """
    Extract each video's cluster assignment across all windows.

    Returns:
        [
            {'video_id': 'video_001', 'path': [0, 1, 0, 1, 2, 0], 'path_str': 'Hook-0 → M1-1 → M2-0 → M3-1 → M4-2 → Closing-0'},
            {'video_id': 'video_002', 'path': [1, 0, 2, 0, 1, 2], 'path_str': 'Hook-1 → M1-0 → M2-2 → M3-0 → M4-1 → Closing-2'},
            ...
        ]
    """
    video_paths = []

    # For each video, extract its cluster assignment from each window
    for video_id in all_video_ids:
        path = []
        for window_type in window_types:
            cluster_id = get_video_cluster(video_id, window_type, kmeans_outputs)
            path.append(cluster_id)

        path_str = format_path(path, window_types)
        video_paths.append({
            'video_id': video_id,
            'path': path,
            'path_str': path_str
        })

    return video_paths


def analyze_path_frequencies(video_paths: list) -> list:
    """
    Identify most common cluster path combinations.

    Returns:
        [
            {'path': [0, 1, 0, 1, 2, 0], 'frequency': 18, 'path_str': '...'},
            {'path': [1, 0, 2, 0, 1, 2], 'frequency': 15, 'path_str': '...'},
            ...
        ]
    """
    path_counts = Counter([tuple(vp['path']) for vp in video_paths])

    # Return top 10 most common paths
    top_paths = []
    for path, count in path_counts.most_common(10):
        top_paths.append({
            'path': list(path),
            'frequency': count,
            'percentage': round(count / len(video_paths) * 100, 1),
            'path_str': format_path(path, window_types)
        })

    return top_paths
```

**Video-Level RF Cross-Window Patterns**:
```json
{
  "bucket": "18-33s",
  "model_type": "video_level_rf",
  "total_videos": 100,
  "input_features": 129,
  "model_performance": {
    "accuracy": 0.87,
    "precision": 0.89,
    "recall": 0.84
  },
  "feature_importance": [
    {
      "feature": "hook_eye_contact_rate",
      "importance": 0.22,
      "top_performer_avg": 0.88,
      "bottom_performer_avg": 0.45,
      "gap": 0.43,
      "rank": 1
    },
    {
      "feature": "middle_3_word_count",
      "importance": 0.18,
      "top_performer_avg": 52,
      "bottom_performer_avg": 28,
      "gap": 24,
      "rank": 2
    },
    {
      "feature": "closing_energy_max",
      "importance": 0.15,
      "top_performer_avg": 0.92,
      "bottom_performer_avg": 0.57,
      "gap": 0.35,
      "rank": 3
    },
    {
      "feature": "hook_to_middle_energy_delta",
      "importance": 0.12,
      "interpretation": "Energy change from hook to middle average",
      "top_performer_avg": 0.15,
      "bottom_performer_avg": -0.08,
      "gap": 0.23,
      "rank": 4,
      "pattern_type": "cross_window"
    },
    {
      "feature": "middle_to_closing_contrast",
      "importance": 0.10,
      "interpretation": "Energy gap between middle avg and closing peak",
      "top_performer_avg": 0.28,
      "bottom_performer_avg": 0.05,
      "gap": 0.23,
      "rank": 5,
      "pattern_type": "cross_window"
    },
    {
      "feature": "eye_contact_consistency",
      "importance": 0.08,
      "interpretation": "Std deviation of eye contact across all windows",
      "top_performer_avg": 0.12,
      "bottom_performer_avg": 0.35,
      "gap": 0.23,
      "rank": 6,
      "pattern_type": "cross_window"
    }
    // ... top 15-20 features including both single-window and cross-window features
  ],
  "cross_window_insights": [
    "Energy progression (hook → middle → closing) has 0.12 importance",
    "Eye contact consistency across windows has 0.08 importance",
    "Closing contrast effect (vs middle) has 0.10 importance"
  ]
}
```

**Key Difference from Window-Level RF**:
- Video-level RF includes **cross-window features** like `hook_to_middle_energy_delta`, `middle_to_closing_contrast`
- These features don't exist in window-level RF (which only sees individual windows)
- Captures temporal progressions and inter-window relationships

---

### LLM Prompt Template (Phase 2 with Video-Level RF)

```
You are synthesizing creative insights for viral videos in the {bucket} duration bucket for #{hashtag}.

You have analyzed 100 viral videos across 6 temporal windows. Each window has been clustered into 3 distinct strategies.

## Per-Window Cluster Analyses

### Hook Analysis:
{Phase 1 hook analysis JSON}

### Middle_1 Analysis:
{Phase 1 middle_1 analysis JSON}

### Middle_2 Analysis:
{Phase 1 middle_2 analysis JSON}

### Middle_3 Analysis:
{Phase 1 middle_3 analysis JSON}

### Middle_4 Analysis:
{Phase 1 middle_4 analysis JSON}

### Closing Analysis:
{Phase 1 closing analysis JSON}

## Most Common Cluster Paths (Video Journey Patterns)

The 10 most common combinations of window strategies:

1. Hook-0 → Middle_1-1 → Middle_2-0 → Middle_3-1 → Middle_4-2 → Closing-0 (18 videos, 18%)
2. Hook-1 → Middle_1-0 → Middle_2-2 → Middle_3-0 → Middle_4-1 → Closing-2 (15 videos, 15%)
3. Hook-0 → Middle_1-0 → Middle_2-0 → Middle_3-0 → Middle_4-0 → Closing-0 (12 videos, 12%)
4. Hook-2 → Middle_1-2 → Middle_2-1 → Middle_3-2 → Middle_4-1 → Closing-1 (10 videos, 10%)
5. Hook-1 → Middle_1-1 → Middle_2-1 → Middle_3-1 → Middle_4-1 → Closing-1 (9 videos, 9%)
...

## Video-Level Random Forest (Cross-Window Pattern Detection)

The features that BEST PREDICT viral success across the ENTIRE VIDEO JOURNEY:

Top Single-Window Features:
1. hook_eye_contact_rate (importance: 0.22, gap: 0.43)
2. middle_3_word_count (importance: 0.18, gap: 24)
3. closing_energy_max (importance: 0.15, gap: 0.35)

Top Cross-Window Features (these only exist at video-level):
4. hook_to_middle_energy_delta (importance: 0.12, gap: 0.23)
   - Top performers: +0.15 energy increase from hook to middle
   - Bottom performers: -0.08 energy decrease
5. middle_to_closing_contrast (importance: 0.10, gap: 0.23)
   - Top performers: 0.28 energy gap (closing peak vs middle avg)
   - Bottom performers: 0.05 energy gap (minimal contrast)
6. eye_contact_consistency (importance: 0.08, gap: 0.23)
   - Top performers: 0.12 std dev (consistent eye contact)
   - Bottom performers: 0.35 std dev (erratic eye contact)

Key Cross-Window Insights from RF:
- Energy progression matters: Building from hook → middle (delta +0.15) predicts virality
- Closing contrast matters: Large energy gap between middle avg and closing peak (0.28) predicts virality
- Consistency matters: Low variance in eye_contact across windows (std 0.12) predicts virality

## Your Task

Identify 3-5 "Winning Formulas" - specific combinations of window strategies that represent successful video archetypes.

For each formula:
1. **Name**: Creative, memorable name (e.g., "The Educator's Arc")
2. **Structure**: Which cluster combination (e.g., Hook-0 + Middle_1-1 + Closing-0)
3. **Frequency**: How common is this pattern?
4. **Temporal Progression**: How do key features evolve across windows? (e.g., "energy builds from 0.55 → 0.75 → 0.85")
   - VALIDATE against video-level RF cross-window features
   - If formula shows energy_delta +0.15, note this matches RF top performers
5. **RF Cross-Window Validation**: Which cross-window RF patterns does this formula exhibit?
   - Does it show hook_to_middle_energy_delta near +0.15 (RF rank #4)?
   - Does it show middle_to_closing_contrast near 0.28 (RF rank #5)?
   - Does it show eye_contact_consistency near 0.12 std dev (RF rank #6)?
6. **Strategy Description**: What is the overall creative approach?
7. **When to Use**: What type of content/creator fits this formula?
8. **Step-by-Step Template**: Concrete steps to replicate this formula
   - Include cross-window targets (e.g., "Energy should increase by ~0.15 from hook to middle")

Output format: JSON
{
  "winning_formulas": [
    {
      "name": "Formula name",
      "structure": {
        "hook": "Cluster name from Phase 1",
        "middle_pattern": "Cluster progression description",
        "closing": "Cluster name from Phase 1"
      },
      "cluster_path": [0, 1, 0, 1, 2, 0],
      "frequency": 18,
      "percentage": 18.0,
      "temporal_progressions": [
        {
          "feature": "energy_level",
          "hook": 0.55,
          "middle_avg": 0.65,
          "closing": 0.85,
          "pattern": "Builds from moderate to high"
        },
        // ... 2-3 more key features
      ],
      "strategy_description": "Overall creative approach",
      "when_to_use": "Content types and creator profiles that fit this formula",
      "step_by_step_template": [
        "Step 1: ...",
        "Step 2: ...",
        "Step 3: ..."
      ]
    },
    // ... 2-4 more formulas
  ],
  "cross_window_insights": [
    "Key insight about temporal patterns",
    "Key insight about transitions between windows",
    "Key insight about consistency vs variation"
  ]
}

Important:
- Prioritize most frequent paths (top 3-5)
- Highlight temporal evolutions (how features change across windows)
- Be specific about when/why creators should use each formula
- Identify patterns across multiple paths (e.g., "All high-performing videos build energy toward closing")
```

---

### Phase 2 Output

**File**: `ml_analysis/llm/winning_formulas.json`

**Example**:
```json
{
  "bucket": "18-33s",
  "hashtag": "#nutrition",
  "total_videos": 100,
  "winning_formulas": [
    {
      "name": "The Educator's Arc",
      "structure": {
        "hook": "The Direct Eye Contact Hook (Cluster 0)",
        "middle_pattern": "Information Dense Middle (Cluster 1 → 1 → 1 → 2)",
        "closing": "High Energy CTA (Cluster 0)"
      },
      "cluster_path": [0, 1, 0, 1, 2, 0],
      "frequency": 18,
      "percentage": 18.0,
      "temporal_progressions": [
        {
          "feature": "energy_level",
          "hook": 0.55,
          "middle_1": 0.60,
          "middle_2": 0.62,
          "middle_3": 0.68,
          "middle_4": 0.75,
          "closing": 0.85,
          "pattern": "Steady build from moderate (0.55) to high (0.85)",
          "hook_to_middle_delta": 0.16,
          "middle_to_closing_contrast": 0.27
        },
        {
          "feature": "word_count",
          "hook": 14,
          "middle_avg": 52,
          "closing": 18,
          "pattern": "Low hook → dense middle → moderate closing (inverted U-shape)"
        },
        {
          "feature": "eye_contact_rate",
          "hook": 0.87,
          "middle_avg": 0.45,
          "closing": 0.82,
          "pattern": "High in hook/closing, lower in middle (bookend pattern)",
          "consistency_std_dev": 0.15
        }
      ],
      "rf_cross_window_validation": {
        "matches_top_patterns": [
          "hook_to_middle_energy_delta: 0.16 (matches RF top performer avg 0.15, RF rank #4, importance 0.12)",
          "middle_to_closing_contrast: 0.27 (matches RF top performer avg 0.28, RF rank #5, importance 0.10)",
          "eye_contact_consistency: 0.15 std dev (close to RF top performer avg 0.12, RF rank #6, importance 0.08)"
        ],
        "insight": "This formula exhibits ALL THREE major cross-window patterns identified by video-level RF as predictive of viral success. The energy build (+0.16) and closing contrast (0.27) are near-perfect matches to RF top performers.",
        "rf_validation_score": "9/10 - Strongly validated by video-level RF cross-window analysis"
      },
      "strategy_description": "Start with intimate eye contact to build trust, deliver dense educational content in middle segments while looking at product/demonstration, return to direct eye contact for high-energy call-to-action. Energy builds throughout, word density peaks in middle, eye contact bookends the video.",
      "when_to_use": "Educational nutrition content, product explanations, how-to videos. Best for creators comfortable with direct camera presence and knowledgeable about their topic. Works when selling credibility and expertise.",
      "step_by_step_template": [
        "Hook (0-3s): Direct eye contact (0.87), minimal words (14), moderate energy (0.55), positive expression",
        "Middle_1 (3-8s): Shift to product/demonstration view, increase talking speed (50+ words), build energy to 0.60 (+0.05 from hook)",
        "Middle_2 (8-13s): Continue information delivery, keep eye contact low, energy 0.62 (steady progression)",
        "Middle_3 (13-18s): Maintain educational pace, build energy to 0.68 (total +0.13 from hook)",
        "Middle_4 (18-23s): Energy 0.75 (total +0.20 from hook - approaching RF target of +0.15 to middle avg)",
        "Closing (23-26s): Return to direct eye contact (0.82), peak energy (0.85), clear CTA with moderate words (18)",
        "CROSS-WINDOW TARGETS (RF validated):",
        "  - Energy delta hook→middle: +0.16 (RF target: +0.15)",
        "  - Energy contrast middle→closing: 0.27 gap (RF target: 0.28)",
        "  - Eye contact consistency: 0.15 std dev across all windows (RF target: 0.12)"
      ]
    },
    {
      "name": "The Text-Driven Viral",
      "structure": {
        "hook": "The Text Overlay Hook (Cluster 1)",
        "middle_pattern": "Fast Cuts & Text (Cluster 0 → 2 → 0 → 1)",
        "closing": "Visual Payoff (Cluster 2)"
      },
      "cluster_path": [1, 0, 2, 0, 1, 2],
      "frequency": 15,
      "percentage": 15.0,
      "temporal_progressions": [
        {
          "feature": "scene_count",
          "hook": 4.1,
          "middle_avg": 3.8,
          "closing": 2.2,
          "pattern": "High pace throughout, slows slightly for closing payoff"
        },
        {
          "feature": "overlay_unique_count",
          "hook": 3.5,
          "middle_avg": 2.8,
          "closing": 1.5,
          "pattern": "Heavy text in hook, moderate in middle, minimal in closing"
        },
        {
          "feature": "energy_level",
          "hook": 0.70,
          "middle_avg": 0.75,
          "closing": 0.65,
          "pattern": "Consistent high energy, slight dip for closing visual focus"
        }
      ],
      "strategy_description": "Text-heavy, fast-paced opening grabs attention with multiple overlays and cuts. Middle maintains energy with text reinforcement and dynamic editing. Closing slows down for visual payoff (product result, transformation, etc.) with fewer text distractions.",
      "when_to_use": "Attention-grabbing content, before/after transformations, product reveals. Works for creators who rely on editing over on-camera presence. Best when visual payoff is strong.",
      "step_by_step_template": [
        "Hook (0-3s): 3-4 text overlays, 4+ scene cuts, high energy, talk fast (45-50 words)",
        "Middle_1 (3-8s): Maintain text overlays (2-3), keep dynamic pace",
        "Middle_2 (8-13s): Reduce text slightly (2), sustain energy and word count",
        "Middle_3 (13-18s): Continue fast pace, prepare for visual transition",
        "Middle_4 (18-23s): Start reducing text overlays (1-2), maintain energy",
        "Closing (23-26s): Minimal text (0-1), slower cuts (2 scenes), focus on visual result, slight energy dip to let visual speak"
      ]
    },
    {
      "name": "The Consistent Approach",
      "structure": {
        "hook": "The Direct Eye Contact Hook (Cluster 0)",
        "middle_pattern": "Steady Consistency (Cluster 0 → 0 → 0 → 0)",
        "closing": "Direct CTA (Cluster 0)"
      },
      "cluster_path": [0, 0, 0, 0, 0, 0],
      "frequency": 12,
      "percentage": 12.0,
      "temporal_progressions": [
        {
          "feature": "eye_contact_rate",
          "hook": 0.87,
          "middle_avg": 0.85,
          "closing": 0.88,
          "pattern": "Consistently high throughout (no variation)"
        },
        {
          "feature": "energy_level",
          "hook": 0.55,
          "middle_avg": 0.58,
          "closing": 0.60,
          "pattern": "Moderate and steady (minimal build)"
        },
        {
          "feature": "emotion_consistency",
          "hook": 0.75,
          "middle_avg": 0.78,
          "closing": 0.80,
          "pattern": "High emotional consistency (focused, unwavering tone)"
        }
      ],
      "strategy_description": "Maintains consistent direct eye contact, moderate energy, and emotional tone throughout the entire video. No dramatic shifts or builds - creates intimacy and trust through unwavering presence. Works like a conversation with a trusted friend.",
      "when_to_use": "Personal stories, vulnerable content, trust-building messages. Best for creators with strong on-camera presence and emotional authenticity. Works when message matters more than production value.",
      "step_by_step_template": [
        "Entire video (0-26s): Maintain 85%+ eye contact, keep energy at moderate/steady level (0.55-0.60), avoid dramatic tone shifts",
        "Focus on emotional authenticity over editing tricks",
        "Let message carry the video, not visual effects or pacing",
        "Keep word count moderate and conversational throughout",
        "Minimal text overlays, minimal cuts - let intimacy build through consistency"
      ]
    }
  ],
  "cross_window_insights": [
    "78% of high-performing videos use 'bookend' eye contact pattern (high in hook/closing, lower in middle)",
    "Energy builds are common (65% of videos), but 12% succeed with consistent energy",
    "Text overlay usage inversely correlates with eye contact (high text = low eye contact pattern)",
    "Middle segment diversity matters - only 12% use same cluster across all middle segments",
    "Closing energy should match or exceed middle average (85% of top performers follow this)"
  ],
  "analysis_metadata": {
    "llm_model": "claude-sonnet-4",
    "timestamp": "2025-01-10T14:35:00Z",
    "api_latency_seconds": 18.5
  }
}
```

---

### Phase 2 Execution

**Implementation**:
```python
def run_phase2_synthesis(
    window_analyses: dict,
    kmeans_outputs: dict,
    bucket: str,
    hashtag: str
) -> dict:
    """
    Synthesize cross-window patterns from Phase 1 analyses.

    Args:
        window_analyses: Phase 1 outputs (hook_analysis, middle_1_analysis, etc.)
        kmeans_outputs: Raw K-Means data for extracting video cluster paths
        bucket: '18-33s'
        hashtag: '#nutrition'

    Returns:
        Phase 2 synthesis JSON (winning formulas)
    """
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Extract video cluster paths
    video_paths = extract_cluster_paths(window_analyses, kmeans_outputs)
    top_paths = analyze_path_frequencies(video_paths)

    # Build Phase 2 prompt
    prompt = build_phase2_prompt(
        window_analyses=window_analyses,
        top_paths=top_paths,
        bucket=bucket,
        hashtag=hashtag
    )

    # Call LLM
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,  # Larger for synthesis
        temperature=0.4,  # Slightly higher for creativity
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse JSON response
    synthesis = json.loads(response.content[0].text)

    # Add metadata
    synthesis['bucket'] = bucket
    synthesis['hashtag'] = hashtag
    synthesis['total_videos'] = len(video_paths)
    synthesis['analysis_metadata'] = {
        'llm_model': 'claude-sonnet-4',
        'timestamp': datetime.now().isoformat(),
        'api_latency_seconds': response.usage.total_time_seconds
    }

    # Save synthesis
    save_json('ml_analysis/llm/winning_formulas.json', synthesis)

    return synthesis
```

**Execution Time**: ~15-20 seconds (larger context, more complex reasoning)

---

## Complete Stage 7 Pipeline

### Orchestration Code

```python
def run_stage7_llm_analysis(bucket: str, hashtag: str) -> dict:
    """
    Complete Stage 7 pipeline: Phase 1 + Phase 2.

    Args:
        bucket: '18-33s'
        hashtag: '#nutrition'

    Returns:
        Complete creative analysis with window insights + winning formulas
    """
    logger.info(f"Starting Stage 7 LLM Analysis for {bucket} / {hashtag}")

    # Determine window types for this bucket
    window_types = get_window_types_for_bucket(bucket)
    # e.g., ['hook', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'closing']

    # Phase 1: Analyze each window in parallel
    logger.info("Phase 1: Analyzing each window type...")
    start_time = time.time()

    window_analyses = run_phase1_parallel(
        bucket=bucket,
        hashtag=hashtag,
        window_types=window_types
    )

    phase1_time = time.time() - start_time
    logger.info(f"Phase 1 completed in {phase1_time:.1f}s ({len(window_types)} windows in parallel)")

    # Validate Phase 1 outputs
    if len(window_analyses) != len(window_types):
        logger.warning(f"Phase 1 incomplete: {len(window_analyses)}/{len(window_types)} windows analyzed")

    # Phase 2: Synthesize cross-window patterns
    logger.info("Phase 2: Synthesizing winning formulas...")
    start_time = time.time()

    # Load K-Means outputs for cluster path extraction
    kmeans_outputs = load_kmeans_outputs(bucket, window_types)

    synthesis = run_phase2_synthesis(
        window_analyses=window_analyses,
        kmeans_outputs=kmeans_outputs,
        bucket=bucket,
        hashtag=hashtag
    )

    phase2_time = time.time() - start_time
    logger.info(f"Phase 2 completed in {phase2_time:.1f}s")

    # Combine Phase 1 + Phase 2 into final output
    complete_analysis = {
        'bucket': bucket,
        'hashtag': hashtag,
        'window_analyses': window_analyses,
        'winning_formulas': synthesis['winning_formulas'],
        'cross_window_insights': synthesis['cross_window_insights'],
        'execution_metrics': {
            'phase1_time_seconds': phase1_time,
            'phase2_time_seconds': phase2_time,
            'total_time_seconds': phase1_time + phase2_time,
            'api_calls': len(window_types) + 1
        }
    }

    # Save complete analysis
    save_json(f'ml_analysis/llm/complete_analysis_{bucket}.json', complete_analysis)

    logger.info(f"Stage 7 complete. Total time: {phase1_time + phase2_time:.1f}s")

    return complete_analysis
```

---

## API Call Summary (with Dual RF Integration)

### Per Bucket (e.g., 18-33s with 6 windows)

| Phase | API Calls | Execution | Context Size | Cost Estimate |
|-------|-----------|-----------|--------------|---------------|
| **Phase 1** | 6 calls (parallel) | ~5-10s wall-clock | 113 numbers each (63 K-Means + 50 window RF) | 6 × $0.03 = $0.18 |
| **Phase 2** | 1 call | ~15-20s | ~800 numbers + summaries (includes video RF) | $0.08 |
| **Total** | 7 calls | ~25-30s | — | **$0.26 per bucket** |

**Note**: Context size increased due to dual RF integration:
- Phase 1: Added window-level RF feature importance (50 numbers per window)
- Phase 2: Added video-level RF cross-window patterns (100 numbers)

### Per Complete Analysis (3 active buckets)

Assuming top 3 buckets are 18-33s (6 windows), 33-60s (7 windows), 60-90s (7 windows):

| Bucket | Windows | Phase 1 Calls | Phase 2 Calls | Total Calls |
|--------|---------|---------------|---------------|-------------|
| 18-33s | 6 | 6 | 1 | 7 |
| 33-60s | 7 | 7 | 1 | 8 |
| 60-90s | 7 | 7 | 1 | 8 |
| **Total** | **20** | **20** | **3** | **23 calls** |

**Total Cost Estimate**: ~$0.78 per complete analysis (3 buckets) - 56% increase due to dual RF integration

**Total Execution Time**: ~90 seconds (phases parallelizable across buckets)

---

## Advantages of Dual RF Hybrid Approach

### vs. Single API Call Per Bucket (Option 2)

| Metric | Dual RF Hybrid (Option 3) | Single Call (Option 2) |
|--------|---------------------------|------------------------|
| **Context per call** | Phase 1: 113 numbers (K-Means + window RF) | 1000+ numbers |
| **Hallucination risk** | Low (focused prompts) | Higher (overwhelming context) |
| **Parallelization** | Yes (6-7 calls in Phase 1) | No (1 sequential call) |
| **Fault tolerance** | High (one window failure doesn't block others) | Low (single failure loses all) |
| **Cross-window patterns** | ✅ Video-level RF in Phase 2 | ✅ Captured |
| **Within-window validation** | ✅ Window-level RF in Phase 1 | ❌ Not available |
| **API calls** | 7 per bucket | 1 per bucket |
| **Cost** | ~$0.26 per bucket | ~$0.05 per bucket |
| **Total time** | ~25-30s | ~30-40s (larger context) |

**Trade-off**: 5.2x more API calls and cost, but gains both cross-window AND within-window RF validation. Complete pattern coverage with no blind spots.

### vs. Per-Window Only (Option 1)

| Metric | Dual RF Hybrid (Option 3) | Per-Window Only (Option 1) |
|--------|---------------------------|----------------------------|
| **Cross-window patterns** | ✅ Video-level RF in Phase 2 | ❌ Lost |
| **Within-window validation** | ✅ Window-level RF in Phase 1 | ✅ Available (if using window RF) |
| **Temporal arcs** | ✅ Identified + RF validated | ❌ Not visible |
| **Winning formulas** | ✅ Generated + RF validated | ❌ Manual combination needed |
| **API calls** | 7 per bucket | 6 per bucket |
| **Cost** | ~$0.26 per bucket | ~$0.18 per bucket (with window RF) |
| **Actionability** | Highest (formulas + window strategies + RF targets) | Medium (window strategies only) |

**Trade-off**: 1 additional API call for synthesis + video-level RF data, but gains holistic insights and complete pattern validation.

---

## Error Handling

### Phase 1 Failure Scenarios

**Problem**: One window's LLM call fails (network error, timeout, invalid JSON)

**Solution**:
```python
# Phase 1 execution handles failures gracefully
for window_type, future in futures.items():
    try:
        analysis = future.result(timeout=60)
        window_analyses[window_type] = analysis
    except TimeoutError:
        logger.error(f"Phase 1 timeout for {window_type} (>60s)")
        # Retry once with exponential backoff
        analysis = retry_with_backoff(analyze_window, window_type, ...)
        window_analyses[window_type] = analysis
    except json.JSONDecodeError:
        logger.error(f"Phase 1 invalid JSON for {window_type}")
        # Save raw response for debugging
        save_raw_response(window_type, response)
        # Skip this window, continue with others
    except Exception as e:
        logger.error(f"Phase 1 failed for {window_type}: {e}")
        # Skip this window, continue with others
```

**Impact**: If 1-2 windows fail, Phase 2 can still run with remaining windows (degraded but functional)

### Phase 2 Failure Scenarios

**Problem**: Synthesis LLM call fails

**Solution**:
```python
# Phase 2 execution with retry logic
max_retries = 3
for attempt in range(max_retries):
    try:
        synthesis = run_phase2_synthesis(...)
        break
    except Exception as e:
        logger.error(f"Phase 2 attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
        else:
            # Fallback: Return Phase 1 outputs only
            logger.error("Phase 2 failed after 3 attempts. Returning Phase 1 only.")
            synthesis = {'winning_formulas': [], 'cross_window_insights': []}
```

**Impact**: Graceful degradation - Phase 1 window analyses still available even if synthesis fails

---

## Output File Structure

```
bucket_18-33s/
└── ml_analysis/
    └── llm/
        ├── hook_analysis.json              # Phase 1 output (3 clusters, named strategies)
        ├── middle_1_analysis.json          # Phase 1 output
        ├── middle_2_analysis.json          # Phase 1 output
        ├── middle_3_analysis.json          # Phase 1 output
        ├── middle_4_analysis.json          # Phase 1 output
        ├── closing_analysis.json           # Phase 1 output
        ├── winning_formulas.json           # Phase 2 output (3-5 formulas)
        └── complete_analysis_18-33s.json   # Combined Phase 1 + Phase 2
```

---

## Next Steps

1. ✅ **Random Forest Architecture**: APPROVED - Documented in MLModelArchitectureStage6.md
2. ✅ **LLM Integration Strategy**: APPROVED - Hybrid two-phase with dual RF validation
3. **Update**: Propagate Stage 7 LLM integration to MLPlanningv2.md
4. **Stage 6 Implementation**: Build dual RF training pipeline (see MLModelArchitectureStage6.md)
5. **Prompt Engineering**: Refine Phase 1 and Phase 2 prompt templates with real data
6. **Cost Validation**: Test with real API calls to validate cost estimates (~$0.78 per 3-bucket analysis)

---

## Document Organization Note

**This document (LLMAnalysis7.md)** covers:
- ✅ How Stage 7 consumes Stage 6 outputs
- ✅ Phase 1 & Phase 2 LLM prompt structures
- ✅ API call orchestration and cost estimates
- ✅ Winning formulas synthesis with RF validation

**MLModelArchitectureStage6.md** covers:
- ✅ Dual RF architecture (49 models: 8 video-level + 41 window-level)
- ✅ Window-level K-Means architecture (41 models)
- ✅ Stage 6 JSON output specifications
- ✅ Complete file architecture and model training logic

**Why this split?**
- Stage 6 is responsible for **producing** ML model outputs
- Stage 7 is responsible for **consuming** those outputs in LLM prompts
- Clear separation of concerns prevents duplication and confusion

---

**Status**: APPROVED - Ready for Stage 6 implementation and MLPlanningv2.md updates
