# ML LLM Data Strategy - Technical Implementation

## Overview

This document provides the technical JSON schema specifications for implementing the LLM data formatting strategy outlined in [ML_LLMData.md](ML_LLMData.md).

---

## Single Hashtag Analysis - JSON Schemas

### Option 1: Full Raw Data

#### Random Forest JSON Schema
```json
{
  "bucket": "18-33s",
  "hashtag": "#fitnesstips",
  "analysis_type": "random_forest",
  "video_count": 60,
  "videos": {
    "video_1": {
      "engagement_score": 245000,
      "hook_face_count": 1.8,
      "hook_emotion_joy": 0.65,
      "middle_scene_changes": 2.3,
      "closing_text_density": 0.42,
      "hook_text_density": 0.15,
      "hook_emotion_neutral": 0.25,
      "middle_face_count": 1.5,
      "middle_emotion_joy": 0.68,
      "closing_face_count": 2.0,
      "closing_emotion_joy": 0.72
      // ... 35 features total
    },
    "video_2": {
      "engagement_score": 180000,
      "hook_face_count": 1.2,
      "hook_emotion_joy": 0.55,
      // ... 35 features
    }
    // ... 60 videos total
  },
  "feature_importance": {
    "hook_face_count": 0.23,
    "closing_text_density": 0.18,
    "middle_scene_changes": 0.15,
    "hook_emotion_joy": 0.12,
    "closing_face_count": 0.10,
    "middle_face_count": 0.08,
    "hook_text_density": 0.06,
    "closing_emotion_joy": 0.05,
    "middle_emotion_joy": 0.03
    // ... ranked features totaling 1.0
  }
}
```

**Size:** ~30KB per JSON

#### K-Means JSON Schema
```json
{
  "bucket": "18-33s",
  "hashtag": "#fitnesstips",
  "analysis_type": "kmeans",
  "video_count": 60,
  "videos": {
    "video_1": {
      "cluster_id": 0,
      "engagement_score": 245000,
      "hook_face_count": 1.8,
      "hook_emotion_joy": 0.65,
      "middle_scene_changes": 2.3,
      "closing_text_density": 0.42
      // ... 35 features total
    },
    "video_2": {
      "cluster_id": 1,
      "engagement_score": 180000,
      "hook_face_count": 1.2,
      // ... 35 features
    }
    // ... 60 videos total
  },
  "clusters": {
    "cluster_0": {
      "count": 18,
      "avg_engagement": 350000,
      "centroid": {
        "hook_face_count": 2.1,
        "hook_emotion_joy": 0.72,
        "middle_scene_changes": 3.5,
        "closing_text_density": 0.55
        // ... 35 feature centroids
      }
    },
    "cluster_1": {
      "count": 25,
      "avg_engagement": 180000,
      "centroid": {
        "hook_face_count": 1.0,
        "hook_emotion_joy": 0.58,
        // ... 35 feature centroids
      }
    },
    "cluster_2": {
      "count": 17,
      "avg_engagement": 120000,
      "centroid": {
        "hook_face_count": 0.5,
        // ... 35 feature centroids
      }
    }
  }
}
```

**Size:** ~30KB per JSON

---

### Option 2: Aggregated Statistics

#### Combined RF + K-Means JSON Schema
```json
{
  "bucket": "18-33s",
  "hashtag": "#fitnesstips",
  "video_count": 60,
  "random_forest_insights": {
    "feature_importance": {
      "hook_face_count": 0.23,
      "closing_text_density": 0.18,
      "middle_scene_changes": 0.15,
      "hook_emotion_joy": 0.12,
      "closing_face_count": 0.10,
      "middle_face_count": 0.08,
      "hook_text_density": 0.06,
      "closing_emotion_joy": 0.05,
      "middle_emotion_joy": 0.03,
      "hook_emotion_neutral": 0.02
      // ... top 10 features
    },
    "top_performers_pattern": "High face count + dense closing text + moderate scene changes"
  },
  "features": {
    "hook_face_count": {
      "mean": 1.8,
      "std": 0.4,
      "min": 0.2,
      "max": 5.0,
      "median": 1.5,
      "quartiles": [1.0, 1.5, 2.3],
      "distribution": "bimodal: 0-1 (30%), 2-3 (50%), 4-5 (20%)"
    },
    "hook_emotion_joy": {
      "mean": 0.65,
      "std": 0.15,
      "min": 0.1,
      "max": 0.95,
      "median": 0.68,
      "quartiles": [0.45, 0.68, 0.82],
      "distribution": "normal"
    },
    "middle_scene_changes": {
      "mean": 2.3,
      "std": 0.8,
      "min": 0.5,
      "max": 5.2,
      "median": 2.1,
      "quartiles": [1.5, 2.1, 3.0],
      "distribution": "right-skewed"
    },
    "closing_text_density": {
      "mean": 0.42,
      "std": 0.18,
      "min": 0.0,
      "max": 0.85,
      "median": 0.40,
      "quartiles": [0.25, 0.40, 0.60],
      "distribution": "normal"
    }
    // ... 35 features with full statistics
  },
  "cluster_insights": {
    "cluster_0": {
      "count": 18,
      "avg_engagement": 350000,
      "defining_features": ["high face count", "low text density", "high emotion joy"]
    },
    "cluster_1": {
      "count": 25,
      "avg_engagement": 180000,
      "defining_features": ["single face", "high emotion joy", "moderate text"]
    },
    "cluster_2": {
      "count": 17,
      "avg_engagement": 120000,
      "defining_features": ["B-roll heavy", "high scene changes", "low face count"]
    }
  }
}
```

**Size:** ~6-8KB per bucket

---

## Multiple Hashtag Comparison - JSON Schemas

### Option 1: Full Raw Data (Not Recommended)

#### Random Forest JSON Schema
```json
{
  "bucket": "18-33s",
  "analysis_type": "random_forest",
  "comparison": [
    {
      "hashtag": "#fitnesstips",
      "video_count": 62,
      "videos": {
        "video_1": {
          "engagement_score": 245000,
          "hook_face_count": 1.8,
          "hook_emotion_joy": 0.65
          // ... 35 features
        },
        "video_2": { /* ... */ }
        // ... 62 videos
      },
      "feature_importance": {
        "hook_face_count": 0.23,
        "closing_text_density": 0.18
        // ... ranked features
      }
    },
    {
      "hashtag": "#workoutmotivation",
      "video_count": 58,
      "videos": {
        "video_1": {
          "engagement_score": 320000,
          "hook_face_count": 2.1,
          "hook_emotion_joy": 0.72
          // ... 35 features
        }
        // ... 58 videos
      },
      "feature_importance": {
        "middle_scene_changes": 0.25,
        "hook_emotion_joy": 0.20
        // ... ranked features
      }
    },
    {
      "hashtag": "#gymlife",
      "video_count": 65,
      "videos": { /* 65 videos with 35 features */ },
      "feature_importance": { /* ranked features */ }
    }
  ]
}
```

**Size:** ~90KB per JSON

#### K-Means JSON Schema
```json
{
  "bucket": "18-33s",
  "analysis_type": "kmeans",
  "comparison": [
    {
      "hashtag": "#fitnesstips",
      "video_count": 62,
      "videos": {
        "video_1": {
          "cluster_id": 0,
          "engagement_score": 245000,
          "hook_face_count": 1.8
          // ... 35 features
        }
        // ... 62 videos
      },
      "clusters": {
        "cluster_0": {
          "count": 18,
          "avg_engagement": 350000,
          "centroid": { /* 35 feature values */ }
        },
        "cluster_1": { /* ... */ },
        "cluster_2": { /* ... */ }
      }
    },
    {
      "hashtag": "#workoutmotivation",
      "video_count": 58,
      "videos": { /* 58 videos */ },
      "clusters": { /* 3 clusters */ }
    },
    {
      "hashtag": "#gymlife",
      "video_count": 65,
      "videos": { /* 65 videos */ },
      "clusters": { /* 3 clusters */ }
    }
  ]
}
```

**Size:** ~90KB per JSON
**Total per bucket:** ~180KB
**Total for 8 buckets:** ~1.44MB (EXCEEDS LIMIT)

---

### Option 2: Aggregated Statistics (Recommended)

#### Combined RF + K-Means JSON Schema
```json
{
  "bucket": "18-33s",
  "comparison": [
    {
      "hashtag": "#fitnesstips",
      "video_count": 62,
      "random_forest_insights": {
        "feature_importance": {
          "hook_face_count": 0.23,
          "closing_text_density": 0.18,
          "middle_scene_changes": 0.15,
          "hook_emotion_joy": 0.12,
          "closing_face_count": 0.10,
          "middle_face_count": 0.08,
          "hook_text_density": 0.06,
          "closing_emotion_joy": 0.05,
          "middle_emotion_joy": 0.03,
          "hook_emotion_neutral": 0.02
        },
        "top_performers_pattern": "High face count + dense closing text"
      },
      "features": {
        "hook_face_count": {
          "mean": 1.8,
          "min": 0.2,
          "max": 5.0,
          "median": 1.5,
          "quartiles": [1.0, 1.5, 2.3]
        },
        "hook_emotion_joy": {
          "mean": 0.65,
          "min": 0.1,
          "max": 0.95,
          "median": 0.68,
          "quartiles": [0.45, 0.68, 0.82]
        }
        // ... 35 features with statistics
      },
      "cluster_insights": {
        "cluster_0": {
          "count": 18,
          "avg_engagement": 350000,
          "defining_features": ["high face count", "low text density"]
        },
        "cluster_1": {
          "count": 25,
          "avg_engagement": 180000,
          "defining_features": ["single face", "high emotion joy"]
        },
        "cluster_2": {
          "count": 19,
          "avg_engagement": 120000,
          "defining_features": ["B-roll heavy", "high scene changes"]
        }
      }
    },
    {
      "hashtag": "#workoutmotivation",
      "video_count": 58,
      "random_forest_insights": {
        "feature_importance": {
          "middle_scene_changes": 0.25,
          "hook_emotion_joy": 0.20,
          "closing_face_count": 0.16,
          "hook_face_count": 0.14
          // ... top 10 features
        },
        "top_performers_pattern": "Dynamic scene changes + high energy emotion"
      },
      "features": {
        "hook_face_count": {
          "mean": 2.1,
          "min": 0.5,
          "max": 4.8,
          "median": 2.0,
          "quartiles": [1.2, 2.0, 2.8]
        }
        // ... 35 features with statistics
      },
      "cluster_insights": {
        "cluster_0": {
          "count": 22,
          "avg_engagement": 420000,
          "defining_features": ["multi-person", "high intensity"]
        },
        "cluster_1": { /* ... */ },
        "cluster_2": { /* ... */ }
      }
    },
    {
      "hashtag": "#gymlife",
      "video_count": 65,
      "random_forest_insights": {
        "feature_importance": {
          "closing_text_density": 0.28,
          "hook_text_density": 0.22,
          "middle_face_count": 0.18
          // ... top 10 features
        },
        "top_performers_pattern": "Heavy text overlays + progress showcases"
      },
      "features": {
        "hook_face_count": {
          "mean": 1.5,
          "min": 0.0,
          "max": 3.5,
          "median": 1.3,
          "quartiles": [0.8, 1.3, 2.0]
        }
        // ... 35 features with statistics
      },
      "cluster_insights": {
        "cluster_0": { /* ... */ },
        "cluster_1": { /* ... */ },
        "cluster_2": { /* ... */ }
      }
    }
  ]
}
```

**Size:** ~20-25KB per bucket
**Total for 8 buckets:** ~200KB

---

## Implementation Notes

### Data Type Specifications

**Numeric Features:**
- Type: `float` or `int`
- Precision: 2 decimal places for most features
- Range: Feature-dependent (e.g., emotion scores 0.0-1.0, face counts 0-10+)

**Engagement Score:**
- Type: `int`
- Formula: `views × (1 + share_rate × 10)`
- See [MLAnalysisMode.md](MLAnalysisMode.md) for calculation details

**Cluster IDs:**
- Type: `int`
- Range: 0-2 (3 clusters per bucket)

**Feature Importance:**
- Type: `float`
- Range: 0.0-1.0
- Sum of all feature importance values = 1.0

### Statistical Measures

**Distribution Types:**
- `normal`: Bell-shaped distribution
- `bimodal`: Two distinct peaks
- `right-skewed`: Long tail on right side
- `left-skewed`: Long tail on left side
- `uniform`: Evenly distributed

**Quartiles Format:**
- Array of 3 values: [Q1, Q2/median, Q3]
- Q1 = 25th percentile
- Q2 = 50th percentile (median)
- Q3 = 75th percentile

---

## API Integration

### Claude API Call Structure

```python
import anthropic

def send_to_llm(json_data, prompt_template):
    """
    Send JSON data to Claude API for analysis

    Args:
        json_data: Dictionary containing bucket analysis data
        prompt_template: String template for LLM prompt

    Returns:
        String containing LLM-generated insights
    """
    client = anthropic.Anthropic(api_key="your-api-key")

    # Convert JSON to formatted string
    json_string = json.dumps(json_data, indent=2)

    # Construct prompt
    prompt = prompt_template.format(data=json_string)

    # Make API call
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text
```

### Prompt Templates

#### Single Hashtag Analysis Prompt
```python
SINGLE_HASHTAG_PROMPT = """
Analyze the following TikTok video data for hashtag {hashtag} in the {bucket} duration bucket.

Data includes:
- Random Forest feature importance rankings
- K-Means cluster analysis with 3 creative archetypes
- 60 videos with 35 extracted features each

JSON Data:
{data}

Please provide:
1. Top 3 features driving engagement in this bucket
2. Description of each creative cluster and performance
3. Strategic recommendations for content creators
4. Specific examples of high-performing video patterns

Format your response as a structured analysis with clear sections.
"""
```

#### Multi-Hashtag Comparison Prompt
```python
MULTI_HASHTAG_COMPARISON_PROMPT = """
Compare the following TikTok hashtags for the {bucket} duration bucket.

Data includes aggregated statistics for each hashtag:
- Random Forest feature importance rankings
- Feature distributions (mean, min, max, quartiles)
- K-Means cluster insights

JSON Data:
{data}

Please provide:
1. Key differences in what drives engagement across hashtags
2. Unique creative patterns per hashtag
3. Overlapping success factors
4. Strategic recommendations for choosing between hashtags
5. Audience preference insights based on cluster distributions

Format your response as a comparative analysis with clear sections.
"""
```

---

## Data Pipeline Flow

### Single Hashtag
1. ML Analysis → Generate 2 JSONs per bucket (RF + K-Means)
2. Load both JSONs
3. Send separately to Claude API (16 total calls)
4. Aggregate LLM responses into report

### Multi-Hashtag Comparison
1. ML Analysis → Generate 2 JSONs per bucket per hashtag (already complete)
2. Aggregate RF + K-Means data into combined JSON per bucket
3. Send combined JSON to Claude API (8 total calls)
4. Aggregate LLM responses into comparison report

---

## Testing & Validation

### JSON Schema Validation
- Ensure all required fields present
- Verify data types match specifications
- Check value ranges (e.g., feature importance sums to 1.0)
- Validate array lengths (quartiles = 3 values)

### Size Validation
- Single hashtag: Each JSON ≤ 35KB
- Multi-hashtag: Combined JSON ≤ 30KB per bucket
- Total payload: ≤ 500KB for single, ≤ 250KB for comparison

### LLM Response Quality
- Monitor hallucination rates
- Check response coherence with large payloads
- Validate strategic recommendations align with data
- Test edge cases (low video counts, sparse features)
