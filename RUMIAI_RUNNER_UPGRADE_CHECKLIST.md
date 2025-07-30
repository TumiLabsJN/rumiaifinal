# RumiAI Runner Upgrade Implementation Checklist

## Overview
This document provides a comprehensive checklist for upgrading `rumiai_runner.py` to support ML features with precomputed insights, proper FPS handling, and 6-block output structure for processing videos up to 2 minutes.

## Key Documents Reference
- **RumiAIvInti.md** - Integration requirements and missing components
- **Testytest_improved.md** - Complete prompt templates and data structures  
- **TEMPORAL_MARKERS_DOCUMENTATION.md** - Temporal markers (DO NOT MODIFY)
- **FRAME_PROCESSING_DOCUMENTATION.md** - FPS handling logic
- **claude_output_structures.md** - Expected 6-block output format
- **test_rumiai_complete_flow.js** - Reference implementation that works

## Critical Requirements

### ⚠️ DO NOT MODIFY
- **Temporal Markers** - Working correctly, integrated via PromptBuilder
- **Unified Architecture** - Keep the pipeline structure intact
- **Error Handling** - Preserve existing JSON output structure

### ✅ MUST IMPLEMENT
1. Precompute functions (245+ metrics)
2. 6-block output structure
3. FPS extraction and handling
4. GPU acceleration verification
5. Enhanced ClaudeClient for large payloads

---

## Implementation Checklist

### 1. Import Precompute Functions ✅
**Location**: `/home/jorge/RumiAIv2-clean0907/run_video_prompts_validated_v2.py`

**Functions to import**:
- [ ] `compute_creative_density_analysis` (49 metrics)
- [ ] `compute_emotional_metrics` (40 metrics)
- [ ] `compute_person_framing_metrics` (11 metrics)
- [ ] `compute_scene_pacing_metrics` (31 metrics)
- [ ] `compute_speech_analysis_metrics` (30 metrics)
- [ ] `compute_visual_overlay_metrics` (40 metrics)
- [ ] `compute_metadata_analysis_metrics` (45 metrics)

**Implementation approach**:
```python
# Option A: Import directly
from run_video_prompts_validated_v2 import (
    compute_creative_density_analysis,
    # ... other functions
)

# Option B: Create new module
# rumiai_v2/processors/ml_precompute.py
```

### 2. Extract and Create Prompt Templates ✅
**Source**: Lines from `Testytest_improved.md`
- [ ] Creative Density (lines 179-311)
- [ ] Emotional Journey (lines 399-484)
- [ ] Person Framing (lines 569-652)
- [ ] Scene Pacing (lines 714-805)
- [ ] Speech Analysis (lines 864-953)
- [ ] Visual Overlay (lines 1029-1114)
- [ ] Metadata Analysis (lines 1179-1300)

**Create files**:
```
prompt_templates/
├── creative_density_v2.txt
├── emotional_journey_v2.txt
├── person_framing_v2.txt
├── scene_pacing_v2.txt
├── speech_analysis_v2.txt
├── visual_overlay_v2.txt
└── metadata_analysis_v2.txt
```

### 3. Replace MLDataExtractor ✅
**Current**: Extracts and filters data from unified analysis
**New**: Call precompute functions and prepare context

```python
def prepare_ml_context(self, unified_data: dict, prompt_type: PromptType) -> dict:
    """Replace MLDataExtractor with precompute approach"""
    if prompt_type == PromptType.CREATIVE_DENSITY:
        return {
            "precomputed_metrics": compute_creative_density_analysis(
                unified_data['timelines'], 
                unified_data['duration_seconds']
            ),
            "video_duration": unified_data['duration_seconds']
        }
    # ... other prompt types
```

### 4. Implement FPS Metadata Extraction ✅
**Add to video processing**:
```python
def extract_video_metadata(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    return VideoMetadata(
        original_fps=fps,
        frame_count=frame_count,
        duration=duration,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
```

**FPS Contexts to track**:
1. Original video FPS (e.g., 30fps)
2. Frame extraction FPS (2-5fps adaptive)
3. ML processing FPS (varies by model)
4. Timeline output (1-second buckets)

### 5. Add Response Validation ✅
```python
class ResponseValidator:
    def __init__(self):
        self.expected_blocks = {
            PromptType.CREATIVE_DENSITY: [
                "densityCoreMetrics", "densityDynamics", 
                "densityInteractions", "densityKeyEvents", 
                "densityPatterns", "densityQuality"
            ],
            # ... other types
        }
    
    def validate_claude_response(self, response: str, prompt_type: PromptType) -> dict:
        # Validate 6-block structure
        # Handle missing blocks gracefully
        # Continue pipeline on partial data
```

### 6. Upgrade ClaudeClient ✅
**Critical changes needed**:
- [ ] Change model to `claude-3-5-sonnet-20241022` (from haiku)
- [ ] Increase max_tokens to 4000 (from 1500)
- [ ] Implement dynamic timeout calculation:
  ```python
  def calculate_timeout(base_timeout, payload_size):
      # Add 15s per 500KB of data
      size_adjustment = min(60, (payload_size / 512000) * 15)
      return base_timeout + size_adjustment
  ```
- [ ] Add payload size monitoring (warn at 200KB)
- [ ] Ensure 25-minute total timeout for all prompts

### 7. Ensure GPU Acceleration ✅
**Verify GPU usage in ML services**:
- [ ] Check CUDA availability at startup
- [ ] Log GPU detection status
- [ ] Ensure ML services use GPU when available
- [ ] Monitor VRAM usage for large videos

**Expected GPU acceleration**:
- EasyOCR: ~2.2GB VRAM
- YOLO v8: Auto-detects GPU
- Whisper: Uses GPU via PyTorch
- 10-30x speedup with GPU

### 8. Update Output Structure ✅
**Final JSON should include**:
```json
{
  "creative_density": {
    "densityCoreMetrics": {...},
    "densityDynamics": {...},
    "densityInteractions": {...},
    "densityKeyEvents": {...},
    "densityPatterns": {...},
    "densityQuality": {...}
  },
  "emotional_journey": {
    "emotionalCoreMetrics": {...},
    // ... 5 more blocks
  },
  // ... other 5 flows
  "temporal_markers": {
    // Existing structure - DO NOT CHANGE
  }
}
```

### 9. Add Error Handling ✅
```python
# Graceful degradation for precompute failures
try:
    precomputed = compute_metrics(unified_data, prompt_type)
except Exception as e:
    logger.error(f"Precompute failed: {e}")
    precomputed = {"error": str(e), "fallback": True}
    # Continue with basic metrics
```

### 10. Testing Requirements ✅
- [ ] Test with short video (<30s)
- [ ] Test with medium video (30-60s)
- [ ] Test with long video (60-120s)
- [ ] Verify all 7 precompute functions execute
- [ ] Confirm 6-block output structure
- [ ] Check temporal markers still work
- [ ] Monitor GPU usage and speedup
- [ ] Validate against test_rumiai_complete_flow.js results

---

## Performance Expectations

### For 2-minute video:
- **Frame reduction**: 3,600 → ~240 frames (93% reduction)
- **GPU speedup**: 10-30x faster processing
- **Data compression**: MB → KB to Claude (99% reduction)
- **Total time**: ~5-10 minutes with GPU

### Memory/VRAM Requirements:
- Minimum: 2GB VRAM
- Recommended: 4GB+ VRAM
- CPU RAM: 8GB minimum, 16GB recommended

---

## Implementation Order

1. **Phase 1**: Core Infrastructure
   - [ ] Import/implement precompute functions
   - [ ] Extract prompt templates
   - [ ] Create PromptManager class

2. **Phase 2**: Data Flow
   - [ ] Replace MLDataExtractor
   - [ ] Add FPS metadata extraction
   - [ ] Implement ResponseValidator

3. **Phase 3**: Claude Integration
   - [ ] Upgrade ClaudeClient
   - [ ] Add dynamic timeouts
   - [ ] Implement error handling

4. **Phase 4**: Testing & Optimization
   - [ ] Verify GPU acceleration
   - [ ] Test with various video lengths
   - [ ] Compare with test_rumiai_complete_flow.js

---

## Success Criteria

✅ Handles 2-minute videos reliably
✅ Generates all 6-block outputs correctly
✅ Temporal markers remain functional
✅ GPU acceleration provides 10-30x speedup
✅ Matches output quality of test_rumiai_complete_flow.js

---

## Notes for Implementation

1. **Temporal Markers are Sacred** - They work correctly via PromptBuilder. Do not modify their implementation or integration.

2. **FPS Strategy is Critical** - Without frame reduction, 2-minute videos will overwhelm the system with data.

3. **GPU is Essential** - CPU-only processing is too slow for practical use with 2-minute videos.

4. **Precompute Functions are Required** - They reduce data size from MB to KB, making Claude API calls feasible.

5. **Model Upgrade Matters** - Claude Haiku struggles with complex 6-block outputs; Sonnet is necessary.

This checklist ensures the production runner matches the capabilities proven in the test flow while maintaining backward compatibility and system stability.