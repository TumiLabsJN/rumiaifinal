# LocalTestv2 - Temporal Windows Local Video Testing Framework

## IMPORTANT
Do NOT make any change to production code unless explicitly instructed to. This test flow should not modify the main script python3 scripts/rumiai_runner.py 
Last test flow we created completely nuked production code and caused us to lose 8 hours of work.

## Executive Summary

A purpose-built testing framework for validating the new temporal windows architecture against human interpretations of local video files. This framework enables rapid iteration and quality assurance of ML analysis by comparing machine output with human annotations.

## Purpose & Goals

### Primary Purpose
Create a streamlined testing pipeline that:
1. Processes local MP4 files through the new temporal windows architecture
2. Generates single unified JSON output (replacing 7-flow/21-file structure)
3. **Validates BOTH ML detection accuracy AND temporal window calculations**
4. Produces actionable comparison reports for improving both ML models and window logic

### Key Goals
- **Dual Validation**: 
  - ML Accuracy: Verify YOLO, Whisper, MediaPipe, OCR detections are correct
  - Temporal Logic: Confirm window boundaries and segment divisions are appropriate
- **Rapid Testing**: Test videos without TikTok scraping/download overhead
- **Human Validation**: Compare ML detection with ground truth annotations
- **Quality Assurance**: Identify gaps in both ML pipeline and temporal calculations
- **Developer Friendly**: Simple command-line interface with clear outputs

## Architecture Design

### Script Name & Location
- **Script**: `temporal_test_runner.py`
- **Location**: `/scripts/testing/temporal_test_runner.py`
- **Rationale**: Separate from production scripts, clear testing purpose

### Integration Approach
- **Uses EXACT same ML pipeline as production** (after video acquisition)
- **Skips only**: Apify scraping, video download, TikTok metadata fetch
- **Preserves**: All ML services, timeline building, temporal window calculations
- **Rationale**: Test results must match what production would generate

### Core Components

```python
class TemporalTestRunner:
    """Main test orchestrator"""
    - process_local_video()      # Run ML pipeline on local file (same as production)
    - create_mock_metadata()     # Generate minimal metadata for local video
    - load_human_annotations()   # Parse human interpretation
    - validate_results()         # Compare ML vs Human
    - generate_report()          # Output validation metrics
    
class AnnotationMapper:
    """Maps flexible human annotations to ML windows"""
    - map_to_temporal_windows()  # Convert time ranges to windows
    - calculate_overlap()        # Find time overlaps
    - extract_window_content()   # Get annotations for specific window
    
class ValidationEngine:
    """Compares ML output with human annotations"""
    - compare_elements()         # Element-by-element comparison
    - calculate_match_scores()   # Quantify accuracy
    - identify_gaps()           # Find missing/extra detections
```

### Input/Output Structure

```
Inputs:
├── video_path: str              # Path to local .mp4 file
├── annotations_path: str        # Path to human annotations (.yaml)
└── output_dir: str             # Where to save results (default: test_results/)

Outputs:
test_results/
├── {video_id}/
│   ├── temporal_unified.json   # ML analysis output
│   ├── annotations.yaml        # Copy of human input
│   ├── validation_report.json  # Detailed comparison
│   ├── validation_summary.md   # Human-readable report
│   └── debug/
│       ├── ml_timelines.json   # Raw timeline data
│       └── window_mapping.json # How annotations mapped to windows
```

## Human Annotation Format

### Annotation Philosophy
- **Golden Test Videos**: Focus on comprehensive annotation of carefully selected test videos
- **Target Time**: 5 minutes per video MAX
- **Target Volume**: 2-3 golden videos per duration bucket (8-12 total)
- **Strategy**: Choose videos that stress-test different ML capabilities and window boundaries

### Golden Test Video Selection Criteria

**Video Source**: Real TikTok videos downloaded locally
- **Rationale**: Tests actual production content patterns
- **Selection Method**: Download videos that match criteria below
- **Storage**: `test_videos/` directory with descriptive names

Each duration bucket should include videos that test:

**Duration Bucket: 0-15s** (Real TikTok Examples)
1. **Hook-Heavy**: Strong 3s hook with multiple elements (text, face, gesture)
2. **Rapid Cuts**: Maximum scene changes to test detection limits

**Duration Bucket: 16-30s** (Real TikTok Examples)
1. **Clear Structure**: Obvious hook → development → CTA pattern
2. **Multi-Modal**: Heavy speech + text overlays + gestures
3. **Minimal/Clean**: Few elements to test detection sensitivity

**Duration Bucket: 31-60s** (Real TikTok Examples)
1. **Tutorial Style**: Steady pacing with clear segments
2. **High Energy Dance**: Peak in middle with multiple climaxes
3. **Story/Skit**: Natural narrative flow with dialogue

**Duration Bucket: 61-120s** (Real TikTok Examples)
1. **Long-Form Educational**: Tests segment boundaries at scale
2. **Multi-Scene Production**: Location changes, varied pacing

**Edge Cases to Include** (Find real examples):
- Video with no speech (music only)
- Video with no text overlays
- Single-shot video (no cuts)
- Extreme density (overlays every second)
- Face never visible (product/food only)

### Annotation Process
- **Single Annotator**: Jorge only (ensures consistency)
- **No inter-rater reliability needed**: Single perspective
- **Benefit**: No need for annotation guidelines or consensus protocols
- **Trade-off**: Accepted subjectivity for speed and simplicity

### Quick Raw Annotation Format

```yaml
# Metadata (30 seconds to fill)
video_id: "test_video_01"
duration: 45.5
annotator: "jorge"  # Always jorge for consistency
date: "2025-01-15"

# Quick observations (4.5 minutes to write)
observations:
  "0-3s": "surprise face, TEXT: watch this + won't believe, pointing up, 1 cut"
  
  "3-10s": "dancing starts, wide shot, 2 cuts, gestures++"
  
  "10-20s": "peak energy, TEXT: insane + crazy, rapid cuts (5+), multiple angles"
  
  "20-35s": "talking/explaining, face visible, some pointing, steady shot"
  
  "35-42s": "winding down, less movement, 1 cut"
  
  "42-45s": "TEXT: follow for part 2, pointing at camera, CTA, happy face"

# Quick tags (entire video)
has_speech: yes
has_music: yes
dominant_person_count: 1
main_theme: "dance tutorial"
```

### Two-Stage Processing

1. **Raw Annotations** → **Structured Annotations** (via parser)
2. **Structured Annotations** → **Production Format** (for comparison)

The parser converts shortcuts like:
- `TEXT: xyz` → text_overlays: ["xyz"]
- `rapid cuts (5+)` → scene_changes: 5
- `gestures++` → multiple gestures detected
- `face visible` → face_visible: true
- `CTA` → has_cta: true

### Output Format
- **ML Output**: Identical `temporal_unified.json` as production will generate
- **No test metadata mixed in**: Clean production format
- **Validation results**: Stored in separate files (`validation_report.json`, `validation_summary.md`)
- **Rationale**: Test output must be production-ready without modifications

# Middle (3s to last 3s) - Variable segments
middle_observations:
  - time_range: "3-8s"
    description: "Dance begins, establishing shot"
    observed_elements:
      face_visible: true
      gestures: ["arms_up", "clapping"]
      scene_changes: 2
      dominant_action: "dancing"
    energy_level: "medium"
    
  - time_range: "8-18s"
    description: "Peak dance sequence, multiple angles"
    observed_elements:
      text_overlays: ["INSANE!", "CRAZY MOVES"]
      scene_changes: 5  # Rapid cuts
      camera_movement: "dynamic"
    energy_level: "very_high"
    notes: "This is the climax moment"
    
  - time_range: "18-30s"
    description: "Tutorial portion, explaining moves"
    observed_elements:
      speech_transcript: "First you do this, then..."
      gestures: ["pointing", "demonstrating"]
      face_visible: true
    energy_level: "medium"
    
  - time_range: "30-42.5s"
    description: "Winding down, preparing for CTA"
    observed_elements:
      scene_changes: 1
    energy_level: "low"

# Closing (last 3s) - Fixed window  
closing:
  time_range: "42.5-45.5s"
  description: "Strong CTA with pointing"
  observed_elements:
    text_overlays: ["FOLLOW FOR PART 2"]
    speech_transcript: "Follow for more"
    gestures: ["pointing_at_camera"]
    face_visible: true
    face_emotion: "happy"
  has_cta: true
  cta_type: "follow"
  energy_level: "high"
```

### Annotation Guidelines

1. **Time Ranges**: Use natural boundaries based on content changes
2. **Observed Elements**: Only annotate what you can clearly see/hear
3. **Energy Levels**: Subjective but consistent scale (low/medium/high/very_high)
4. **Optional Fields**: Not every field needed for every segment
5. **Overlap Handling**: System will map to actual ML windows automatically

## Validation Logic

### Mapping Process

```python
def map_annotations_to_ml_windows(annotations, ml_windows):
    """
    Maps human time-range annotations to ML-detected windows
    
    Example:
    - Human: "8-18s" (peak dance)
    - ML Windows: segment_1 (3-11s), segment_2 (11-19s), segment_3 (19-27s)
    - Mapping: 
      - segment_1 gets 3s of human annotation (8-11s)
      - segment_2 gets 7s of human annotation (11-18s)
      - Weighted by overlap percentage
    """
```

### Success Criteria & Metrics

Test success is measured across multiple dimensions with weighted priorities:

```python
success_criteria = {
    # Priority 1: Key Moment Detection (40% weight)
    "key_moments": {
        "hook_effectiveness": 0.85,      # Did we identify attention-grabbing hook?
        "climax_detection": 0.80,        # Found peak energy/engagement moment?
        "cta_identification": 0.95,      # Detected call-to-action in closing?
        "weight": 0.40
    },
    
    # Priority 2: ML Detection Accuracy (35% weight)
    "ml_accuracy": {
        "text_overlay_recall": 0.80,     # 80% of annotated text found
        "face_visibility_accuracy": 0.85, # Face detection alignment
        "scene_change_accuracy": 0.75,   # Within ±1 of annotated count
        "speech_coverage_accuracy": 0.70, # Speech detection alignment
        "weight": 0.35
    },
    
    # Priority 3: Temporal Window Validity (25% weight)
    "temporal_structure": {
        "window_rhythm_captured": 0.80,  # Windows align with content flow
        "segment_boundaries_logical": 0.75, # Segments split at natural points
        "pacing_pattern_detected": 0.70, # Fast/slow sections identified
        "weight": 0.25
    },
    
    # Overall Success Thresholds
    "pass_thresholds": {
        "minimum_overall": 0.75,         # 75% weighted score to pass
        "critical_failures": [           # Any of these = automatic fail
            "cta_identification < 0.50",  # Missing most CTAs
            "hook_effectiveness < 0.50",  # Missing most hooks
            "ml_accuracy_avg < 0.60"      # ML performing poorly overall
        ]
    }
}
```

### Validation Report Structure

```markdown
# Validation Report: test_video_01

## Summary
- Overall Accuracy: 84%
- Best Performing: Closing Window (92%)
- Needs Improvement: Scene Change Detection (80%)

## Hook Window (0-3s)
✅ **Matched Elements:**
- Face visibility: Correctly detected
- Text overlays: 2/2 detected ("WATCH THIS!", "You won't believe")
- Energy level: High (matched)

❌ **Discrepancies:**
- Gesture "pointing_up" not detected
- Extra object detected: "phone" (not in annotations)

## Middle Window (3-42.5s)
### Segment Mapping
- Human annotation "3-8s" → ML segment_1 (3-11s)
- Human annotation "8-18s" → ML segment_1 (partial) + segment_2

### Key Findings
- Peak energy moment detected at 12s (human: ~13s) ✅
- Scene changes: ML detected 8, human annotated 8 ✅
- Text overlay accuracy: 4/5 detected (80%)

## Closing Window (42.5-45.5s)
✅ **Perfect Match:**
- CTA correctly identified
- Text "FOLLOW FOR PART 2" detected
- Pointing gesture recognized

## Recommendations
1. Improve gesture detection sensitivity
2. Add object filtering for common false positives
3. Scene change detection working well
```

## Command-Line Interface

### Basic Usage
```bash
# Simple test with video only
python scripts/testing/temporal_test_runner.py path/to/video.mp4

# Test with human annotations
python scripts/testing/temporal_test_runner.py path/to/video.mp4 \
    --annotations path/to/annotations.yaml

# Specify output directory
python scripts/testing/temporal_test_runner.py path/to/video.mp4 \
    --annotations path/to/annotations.yaml \
    --output-dir custom_test_results/

# Debug mode (saves intermediate files)
python scripts/testing/temporal_test_runner.py path/to/video.mp4 \
    --annotations path/to/annotations.yaml \
    --debug
```

### Output Examples
```
🎬 Processing: test_video_01.mp4
📊 Duration: 45.5 seconds
🔄 Temporal Windows:
   - Hook: 0-3s
   - Middle: 3-42.5s (4 segments)
   - Closing: 42.5-45.5s

✅ ML Analysis Complete
📝 Loading human annotations...
🔍 Validating results...

Results Summary:
================
Overall Accuracy: 84%
- Hook Window: 88% match
- Middle Window: 76% match  
- Closing Window: 92% match

Full report saved to: test_results/test_video_01/validation_summary.md
```

## Testing Strategy & Frequency

### Usage Pattern
- **Initial Development Phase**: Intensive daily testing during temporal windows implementation
- **Ongoing**: Run for major updates to ML models or temporal window logic
- **Not for**: Every commit or minor code changes
- **Frequency**: 
  - Now through MVP: Daily/multiple times per day
  - Post-MVP: Weekly or when changing core logic

### Test Execution Workflow
1. **Major ML Change** → Run golden test suite
2. **Window Logic Update** → Run all duration bucket tests
3. **New ML Model** → Full validation suite
4. **Bug Fix** → Run affected duration bucket only

## MVP Scope

### MVP Target: Full Validation Framework
- **Complete system from day one**: Not iterating through partial solutions
- **Rationale**: Need comprehensive testing during temporal windows development
- **All components included**: ML processing, annotation parsing, validation, reporting

## Implementation Phases

### Phase 1: Full System Build (Days 1-5)
**Build everything together for immediate use:**
- [ ] Create temporal_test_runner.py with all components
- [ ] Integrate with temporal_compute.py for ML analysis
- [ ] YAML parser for raw annotation format
- [ ] Two-stage annotation processing pipeline
- [ ] AnnotationMapper with time-range to window mapping
- [ ] Validation engine with weighted metrics
- [ ] Report generation (JSON + Markdown)
- [ ] Command-line interface with all options

### Phase 2: Golden Test Videos (Days 6-7)
- [ ] Download/prepare golden videos per criteria
- [ ] Create first set of annotations (2-3 videos)
- [ ] Run initial validation tests
- [ ] Refine parser based on real usage

### Phase 3: Refinement (Week 2)
- [ ] Adjust validation metrics based on initial results
- [ ] Complete golden video annotations (8-12 total)
- [ ] Establish baseline accuracy metrics
- [ ] Document known gaps and limitations

## Success Criteria

1. **Functionality**
   - Successfully processes local MP4 files
   - Generates valid temporal_unified.json
   - Handles videos of any duration (6s to 120s)

2. **Accuracy Validation**
   - Correctly maps human annotations to ML windows
   - Produces meaningful match scores
   - Identifies specific gaps in ML detection

3. **Developer Experience**
   - Single command to run test
   - Clear, actionable output
   - Debug mode for troubleshooting
   - Fast execution (<60s per video)

4. **Reporting**
   - Human-readable summary
   - Detailed JSON for programmatic analysis
   - Specific recommendations for improvements

## Key Decisions Summary

Based on our Q&A session, these are the finalized decisions:

1. **Primary Goal**: Validate BOTH ML detection accuracy AND temporal window calculations (Option C)
2. **Ground Truth**: Comprehensive capture of everything visible/audible with 5-minute annotation limit (Option A)
3. **Integration**: Use EXACT same ML pipeline as production, skip only video acquisition (Option A)
4. **Output Format**: Identical to production temporal_unified.json, no test metadata mixed in (Option A)
5. **Success Metrics**: Weighted multi-dimensional - Key moments (40%), ML accuracy (35%), Temporal structure (25%) (Option D)
6. **Annotation Strategy**: Golden test videos only, 2-3 per duration bucket (Option B)
7. **Test Frequency**: Major updates only, intensive during initial development (Option B)
8. **Annotator**: Single annotator (Jorge) for consistency (Option A)
9. **MVP Scope**: Full validation framework from day one (Option C)
10. **Video Source**: Real TikTok videos downloaded locally (Option A)

## Remaining Open Questions

1. **Parser Rules**: What specific shortcuts should the raw annotation parser support?
2. **Validation Thresholds**: Should pass/fail thresholds be configurable per test?
3. **Baseline Establishment**: How many test runs before we have reliable baseline metrics?
4. **Edge Case Handling**: How to handle videos that break expected patterns?

## Next Steps

1. **Critique this HLD**: Identify gaps, overcomplications, or missing requirements
2. **Finalize annotation format**: Ensure it covers all needed test cases  
3. **Build Phase 1**: Get basic functionality working
4. **Create test videos**: Curate diverse test set (short, long, simple, complex)
5. **Begin annotation**: Create ground truth for initial test videos

## 16.09 Brainstorm

### Context from Our Learnings

From building `test_temporal_compute_v2.py`, we learned:
1. **Reuse production components**: Call actual functions, don't duplicate logic
2. **MLAnalysisResult wrapper needed**: TimelineBuilder expects specific object format
3. **Mixed strategy is real**: Timeline entries + raw ML data both needed
4. **Integration point matters**: We inject at TimelineBuilder level for cached ML outputs. This will not work for this test, as we need to inject it at the beginning of the flow

### Architecture Alternatives for ML Service Testing

#### Alternative 1: Full Pipeline with Local Video (Most Authentic)
```python
class LocalVideoTestRunner:
    def process_local_video(self, video_path):
        # Use REAL VideoAnalyzer to run all ML services
        video_analyzer = VideoAnalyzer()
        ml_results = await video_analyzer.analyze_video(video_path)
        
        # Use REAL TimelineBuilder
        timeline_builder = TimelineBuilder()
        unified_analysis = timeline_builder.build_timeline(
            video_id, metadata, ml_results
        )
        
        # Use REAL compute_temporal_windows
        return compute_temporal_windows(unified_analysis.to_dict())
```
**Pros**: 
- Exactly mirrors production after video download
- Tests actual ML service integration
- No mock objects needed

**Cons**: 
- SLOW (3-5 minutes per video for ML services)
- Requires GPU for some services
- Can't test with different ML outputs easily

#### Alternative 2: Hybrid with Optional ML Cache (Pragmatic)
```python
class HybridTestRunner:
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.cache_dir = Path("ml_cache")
    
    def process_video(self, video_path):
        video_id = self._get_video_id(video_path)
        
        if self.use_cache and self._cache_exists(video_id):
            # Load from cache (like our test_temporal_compute_v2.py)
            ml_results = self._load_cached_ml_results(video_id)
        else:
            # Run actual ML services
            video_analyzer = VideoAnalyzer()
            ml_results = await video_analyzer.analyze_video(video_path)
            
            if self.use_cache:
                self._save_to_cache(video_id, ml_results)
        
        # Continue with real pipeline
        timeline_builder = TimelineBuilder()
        unified_analysis = timeline_builder.build_timeline(...)
        return compute_temporal_windows(unified_analysis.to_dict())
```
**Pros**: 
- Fast iteration after first run
- Can force fresh ML analysis when needed
- Reuses production components
- Good for regression testing

**Cons**: 
- First run still slow
- Cache management complexity
- Cache can become stale

#### Alternative 3: VideoAnalyzer with Pluggable ML Services
```python
class TestableVideoAnalyzer(VideoAnalyzer):
    """Extends VideoAnalyzer to support loading cached ML results"""
    
    def __init__(self, ml_cache_dir=None):
        super().__init__()
        self.ml_cache_dir = ml_cache_dir
    
    async def _run_yolo(self, video_id, video_path):
        if self.ml_cache_dir:
            cached = self._load_cached("yolo", video_id)
            if cached:
                return cached
        return await super()._run_yolo(video_id, video_path)
    
    # Similar for other ML services...
```
**Pros**: 
- Minimal changes to production code
- Can mix cached and fresh ML results
- Maintains MLAnalysisResult format

**Cons**: 
- Requires modifying VideoAnalyzer
- Still need to manage cache

#### Alternative 4: Mock ML Services at Runtime
```python
class MockMLServices:
    """Replaces ml_services module with cached results"""
    
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
    
    async def run_yolo_detection(self, video_path, output_dir):
        # Return cached YOLO results
        video_id = self._extract_video_id(video_path)
        return self._load_json(f"yolo/{video_id}.json")
    
    # Similar for other services...

# Monkey-patch for testing
video_analyzer.ml_services = MockMLServices("ml_cache/")
```
**Pros**: 
- No production code changes
- Complete control over ML outputs
- Can simulate failures

**Cons**: 
- Monkey-patching feels fragile
- Need to maintain mock interface

#### Alternative 5: Command-Line Flag in Runner (Cleanest)
Modify `rumiai_runner.py` to support:
```bash
python scripts/rumiai_runner.py local_video.mp4 --use-ml-cache
```

```python
# In rumiai_runner.py
if args.use_ml_cache:
    ml_results = load_ml_results_from_cache(video_id)
else:
    ml_results = await video_analyzer.analyze_video(video_path)

# Continue with normal pipeline
unified_analysis = timeline_builder.build_timeline(...)
```
**Pros**: 
- Part of production runner
- Clear separation of concerns
- Easy to switch between modes

**Cons**: 
- Modifies production runner
- Cache loading logic in production code

### Recommendation Based on LocalTestv2.md Goals

Given the requirements in LocalTestv2.md:
- Need to validate **both ML accuracy AND temporal calculations**
- Want to process **local MP4 files**
- Need **rapid iteration** during development
- Must produce **identical output to production**

**I recommend Alternative 2: Hybrid with Optional ML Cache**

This approach:
1. **Runs full ML pipeline** when you need fresh analysis (validating ML accuracy)
2. **Uses cache** for rapid iteration (validating temporal calculations)
3. **Reuses all production components** after ML stage
4. **Matches production output exactly**

Implementation would be:
```python
# For first run or ML validation
python temporal_test_runner.py video.mp4 --no-cache --annotations human.yaml

# For rapid temporal logic testing
python temporal_test_runner.py video.mp4 --use-cache --annotations human.yaml
```

This is essentially our `test_temporal_compute_v2.py` **plus** the ability to run real ML services, giving us the best of both worlds:
- Fast iteration when testing temporal logic
- Full ML validation when needed
- Complete reuse of production components

The key insight from our learnings: **The injection point is after ML services but before TimelineBuilder**, and we can choose whether to run ML services or load cached results at that point.