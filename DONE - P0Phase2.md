# Phase 2: Integration Plan

## Overview
This document outlines the complete implementation plan for Phase 2: Integration of the temporal_compute.py module into the existing RumiAI pipeline.

## Phase 2 Tasks
- [ ] Update rumiai_runner.py to use new temporal_compute
- [ ] Remove old compute functions from precompute_functions.py  
- [ ] Test with real video files (3s, 5s, 10s, 30s, 120s)
- [ ] Verify JSON output structure matches specification
- [ ] Validate element_count excludes scene_changes
- [ ] Confirm speech coverage filters music correctly

---

## Task 1: Update rumiai_runner.py

### Current State Analysis
The rumiai_runner.py currently:
1. Loops through 7 compute functions (COMPUTE_FUNCTIONS dictionary)
2. Saves 3 JSON files per function (COMPLETE, ML, RESULT)
3. Creates 7 folders under insights/{video_id}/

### Integration Code

```python
# scripts/rumiai_runner.py - MODIFIED VERSION

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio

# Import the new temporal compute module
from rumiai_v2.processors.temporal_compute import (
    compute_temporal_windows,
    save_temporal_unified
)

class VideoProcessor:
    """Enhanced video processor with temporal windows"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def process_video(self, video_path: str, video_id: str) -> Dict[str, Any]:
        """
        Process video with new temporal windows architecture
        
        Args:
            video_path: Path to video file
            video_id: Unique identifier for video
            
        Returns:
            Temporal unified analysis results
        """
        try:
            # Step 1: Extract all timeline data
            timelines = await self.extract_timelines(video_path, video_id)
            
            # Step 2: Get video metadata
            video_metadata = await self.extract_metadata(video_path, video_id)
            
            # Step 3: Get speech segments
            speech_segments = await self.extract_speech(video_path, video_id)
            
            # Step 4: Get audio path (for energy calculation)
            audio_path = await self.extract_audio(video_path, video_id)
            
            # Step 5: Compute temporal windows (NEW)
            result = compute_temporal_windows(
                timelines=timelines,
                video_metadata=video_metadata,
                speech_segments=speech_segments,
                audio_path=audio_path
            )
            
            # Step 6: Save unified JSON (NEW)
            output_path = Path(f"insights/{video_id}/temporal_unified.json")
            save_temporal_unified(result, output_path)
            
            self.logger.info(f"✓ Completed temporal processing for {video_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"✗ Failed to process {video_id}: {e}")
            raise
    
    async def extract_timelines(self, video_path: str, video_id: str) -> Dict[str, Any]:
        """Extract all timeline data from existing services"""
        
        # Import existing services
        from rumiai_v2.ml_services.object_detection_service import get_object_detection_service
        from rumiai_v2.ml_services.scene_detection_service import get_scene_detection_service
        from rumiai_v2.ml_services.text_extraction_service import get_text_extraction_service
        from rumiai_v2.ml_services.expression_recognition_service import get_expression_recognition_service
        from rumiai_v2.ml_services.gesture_recognition_service import get_gesture_recognition_service
        from rumiai_v2.ml_services.sticker_detection_service import get_sticker_detection_service
        from rumiai_v2.ml_services.person_framing_service import get_person_framing_service
        from rumiai_v2.ml_services.gaze_detection_service import get_gaze_detection_service
        
        self.logger.info(f"Extracting timelines for {video_id}")
        
        timelines = {}
        
        # Text overlay timeline
        text_service = get_text_extraction_service()
        text_result = await text_service.extract(video_path)
        timelines['text_overlay_timeline'] = text_result.get('timeline', [])
        
        # Sticker timeline
        sticker_service = get_sticker_detection_service()
        sticker_result = await sticker_service.detect(video_path)
        timelines['sticker_timeline'] = sticker_result.get('timeline', [])
        
        # Object timeline
        object_service = get_object_detection_service()
        object_result = await object_service.detect(video_path)
        timelines['object_timeline'] = object_result.get('timeline', [])
        
        # Gesture timeline
        gesture_service = get_gesture_recognition_service()
        gesture_result = await gesture_service.recognize(video_path)
        timelines['gesture_timeline'] = gesture_result.get('timeline', [])
        
        # Expression timeline
        expression_service = get_expression_recognition_service()
        expression_result = await expression_service.recognize(video_path)
        timelines['expression_timeline'] = expression_result.get('timeline', [])
        
        # Scene boundaries
        scene_service = get_scene_detection_service()
        scene_result = await scene_service.detect(video_path)
        timelines['scene_boundaries'] = scene_result.get('scene_boundaries', [])
        
        # Person timeline (face detection)
        person_service = get_person_framing_service()
        person_result = await person_service.analyze(video_path)
        timelines['personTimeline'] = person_result.get('personTimeline', {})
        
        # Gaze timeline
        gaze_service = get_gaze_detection_service()
        gaze_result = await gaze_service.detect(video_path)
        timelines['gaze_timeline'] = gaze_result.get('gaze_timeline', {})
        
        # Camera distance timeline
        timelines['camera_distance_timeline'] = person_result.get('camera_distance_timeline', {})
        
        # Framing timeline (if available)
        timelines['framing_timeline'] = person_result.get('framing_timeline', {})
        
        return timelines
    
    async def extract_metadata(self, video_path: str, video_id: str) -> Dict[str, Any]:
        """Extract video metadata"""
        
        from rumiai_v2.utils.video_utils import get_video_metadata
        
        metadata = get_video_metadata(video_path)
        
        return {
            'video_id': video_id,
            'duration': metadata.get('duration', 0),
            'publish_hour': metadata.get('publish_hour', 0),
            'caption_length': len(metadata.get('caption', '')),
            'hashtag_count': len(metadata.get('hashtags', [])),
            'has_captions': metadata.get('has_captions', False),
            'has_soundtrack': metadata.get('has_soundtrack', False),
            'view_count': metadata.get('view_count', 0),
            'like_count': metadata.get('like_count', 0),
            'comment_count': metadata.get('comment_count', 0),
            'share_count': metadata.get('share_count', 0),
        }
    
    async def extract_speech(self, video_path: str, video_id: str) -> List[Dict]:
        """Extract speech segments using whisper.cpp"""
        
        from rumiai_v2.api.whisper_cpp_service import get_whisper_cpp_transcriber
        
        transcriber = get_whisper_cpp_transcriber()
        result = await transcriber.transcribe_with_preprocessing(
            audio_path=Path(video_path),
            video_id=video_id
        )
        
        return result.get('segments', [])
    
    async def extract_audio(self, video_path: str, video_id: str) -> Optional[Path]:
        """Extract audio for energy calculation"""
        
        from rumiai_v2.api.shared_audio_extractor import SharedAudioExtractor
        
        # Get the shared audio file
        audio_path = await SharedAudioExtractor.extract_once(
            video_path, 
            video_id, 
            service_name="temporal_compute"
        )
        
        return audio_path

# ============== MAIN RUNNER ==============

async def main():
    """Main entry point for rumiai_runner"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='RumiAI Video Processor')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--video-id', help='Video ID', required=True)
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Process video
    processor = VideoProcessor()
    result = await processor.process_video(args.video_path, args.video_id)
    
    print(f"✓ Processing complete. Output: insights/{args.video_id}/temporal_unified.json")
    
if __name__ == "__main__":
    asyncio.run(main())
```

---

## Task 2: Remove Old Compute Functions

### Files to Modify

```python
# rumiai_v2/processors/precompute_functions.py - MODIFICATIONS

# REMOVE these functions:
def compute_visual_overlay_metrics(video_id: str, insights_path: Path) -> Dict:
    """REMOVE - replaced by temporal_compute"""
    pass

def compute_person_framing_metrics(video_id: str, insights_path: Path) -> Dict:
    """REMOVE - replaced by temporal_compute"""
    pass

def compute_emotional_journey_metrics(video_id: str, insights_path: Path) -> Dict:
    """REMOVE - replaced by temporal_compute"""
    pass

def compute_audio_dynamics_metrics(video_id: str, insights_path: Path) -> Dict:
    """REMOVE - replaced by temporal_compute"""
    pass

def compute_creative_density_metrics(video_id: str, insights_path: Path) -> Dict:
    """REMOVE - replaced by temporal_compute"""
    pass

def compute_virality_mechanics_metrics(video_id: str, insights_path: Path) -> Dict:
    """REMOVE - replaced by temporal_compute"""
    pass

def compute_scene_dynamics_metrics(video_id: str, insights_path: Path) -> Dict:
    """REMOVE - replaced by temporal_compute"""
    pass

# REMOVE the COMPUTE_FUNCTIONS dictionary:
COMPUTE_FUNCTIONS = {}  # Empty - no longer used

# ADD migration notice:
def get_compute_functions():
    """
    DEPRECATED: Use temporal_compute.compute_temporal_windows instead
    
    Migration guide:
    Old: result = compute_visual_overlay_metrics(video_id, path)
    New: result = compute_temporal_windows(timelines, metadata, speech)
    """
    raise DeprecationWarning(
        "Old compute functions have been replaced by temporal_compute. "
        "Use compute_temporal_windows() instead."
    )
```

---

## Task 3: Test Script for Real Videos

```python
# test_real_videos.py

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Test video configurations
TEST_VIDEOS = [
    {"path": "test_videos/3s_video.mp4", "id": "test_3s", "expected_duration": 3},
    {"path": "test_videos/5s_video.mp4", "id": "test_5s", "expected_duration": 5},
    {"path": "test_videos/10s_video.mp4", "id": "test_10s", "expected_duration": 10},
    {"path": "test_videos/30s_video.mp4", "id": "test_30s", "expected_duration": 30},
    {"path": "test_videos/120s_video.mp4", "id": "test_120s", "expected_duration": 120},
]

async def test_video(video_config: Dict) -> Dict[str, Any]:
    """Test a single video"""
    
    from scripts.rumiai_runner import VideoProcessor
    
    processor = VideoProcessor()
    
    print(f"\nTesting {video_config['id']}...")
    
    try:
        # Process video
        result = await processor.process_video(
            video_config['path'], 
            video_config['id']
        )
        
        # Validate duration
        if abs(result['duration'] - video_config['expected_duration']) > 1:
            print(f"  ⚠ Duration mismatch: got {result['duration']}, expected {video_config['expected_duration']}")
        
        # Check output file
        output_path = Path(f"insights/{video_config['id']}/temporal_unified.json")
        if not output_path.exists():
            raise FileNotFoundError(f"Output file not created: {output_path}")
        
        # Load and validate JSON
        with open(output_path) as f:
            data = json.load(f)
        
        # Validate structure
        assert 'temporal_windows' in data
        assert 'global_metadata' in data
        assert 'outcomes' in data
        
        print(f"  ✓ {video_config['id']} passed")
        return {"status": "passed", "result": result}
        
    except Exception as e:
        print(f"  ✗ {video_config['id']} failed: {e}")
        return {"status": "failed", "error": str(e)}

async def run_all_tests():
    """Run all video tests"""
    
    print("=" * 60)
    print("REAL VIDEO INTEGRATION TESTS")
    print("=" * 60)
    
    results = []
    for video_config in TEST_VIDEOS:
        result = await test_video(video_config)
        results.append(result)
    
    # Summary
    passed = sum(1 for r in results if r['status'] == 'passed')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    asyncio.run(run_all_tests())
```

---

## Task 4: Validation Script

```python
# validate_temporal_output.py

import json
from pathlib import Path
from typing import Dict, Any, List

def validate_json_structure(json_path: Path) -> List[str]:
    """Validate temporal unified JSON structure"""
    
    errors = []
    
    with open(json_path) as f:
        data = json.load(f)
    
    # Check top-level keys
    required_keys = ['video_id', 'duration', 'temporal_windows', 'global_metadata', 'outcomes']
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")
    
    # Check temporal windows
    if 'temporal_windows' in data:
        windows = data['temporal_windows']
        
        # Check hook (always required for videos >= 3s)
        if data.get('duration', 0) >= 3 and 'hook' not in windows:
            errors.append("Missing hook window for video >= 3s")
        
        # Check hook features
        if 'hook' in windows:
            hook = windows['hook']
            required_hook_features = [
                'hook_element_count', 'hook_text_count', 'hook_speech_coverage',
                'hook_avg_density', 'hook_word_count'
            ]
            for feature in required_hook_features:
                if feature not in hook:
                    errors.append(f"Missing hook feature: {feature}")
    
    return errors

def validate_element_count(json_path: Path) -> bool:
    """Validate that element_count excludes scene_changes"""
    
    with open(json_path) as f:
        data = json.load(f)
    
    if 'temporal_windows' in data and 'hook' in data['temporal_windows']:
        hook = data['temporal_windows']['hook']
        
        # element_count should equal sum of 5 visual types
        element_count = hook.get('hook_element_count', 0)
        calculated = (
            hook.get('hook_text_count', 0) +
            hook.get('hook_sticker_count', 0) +
            hook.get('hook_object_count', 0) +
            hook.get('hook_gesture_count', 0) +
            hook.get('hook_expression_count', 0)
        )
        
        if element_count != calculated:
            print(f"✗ Element count mismatch: {element_count} != {calculated}")
            print(f"  Scene changes should NOT be included")
            return False
        
        print(f"✓ Element count correct: {element_count} (excludes scene changes)")
        return True
    
    return False

def validate_speech_coverage(json_path: Path) -> bool:
    """Validate that speech coverage filters music"""
    
    with open(json_path) as f:
        data = json.load(f)
    
    # Check if speech coverage is between 0 and 1
    valid = True
    
    def check_coverage(window_name: str, coverage_value: float):
        nonlocal valid
        if not (0 <= coverage_value <= 1):
            print(f"✗ Invalid {window_name} speech coverage: {coverage_value}")
            valid = False
        else:
            print(f"✓ {window_name} speech coverage valid: {coverage_value}")
    
    if 'temporal_windows' in data:
        windows = data['temporal_windows']
        
        if 'hook' in windows:
            check_coverage('hook', windows['hook'].get('hook_speech_coverage', 0))
        
        if 'middle' in windows:
            check_coverage('middle', windows['middle'].get('middle_speech_coverage', 0))
        
        if 'closing' in windows:
            check_coverage('closing', windows['closing'].get('closing_speech_coverage', 0))
    
    return valid

def run_all_validations(video_id: str):
    """Run all validation checks"""
    
    json_path = Path(f"insights/{video_id}/temporal_unified.json")
    
    print(f"\n{'=' * 60}")
    print(f"VALIDATING: {video_id}")
    print(f"{'=' * 60}")
    
    # Structure validation
    print("\n1. JSON Structure Validation:")
    errors = validate_json_structure(json_path)
    if errors:
        for error in errors:
            print(f"  ✗ {error}")
    else:
        print("  ✓ All required fields present")
    
    # Element count validation
    print("\n2. Element Count Validation:")
    validate_element_count(json_path)
    
    # Speech coverage validation
    print("\n3. Speech Coverage Validation:")
    validate_speech_coverage(json_path)
    
    print(f"\n{'=' * 60}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_temporal_output.py <video_id>")
        sys.exit(1)
    
    run_all_validations(sys.argv[1])
```

---

## Task 5 & 6: Already Validated

From our Phase 1 testing, we've already confirmed:

✅ **Task 5: element_count excludes scene_changes**
- Verified in test output: element_count = sum of 5 visual types
- Scene changes tracked separately as scene_count

✅ **Task 6: Speech coverage filters music correctly**
- Test showed "♪♪♪" segment was filtered out
- Speech coverage = 0.67 for hook (2s of speech in 3s window)
- Music filtering working with is_likely_music() function

---

## Integration Checklist

### Before Integration
- [ ] Backup existing rumiai_runner.py
- [ ] Backup precompute_functions.py
- [ ] Create test video set (3s, 5s, 10s, 30s, 120s)

### During Integration
- [ ] Update rumiai_runner.py with new code
- [ ] Comment out old compute functions
- [ ] Run test suite with sample data
- [ ] Run test suite with real videos

### After Integration
- [ ] Verify all JSON outputs created
- [ ] Run validation script on outputs
- [ ] Check file sizes are reasonable
- [ ] Compare with old outputs for consistency
- [ ] Document any discrepancies

### Rollback Plan
If issues occur:
1. Restore backed up rumiai_runner.py
2. Restore backed up precompute_functions.py  
3. Keep temporal_compute.py for future use
4. Document issues encountered

---

## Success Criteria

Phase 2 is complete when:
1. ✅ rumiai_runner.py successfully uses temporal_compute
2. ✅ Old compute functions are deprecated/removed
3. ✅ All test videos (3s-120s) process without errors
4. ✅ JSON structure matches specification
5. ✅ element_count validation passes
6. ✅ Speech coverage validation passes
7. ✅ Output files are created in correct location
8. ✅ No regression in functionality

---

## Notes

- The integration maintains backward compatibility by keeping the same insights/{video_id}/ structure
- The new temporal_unified.json replaces 21 separate JSON files (7 functions × 3 formats)
- Performance should improve due to single-pass processing
- Memory usage should decrease due to unified computation