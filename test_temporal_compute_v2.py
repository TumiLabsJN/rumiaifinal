#!/usr/bin/env python3
"""
Test script for temporal compute using existing ML outputs.
This script follows the EXACT same flow as the production pipeline,
using real TimelineBuilder and compute_temporal_windows functions.

Option 3 Architecture:
1. Load ML outputs from disk (manual but unavoidable)
2. Call real TimelineBuilder.build_timeline()
3. Call real compute_temporal_windows()

This ensures any fixes to production code automatically apply to tests.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Import ACTUAL production components - not duplicating any logic
from rumiai_v2.processors import TimelineBuilder
from rumiai_v2.processors.temporal_compute import compute_temporal_windows
from rumiai_v2.core.models.analysis import MLAnalysisResult


class MLOutputLoader:
    """
    Loads pre-computed ML outputs from disk.
    This is the ONLY manual part - everything else uses production code.
    """
    
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.ml_results = {}
        self.metadata = {}
        
    def load_all_ml_outputs(self) -> Dict[str, Any]:
        """
        Load all ML service outputs from their respective directories.
        Returns ml_results dict in the exact format VideoAnalyzer would produce.
        """
        print(f"\n📂 Loading ML outputs for video: {self.video_id}")
        
        # Load each ML service output
        self._load_yolo()
        self._load_whisper()
        self._load_ocr()
        self._load_mediapipe()
        self._load_scene_detection()
        self._load_audio_energy()
        self._load_emotion_detection()
        
        # Load metadata
        self._load_metadata()
        
        return self.ml_results
    
    def _load_yolo(self):
        """Load YOLO object detection results."""
        paths = [
            Path(f"object_detection_outputs/{self.video_id}/{self.video_id}_yolo_detections.json"),
            Path(f"object_detection_outputs/{self.video_id}/{self.video_id}_objects.json"),
        ]
        
        for path in paths:
            if path.exists():
                print(f"  ✓ YOLO: {path}")
                with open(path) as f:
                    data = json.load(f)
                self.ml_results['yolo'] = MLAnalysisResult(
                    model_name='yolo',
                    model_version='v8',
                    success=True,
                    data=data
                )
                return
        
        print(f"  ✗ YOLO: Not found, using empty results")
        self.ml_results['yolo'] = MLAnalysisResult(
            model_name='yolo',
            model_version='v8',
            success=True,
            data={"objectAnnotations": []}
        )
    
    def _load_whisper(self):
        """Load Whisper speech transcription results."""
        paths = [
            Path(f"speech_transcriptions/{self.video_id}_whisper.json"),
            Path(f"whisper_outputs/{self.video_id}/{self.video_id}_transcription.json"),
            Path(f"speech_transcription_outputs/{self.video_id}/{self.video_id}_whisper.json"),
        ]
        
        for path in paths:
            if path.exists():
                print(f"  ✓ Whisper: {path}")
                with open(path) as f:
                    data = json.load(f)
                self.ml_results['whisper'] = MLAnalysisResult(
                    model_name='whisper',
                    model_version='base',
                    success=True,
                    data=data
                )
                return
        
        print(f"  ✗ Whisper: Not found, using empty results")
        self.ml_results['whisper'] = MLAnalysisResult(
            model_name='whisper',
            model_version='base',
            success=True,
            data={"segments": []}
        )
    
    def _load_ocr(self):
        """Load OCR text detection results."""
        paths = [
            Path(f"ocr_outputs/{self.video_id}/{self.video_id}_ocr.json"),
            Path(f"text_detection_outputs/{self.video_id}/{self.video_id}_text.json"),
        ]
        
        for path in paths:
            if path.exists():
                print(f"  ✓ OCR: {path}")
                with open(path) as f:
                    data = json.load(f)
                self.ml_results['ocr'] = MLAnalysisResult(
                    model_name='ocr',
                    model_version='v1',
                    success=True,
                    data=data
                )
                return
        
        print(f"  ✗ OCR: Not found, using empty results")
        self.ml_results['ocr'] = MLAnalysisResult(
            model_name='ocr',
            model_version='v1',
            success=True,
            data={"textAnnotations": [], "stickers": []}
        )
    
    def _load_mediapipe(self):
        """Load MediaPipe human analysis results."""
        paths = [
            Path(f"human_analysis_outputs/{self.video_id}/{self.video_id}_human_analysis.json"),
            Path(f"mediapipe_outputs/{self.video_id}/{self.video_id}_mediapipe.json"),
        ]
        
        for path in paths:
            if path.exists():
                print(f"  ✓ MediaPipe: {path}")
                with open(path) as f:
                    data = json.load(f)
                self.ml_results['mediapipe'] = MLAnalysisResult(
                    model_name='mediapipe',
                    model_version='v1',
                    success=True,
                    data=data
                )
                return
        
        print(f"  ✗ MediaPipe: Not found, using empty results")
        self.ml_results['mediapipe'] = MLAnalysisResult(
            model_name='mediapipe',
            model_version='v1',
            success=True,
            data={
                "poses": [],
                "faces": [],
                "hands": [],
                "gaze": [],
                "gestures": []
            }
        )
    
    def _load_scene_detection(self):
        """Load scene detection results."""
        paths = [
            Path(f"scene_detection_outputs/{self.video_id}/{self.video_id}_scenes.json"),
            Path(f"pyscenedetect_outputs/{self.video_id}/{self.video_id}_scenes.json"),
        ]
        
        for path in paths:
            if path.exists():
                print(f"  ✓ Scene Detection: {path}")
                with open(path) as f:
                    data = json.load(f)
                self.ml_results['scene_detection'] = MLAnalysisResult(
                    model_name='scene_detection',
                    model_version='v1',
                    success=True,
                    data=data
                )
                return
        
        print(f"  ✗ Scene Detection: Not found, using empty results")
        self.ml_results['scene_detection'] = MLAnalysisResult(
            model_name='scene_detection',
            model_version='v1',
            success=True,
            data={"scenes": [], "scene_changes": []}
        )
    
    def _load_audio_energy(self):
        """Load audio energy analysis results."""
        paths = [
            Path(f"audio_energy_outputs/{self.video_id}/{self.video_id}_energy.json"),
            Path(f"audio_analysis_outputs/{self.video_id}/{self.video_id}_audio.json"),
        ]
        
        for path in paths:
            if path.exists():
                print(f"  ✓ Audio Energy: {path}")
                with open(path) as f:
                    data = json.load(f)
                self.ml_results['audio_energy'] = MLAnalysisResult(
                    model_name='audio_energy',
                    model_version='v1',
                    success=True,
                    data=data
                )
                return
        
        print(f"  ✗ Audio Energy: Not found, using empty results")
        self.ml_results['audio_energy'] = MLAnalysisResult(
            model_name='audio_energy',
            model_version='v1',
            success=True,
            data={
                "rms_frames": [],
                "frames_per_second": 30.0,
                "energy_variance": 0.0,
                "burst_pattern": "steady"
            }
        )
    
    def _load_emotion_detection(self):
        """Load emotion detection results."""
        paths = [
            Path(f"emotion_detection_outputs/{self.video_id}/{self.video_id}_emotions.json"),
            Path(f"feat_outputs/{self.video_id}/{self.video_id}_feat.json"),
        ]
        
        for path in paths:
            if path.exists():
                print(f"  ✓ Emotion Detection: {path}")
                with open(path) as f:
                    data = json.load(f)
                self.ml_results['emotion_detection'] = MLAnalysisResult(
                    model_name='emotion_detection',
                    model_version='feat',
                    success=True,
                    data=data
                )
                return
        
        print(f"  ✗ Emotion Detection: Not found, using empty results")
        self.ml_results['emotion_detection'] = MLAnalysisResult(
            model_name='emotion_detection',
            model_version='feat',
            success=True,
            data={"emotions": [], "timeline": {}}
        )
    
    def _load_metadata(self):
        """Load video metadata."""
        # First check if we have a unified analysis file with duration
        unified_path = Path(f"unified_analysis/{self.video_id}.json")
        if unified_path.exists():
            with open(unified_path) as f:
                unified = json.load(f)
                duration = unified.get('duration', 35.0)
        # Otherwise try to get duration from scene detection
        elif 'scene_detection' in self.ml_results and self.ml_results['scene_detection'].success:
            scenes = self.ml_results['scene_detection'].data.get('scenes', [])
            if scenes and 'end_time' in scenes[-1]:
                duration = scenes[-1]['end_time']
            else:
                duration = 35.0  # Default
        else:
            duration = 35.0
        
        # Try to load existing metadata if available
        metadata_paths = [
            Path(f"insights/{self.video_id}_metadata.json"),
            Path(f"metadata/{self.video_id}.json"),
        ]
        
        for path in metadata_paths:
            if path.exists():
                print(f"  ✓ Metadata: {path}")
                with open(path) as f:
                    self.metadata = json.load(f)
                    self.metadata['video_duration'] = duration
                    return
        
        # Create minimal metadata if not found
        print(f"  ℹ Metadata: Using defaults")
        self.metadata = {
            'video_id': self.video_id,
            'video_duration': duration,
            'duration': duration,
            'likes': 0,
            'views': 0,
            'saves': 0,
            'shares': 0,
            'comments': 0,
            'username': 'test_user',
            'description': '',
            'create_time': '2024-01-01T00:00:00Z'
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get loaded metadata."""
        return self.metadata


def validate_temporal_windows(result: Dict[str, Any]) -> None:
    """
    Validate that all expected features are present in temporal windows.
    """
    print("\n" + "="*60)
    print("📊 FEATURE VALIDATION REPORT")
    print("="*60)
    
    # Expected features based on our implementation
    expected_features = {
        'P0 Core': ['text_count', 'sticker_count', 'object_count', 
                    'gesture_count', 'expression_count', 'scene_count', 
                    'element_count'],
        'P0 Density': ['max_density', 'min_density', 'avg_density'],
        'P0 Speech': ['speech_coverage', 'word_count'],
        'P0 Emotions': ['joy_ratio', 'sadness_ratio', 'anger_ratio', 
                       'fear_ratio', 'disgust_ratio', 'surprise_ratio', 
                       'neutral_ratio'],
        'P0 Framing': ['close_ratio', 'medium_ratio', 'wide_ratio', 'none_ratio'],
        'P0 Audio': ['energy_level', 'energy_variance', 'energy_max', 'burst_pattern'],
        'P1 Scene': ['shortest_scene', 'longest_scene'],
        'P2 Variance': ['scene_duration_variance']
    }
    
    # Check hook window
    hook = result['temporal_windows']['hook']
    print("\n🎬 Hook Window (0-3s):")
    
    for category, features in expected_features.items():
        print(f"\n{category}:")
        for feature in features:
            if feature in hook:
                value = hook[feature]
                if isinstance(value, float):
                    print(f"  ✓ {feature}: {value:.4f}")
                else:
                    print(f"  ✓ {feature}: {value}")
            else:
                print(f"  ✗ {feature}: MISSING")
    
    # Summary statistics
    all_features = [f for features in expected_features.values() for f in features]
    present = sum(1 for f in all_features if f in hook)
    total = len(all_features)
    
    print("\n" + "="*60)
    print(f"📈 SUMMARY: {present}/{total} features present ({present*100//total}%)")
    print("="*60)
    
    # Check for any unexpected features
    expected_set = set(all_features)
    actual_set = set(hook.keys()) - {'start', 'end', 'duration'}
    unexpected = actual_set - expected_set
    if unexpected:
        print(f"\n⚠️  Unexpected features found: {unexpected}")


def main():
    """
    Main test flow that mirrors production pipeline exactly.
    """
    # Get video ID from command line or use default
    video_id = sys.argv[1] if len(sys.argv) > 1 else "7430952519439846698"
    
    print("="*60)
    print("🚀 TEMPORAL COMPUTE TEST - Production Pipeline Mirror")
    print("="*60)
    print(f"Video ID: {video_id}")
    
    # Step 1: Load ML outputs from disk (only manual part)
    loader = MLOutputLoader(video_id)
    ml_results = loader.load_all_ml_outputs()
    metadata = loader.get_metadata()
    
    # Step 2: Use REAL TimelineBuilder from production
    print(f"\n🔨 Building timeline with production TimelineBuilder...")
    timeline_builder = TimelineBuilder()
    
    # This is the EXACT call from rumiai_runner.py line 270-273
    unified_analysis = timeline_builder.build_timeline(
        video_id=video_id,
        video_metadata=metadata,
        ml_results=ml_results
    )
    
    print(f"  ✓ Timeline built with {len(unified_analysis.timeline.entries)} entries")
    
    # Step 3: Use REAL compute_temporal_windows from production
    print(f"\n⚙️  Computing temporal windows with production function...")
    
    # This is the EXACT call from rumiai_runner.py line 293
    result = compute_temporal_windows(unified_analysis.to_dict())
    
    print(f"  ✓ Temporal windows computed")
    
    # Step 4: Validate results
    validate_temporal_windows(result)
    
    # Step 5: Save results for comparison
    output_path = Path(f"test_outputs/{video_id}_temporal_windows_test.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Step 6: Compare with original if exists
    original_path = Path(f"insights/{video_id}_temporal_windows.json")
    if original_path.exists():
        with open(original_path) as f:
            original = json.load(f)
        
        print(f"\n🔍 Comparing with original temporal windows...")
        
        # Compare a few key metrics
        orig_hook = original['temporal_windows']['hook']
        test_hook = result['temporal_windows']['hook']
        
        metrics_to_compare = ['element_count', 'speech_coverage', 'joy_ratio', 
                             'shortest_scene', 'scene_duration_variance']
        
        for metric in metrics_to_compare:
            orig_val = orig_hook.get(metric, 'N/A')
            test_val = test_hook.get(metric, 'N/A')
            match = "✓" if orig_val == test_val else "✗"
            print(f"  {match} {metric}: orig={orig_val}, test={test_val}")
    
    print("\n✅ Test complete!")
    print("\nThis test uses the EXACT production pipeline:")
    print("  1. ML outputs loaded from disk (only manual step)")
    print("  2. Real TimelineBuilder.build_timeline()")
    print("  3. Real compute_temporal_windows()")
    print("\nAny fixes to production code automatically apply here! 🎉")


if __name__ == "__main__":
    main()