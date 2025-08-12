#!/usr/bin/env python3
"""Test FEAT emotion detection integration with timeline builder."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from rumiai_v2.core.models import Timeline, TimelineEntry, Timestamp, MLAnalysisResult
from rumiai_v2.processors.timeline_builder import TimelineBuilder
from rumiai_v2.processors.precompute_functions import _extract_timelines_from_analysis

def test_feat_timeline_integration():
    """Test that FEAT data is properly added to timeline and extracted."""
    
    print("Testing FEAT Timeline Integration...")
    
    # 1. Create sample FEAT output data
    feat_data = {
        'emotions': [
            {
                'timestamp': 0.0,
                'emotion': 'joy',
                'confidence': 0.9,
                'all_scores': {'joy': 0.9, 'neutral': 0.1},
                'action_units': [6, 12],
                'au_intensities': {'6': 0.8, '12': 0.9}
            },
            {
                'timestamp': 1.5,
                'emotion': 'surprise',
                'confidence': 0.85,
                'all_scores': {'surprise': 0.85, 'neutral': 0.15},
                'action_units': [1, 2, 5],
                'au_intensities': {'1': 0.7, '2': 0.8, '5': 0.9}
            }
        ]
    }
    
    # 2. Test timeline builder
    print("\n1. Testing Timeline Builder...")
    timeline = Timeline(video_id='test', duration=60)
    builder = TimelineBuilder()
    
    try:
        builder._add_emotion_entries(timeline, feat_data)
        print("   ✅ Successfully added emotion entries to timeline")
    except Exception as e:
        print(f"   ❌ Failed to add emotion entries: {e}")
        return False
    
    # Verify emotion entries exist in timeline
    emotion_entries = [e for e in timeline.entries if e.entry_type == 'emotion']
    assert len(emotion_entries) == 2, f"Expected 2 entries, got {len(emotion_entries)}"
    assert emotion_entries[0].data['emotion'] == 'joy'
    assert emotion_entries[0].data['source'] == 'feat'
    assert 'action_units' in emotion_entries[0].data
    print(f"   ✅ Found {len(emotion_entries)} emotion entries with correct data")
    
    # 3. Test extraction to expressionTimeline
    print("\n2. Testing Timeline Extraction...")
    
    # Create analysis dict with timeline data
    analysis_dict = {
        'video_id': 'test',
        'timeline': {
            'duration': 60,
            'entries': [
                {
                    'entry_type': 'emotion',
                    'start': entry.start,
                    'end': entry.end,
                    'data': entry.data
                }
                for entry in emotion_entries
            ]
        },
        'ml_data': {}
    }
    
    try:
        timelines = _extract_timelines_from_analysis(analysis_dict)
        print("   ✅ Successfully extracted timelines")
    except Exception as e:
        print(f"   ❌ Failed to extract timelines: {e}")
        return False
    
    # Verify expressionTimeline contains FEAT data
    expression_timeline = timelines.get('expressionTimeline', {})
    assert len(expression_timeline) > 0, "Expression timeline is empty"
    
    # Check first entry
    first_key = list(expression_timeline.keys())[0]
    first_entry = expression_timeline[first_key]
    assert first_entry.get('source') == 'feat', f"Expected source 'feat', got {first_entry.get('source')}"
    assert first_entry.get('emotion') == 'joy', f"Expected emotion 'joy', got {first_entry.get('emotion')}"
    assert 'action_units' in first_entry, "Missing action_units in expression timeline"
    
    print(f"   ✅ Expression timeline contains {len(expression_timeline)} FEAT entries")
    print(f"   ✅ First entry: {first_key} -> emotion={first_entry.get('emotion')}, AUs={first_entry.get('action_units')}")
    
    # 4. Test with MediaPipe data present (should be ignored)
    print("\n3. Testing FEAT Priority over MediaPipe...")
    analysis_dict['ml_data']['mediapipe'] = {
        'faces': [
            {'timestamp': 0, 'expression': 'neutral', 'confidence': 0.7}
        ]
    }
    
    timelines = _extract_timelines_from_analysis(analysis_dict)
    expression_timeline = timelines.get('expressionTimeline', {})
    
    # Should still have FEAT data, not MediaPipe
    first_entry = expression_timeline.get('0-1s', {})
    assert first_entry.get('source') == 'feat', "FEAT should take priority over MediaPipe"
    print("   ✅ FEAT data takes priority when both are present")
    
    print("\n✅ All tests passed! FEAT integration is working correctly.")
    return True

if __name__ == "__main__":
    success = test_feat_timeline_integration()
    sys.exit(0 if success else 1)