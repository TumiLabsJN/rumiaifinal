#!/usr/bin/env python3
"""
Demo script to show temporal marker extraction and integration
"""

import json
import sys
from pathlib import Path
from python.temporal_marker_integration import TemporalMarkerPipeline


def demo_temporal_markers(video_id: str):
    """Demonstrate temporal marker extraction for a video"""
    
    print(f"🎬 Temporal Marker Extraction Demo for: {video_id}")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = TemporalMarkerPipeline(video_id)
    
    # Extract markers
    print("\n📊 Extracting temporal markers from all analyzers...")
    try:
        markers = pipeline.extract_all_markers()
        
        # Get summary
        summary = pipeline.get_marker_summary(markers)
        
        # Display results
        print("\n✅ Temporal Markers Extracted Successfully!")
        print("\n📈 Summary Statistics:")
        print(f"   Video Duration: {summary['duration']:.1f}s")
        print(f"   Total Size: {summary['size_kb']:.1f}KB")
        
        print("\n🎯 First 5 Seconds Analysis:")
        print(f"   - Text Moments: {summary['first_5_seconds']['text_moments']}")
        print(f"   - Average Density: {summary['first_5_seconds']['density_avg']:.1f} events/second")
        print(f"   - Emotion Sequence: {summary['first_5_seconds']['emotions']}")
        print(f"   - Gesture Count: {summary['first_5_seconds']['gesture_count']}")
        print(f"   - Object Appearances: {summary['first_5_seconds']['object_appearances']}")
        
        print(f"\n🔔 CTA Window Analysis ({summary['cta_window']['time_range']}):")
        print(f"   - CTA Appearances: {summary['cta_window']['cta_count']}")
        print(f"   - Gesture Synchronization: {summary['cta_window']['gesture_sync_count']}")
        print(f"   - Object Focus: {summary['cta_window']['object_focus_count']}")
        
        # Show sample data
        print("\n📝 Sample Temporal Marker Data:")
        
        # First 5 seconds sample
        first_5 = markers.get('first_5_seconds', {})
        if first_5.get('text_moments'):
            print("\n   Text Moments (first 3):")
            for i, moment in enumerate(first_5['text_moments'][:3]):
                print(f"     {i+1}. {moment['time']:.1f}s: \"{moment['text']}\" ({moment.get('size', 'M')})")
        
        if first_5.get('gesture_moments'):
            print("\n   Gesture Moments (first 3):")
            for i, gesture in enumerate(first_5['gesture_moments'][:3]):
                target = f" → {gesture['target']}" if 'target' in gesture else ""
                print(f"     {i+1}. {gesture['time']:.1f}s: {gesture['gesture']}{target} (conf: {gesture['confidence']:.2f})")
        
        # CTA window sample
        cta = markers.get('cta_window', {})
        if cta.get('cta_appearances'):
            print("\n   CTA Appearances:")
            for i, cta_item in enumerate(cta['cta_appearances'][:3]):
                print(f"     {i+1}. {cta_item['time']:.1f}s: \"{cta_item['text']}\" ({cta_item.get('type', 'overlay')})")
        
        # Density visualization
        print("\n📊 Density Progression (First 5 seconds):")
        density = first_5.get('density_progression', [0]*5)
        for i, d in enumerate(density):
            bar = "█" * int(d * 2)  # Scale for visualization
            print(f"   Second {i}: {bar} ({d})")
        
        # Save option
        save = input("\n💾 Save temporal markers to file? (y/n): ")
        if save.lower() == 'y':
            output_path = pipeline.save_markers(markers)
            print(f"   ✅ Saved to: {output_path}")
        
        # Show how to use in unified analysis
        print("\n🔄 Integration with Unified Analysis:")
        print("   The temporal markers are automatically integrated when running:")
        print(f"   python update_unified_analysis.py {video_id}")
        
        return markers
        
    except Exception as e:
        print(f"\n❌ Error extracting temporal markers: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    if len(sys.argv) != 2:
        print("Usage: python demo_temporal_markers.py <video_id>")
        print("\nExample: python demo_temporal_markers.py 7142620042085264642")
        sys.exit(1)
    
    video_id = sys.argv[1]
    demo_temporal_markers(video_id)


if __name__ == "__main__":
    main()