#!/usr/bin/env python3
"""
Analyze text overlay distribution across videos to validate OCR sampling assumption
"""

import json
import glob
from pathlib import Path
import numpy as np

def analyze_text_distribution(json_file):
    """Analyze where text appears in a video"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    video_id = data.get('video_id', 'unknown')
    duration = data.get('duration', 0)
    temporal_windows = data.get('temporal_windows', {})

    # Get text metrics from each window
    hook = temporal_windows.get('hook', {})
    middle_segments = temporal_windows.get('middle_segments', [])
    closing = temporal_windows.get('closing', {})

    results = {
        'video_id': video_id,
        'duration': duration,
        'hook': {
            'overlay_unique_count': hook.get('overlay_unique_count', 0),
            'overlay_coverage': hook.get('overlay_coverage', 0),
            'has_captions': hook.get('has_captions', False),
            'window_duration': hook.get('duration', 0)
        },
        'middle': {
            'overlay_unique_count': 0,
            'overlay_coverage': 0,
            'has_captions': False,
            'window_duration': 0,
            'segments': len(middle_segments)
        },
        'closing': {
            'overlay_unique_count': closing.get('overlay_unique_count', 0),
            'overlay_coverage': closing.get('overlay_coverage', 0),
            'has_captions': closing.get('has_captions', False),
            'window_duration': closing.get('duration', 0)
        }
    }

    # Aggregate middle segments
    if middle_segments:
        overlay_counts = [seg.get('overlay_unique_count', 0) for seg in middle_segments]
        overlay_coverages = [seg.get('overlay_coverage', 0) for seg in middle_segments]
        has_captions = [seg.get('has_captions', False) for seg in middle_segments]
        durations = [seg.get('duration', 0) for seg in middle_segments]

        results['middle']['overlay_unique_count'] = np.mean(overlay_counts)
        results['middle']['overlay_coverage'] = np.mean(overlay_coverages)
        results['middle']['has_captions'] = any(has_captions)
        results['middle']['window_duration'] = sum(durations)
        results['middle']['total_text_instances'] = sum(overlay_counts)

    return results

def main():
    # Find all temporal windows JSON files
    insights_dir = Path('/home/jorge/rumiaifinal/insights')
    json_files = glob.glob(str(insights_dir / '*_temporal_windows_updated.json'))

    if not json_files:
        print("No temporal windows JSON files found")
        return

    print(f"Analyzing {len(json_files)} videos for text distribution...")
    print("="*80)

    all_results = []

    for json_file in json_files:
        try:
            results = analyze_text_distribution(json_file)
            all_results.append(results)

            # Print individual video results
            print(f"\nVideo: {results['video_id']} (Duration: {results['duration']:.1f}s)")
            print(f"  Hook (0-3s):")
            print(f"    - Unique overlays: {results['hook']['overlay_unique_count']}")
            print(f"    - Coverage: {results['hook']['overlay_coverage']:.2%}")
            print(f"    - Has captions: {results['hook']['has_captions']}")

            print(f"  Middle ({results['middle']['segments']} segments, {results['middle']['window_duration']:.1f}s):")
            print(f"    - Avg unique overlays: {results['middle']['overlay_unique_count']:.2f}")
            print(f"    - Avg coverage: {results['middle']['overlay_coverage']:.2%}")
            print(f"    - Total text instances: {results['middle']['total_text_instances']}")

            print(f"  Closing (last 3s):")
            print(f"    - Unique overlays: {results['closing']['overlay_unique_count']}")
            print(f"    - Coverage: {results['closing']['overlay_coverage']:.2%}")
            print(f"    - Has captions: {results['closing']['has_captions']}")

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    # Calculate aggregates
    if all_results:
        print("\n" + "="*80)
        print("AGGREGATE ANALYSIS")
        print("="*80)

        # Calculate averages for each window
        hook_overlays = [r['hook']['overlay_unique_count'] for r in all_results]
        middle_overlays = [r['middle']['overlay_unique_count'] for r in all_results]
        closing_overlays = [r['closing']['overlay_unique_count'] for r in all_results]

        hook_coverage = [r['hook']['overlay_coverage'] for r in all_results]
        middle_coverage = [r['middle']['overlay_coverage'] for r in all_results]
        closing_coverage = [r['closing']['overlay_coverage'] for r in all_results]

        print(f"\nAverage Unique Overlays per Window:")
        print(f"  Hook: {np.mean(hook_overlays):.2f} (std: {np.std(hook_overlays):.2f})")
        print(f"  Middle: {np.mean(middle_overlays):.2f} (std: {np.std(middle_overlays):.2f})")
        print(f"  Closing: {np.mean(closing_overlays):.2f} (std: {np.std(closing_overlays):.2f})")

        print(f"\nAverage Coverage per Window:")
        print(f"  Hook: {np.mean(hook_coverage):.2%} (std: {np.std(hook_coverage):.2%})")
        print(f"  Middle: {np.mean(middle_coverage):.2%} (std: {np.std(middle_coverage):.2%})")
        print(f"  Closing: {np.mean(closing_coverage):.2%} (std: {np.std(closing_coverage):.2%})")

        # Test the hypothesis
        print(f"\n📊 HYPOTHESIS TEST: 'Text appears more at beginning/end'")
        print("-"*60)

        avg_hook = np.mean(hook_overlays)
        avg_middle = np.mean(middle_overlays)
        avg_closing = np.mean(closing_overlays)

        beginning_end_avg = (avg_hook + avg_closing) / 2

        print(f"Beginning/End average: {beginning_end_avg:.2f} overlays")
        print(f"Middle average: {avg_middle:.2f} overlays")

        if beginning_end_avg > avg_middle * 1.2:  # 20% more
            print("✅ VALIDATED: Text DOES appear more at beginning/end")
            print(f"   Beginning/end has {(beginning_end_avg/avg_middle - 1)*100:.1f}% more text")
        elif avg_middle > beginning_end_avg * 1.2:
            print("❌ INVALIDATED: Text appears MORE in the middle")
            print(f"   Middle has {(avg_middle/beginning_end_avg - 1)*100:.1f}% more text")
        else:
            print("➖ NEUTRAL: Text distribution is relatively uniform")
            print(f"   Difference is only {abs(beginning_end_avg - avg_middle):.2f} overlays")

        # Additional insights
        print(f"\n📈 Additional Insights:")
        videos_with_hook_text = sum(1 for r in all_results if r['hook']['overlay_unique_count'] > 0)
        videos_with_closing_text = sum(1 for r in all_results if r['closing']['overlay_unique_count'] > 0)

        print(f"  {videos_with_hook_text}/{len(all_results)} videos have text in hook")
        print(f"  {videos_with_closing_text}/{len(all_results)} videos have text in closing")

if __name__ == "__main__":
    main()