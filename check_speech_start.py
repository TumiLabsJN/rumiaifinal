#!/usr/bin/env python3
import json
import os
from pathlib import Path

# Analyze speech start times across videos
unified_dir = Path("/home/jorge/rumiaifinal/unified_analysis")
results = []

# Include our known case
target_files = ['830916697805225.json', '556666642086470.json']
json_files = [unified_dir / f for f in target_files if (unified_dir / f).exists()]
# Add more random files
json_files.extend(list(unified_dir.glob("*.json"))[:20])

for json_file in json_files:
    try:
        with open(json_file) as f:
            data = json.load(f)

        video_id = json_file.stem

        # Get first speech segment start time
        # Try multiple paths where segments might be
        segments = (data.get('ml_data', {}).get('whisper', {}).get('segments', []) or
                   data.get('transcription', {}).get('segments', []) or
                   data.get('segments', []))
        first_speech_start = segments[0]['start'] if segments else None

        # Check if audio has energy from start
        audio_energy = (data.get('ml_data', {}).get('audio_energy', {}).get('rms_frames', []) or
                       data.get('audio_energy', {}).get('rms_frames', []))
        has_audio_from_start = len(audio_energy) > 0

        # Get first few words of transcript
        first_text = segments[0].get('text', '')[:50] if segments else "No speech"

        # Check for music markers in first segment
        has_music_marker = '[Music]' in first_text or '[music]' in first_text or '*music*' in first_text.lower()

        results.append({
            'video_id': video_id,
            'first_speech_start': first_speech_start,
            'has_audio_from_start': has_audio_from_start,
            'has_music_marker': has_music_marker,
            'first_text': first_text,
            'segments_found': len(segments) > 0
        })
    except Exception as e:
        print(f"Error processing {json_file}: {e}")
        continue

# Analyze results
print("=== Speech Start Time Analysis ===\n")

# Count videos with delayed speech start
delayed_start = [r for r in results if r['first_speech_start'] and r['first_speech_start'] > 0.2]
any_delay = [r for r in results if r['first_speech_start'] and r['first_speech_start'] > 0]
print(f"Videos with ANY delayed speech start (>0s): {len(any_delay)}/{len(results)}")
print(f"Videos with significant delay (>0.2s): {len(delayed_start)}/{len(results)}")

# Show examples
print("\nExamples of ANY delays:")
for r in any_delay[:5]:
    print(f"  {r['video_id']}: Speech starts at {r['first_speech_start']}s")
    print(f"    Has music marker: {r['has_music_marker']}")
    print(f"    Text: '{r['first_text']}'")

# Check correlation with music
music_videos = [r for r in results if r['has_music_marker']]
print(f"\nVideos with music markers: {len(music_videos)}/{len(results)}")

# Check how many have segments at all
videos_with_segments = [r for r in results if r['segments_found']]
print(f"\nVideos with transcription segments: {len(videos_with_segments)}/{len(results)}")

# Summary
print("\n=== Summary ===")
start_times = [r['first_speech_start'] for r in results if r['first_speech_start'] is not None]
if start_times:
    print(f"Average first speech start: {sum(start_times)/len(start_times):.2f}s")
    print(f"Min start time: {min(start_times)}s")
    print(f"Max start time: {max(start_times)}s")
else:
    print("No videos with speech segments found - check data structure")