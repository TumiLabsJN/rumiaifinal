#!/usr/bin/env python3
"""
Test pitch extraction implementation
Tests the extended audio energy service with pitch capabilities
"""

import asyncio
import json
from pathlib import Path
import sys
sys.path.append('/home/jorge/rumiaifinal')

from rumiai_v2.ml_services.audio_energy_service_extended import AudioEnergyService


async def test_pitch_extraction():
    """Test pitch extraction on sample video"""

    # Initialize service
    service = AudioEnergyService()

    # Test audio path (using existing WAV file)
    test_audio = Path("/home/jorge/rumiaifinal/temp/7529056602041699592_audio.wav")

    if not test_audio.exists():
        print(f"Error: Test audio not found at {test_audio}")
        print("Looking for available audio files...")
        test_dir = Path("/home/jorge/rumiaifinal/temp")
        audios = list(test_dir.glob("*.wav"))
        if audios:
            test_audio = audios[0]
            print(f"Using {test_audio}")
        else:
            print("No audio files found. Please provide an audio file.")
            return

    print(f"Testing pitch extraction on: {test_audio}")

    # Extract ID from filename
    audio_id = test_audio.stem

    # Run analysis
    print("\nRunning audio analysis with pitch extraction...")
    try:
        result = await service.analyze(test_audio, video_id=None)

        # Check if pitch was extracted
        if 'pitch_frames' in result:
            print("✓ Pitch extraction successful!")
            print(f"  - Pitch frames: {len(result.get('pitch_frames', []))}")
            print(f"  - Pitch FPS: {result.get('pitch_fps', 0):.2f}")

            if 'pitch_statistics' in result:
                stats = result['pitch_statistics']
                print("\n  Pitch Statistics:")
                print(f"    - Mean pitch: {stats.get('mean', 0):.2f} Hz")
                print(f"    - P50 (median): {stats.get('p50', 0):.2f} Hz")
                print(f"    - P20: {stats.get('p20', 0):.2f} Hz")
                print(f"    - P80: {stats.get('p80', 0):.2f} Hz")
                print(f"    - Voiced frames: {stats.get('total_voiced_frames', 0)}")
        else:
            print("✗ No pitch data in result")

        # Check energy extraction still works
        if 'energy_windows' in result:
            print("\n✓ Energy extraction still functional")
            print(f"  - Energy windows: {len(result['energy_windows'])}")
            print(f"  - Duration: {result.get('duration', 0):.2f}s")

        # Save full result for inspection
        output_file = Path(f"/home/jorge/rumiaifinal/test_outputs/{audio_id}_pitch_test.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Full results saved to: {output_file}")

        # Performance check
        processing_ms = result.get('processing_ms', 0)
        if processing_ms > 0:
            print(f"\n✓ Processing time: {processing_ms/1000:.2f}s")
            if processing_ms > 15000:
                print("  ⚠ Warning: Processing took longer than 15s threshold")

    except Exception as e:
        print(f"✗ Error during pitch extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("PITCH EXTRACTION TEST")
    print("=" * 60)

    asyncio.run(test_pitch_extraction())

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)