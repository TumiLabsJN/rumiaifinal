#!/usr/bin/env python3
"""
Test script to verify Audio Energy Service integration
"""

import asyncio
import sys
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, '/home/jorge/rumiaifinal')

from rumiai_v2.ml_services.audio_energy_service import AudioEnergyService

async def test_audio_energy():
    """Test audio energy analysis"""
    
    print("Testing Audio Energy Service...")
    
    try:
        # Initialize service
        print("1. Initializing AudioEnergyService...")
        service = AudioEnergyService()
        print("   ✓ Service initialized successfully")
        
        # Test with the audio file we created
        test_audio = Path("/home/jorge/rumiaifinal/test_audio.wav")
        if not test_audio.exists():
            print("   Creating test audio file...")
            import subprocess
            subprocess.run([
                'ffmpeg', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=10',
                '-ar', '16000', '-ac', '1', str(test_audio), '-y'
            ], capture_output=True)
            
        print(f"2. Analyzing test audio: {test_audio}")
        result = await service.analyze(test_audio)
        
        print("3. Checking results...")
        if result and result.get('metadata', {}).get('success'):
            print(f"   ✓ Analysis successful")
            print(f"   - Energy windows: {len(result.get('energy_level_windows', {}))}")
            print(f"   - Energy variance: {result.get('energy_variance', 0):.4f}")
            print(f"   - Climax timestamp: {result.get('climax_timestamp', 0):.2f}s")
            print(f"   - Burst pattern: {result.get('burst_pattern', 'unknown')}")
            print(f"   - Duration: {result.get('duration', 0):.2f}s")
            
            # Save results
            print("4. Saving results...")
            output_dir = Path("/home/jorge/rumiaifinal/ml_outputs")
            await service.save_results(result, "test_audio", output_dir)
            print(f"   ✓ Results saved to {output_dir}/test_audio_audio_energy.json")
            
            return True
        else:
            print(f"   ✗ Analysis failed: {result.get('metadata', {}).get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_audio_energy())
    if success:
        print("\n✅ Audio Energy Service test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Audio Energy Service test FAILED")
        sys.exit(1)