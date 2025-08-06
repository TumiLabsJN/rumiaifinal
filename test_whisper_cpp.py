#!/usr/bin/env python3
"""
Test script to verify Whisper.cpp integration
"""

import asyncio
import sys
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, '/home/jorge/rumiaifinal')

from rumiai_v2.api.whisper_cpp_service import WhisperCppTranscriber

async def test_whisper_cpp():
    """Test Whisper.cpp transcription"""
    
    print("Testing Whisper.cpp integration...")
    
    try:
        # Initialize transcriber
        print("1. Initializing WhisperCppTranscriber...")
        transcriber = WhisperCppTranscriber()
        print("   ✓ Transcriber initialized successfully")
        
        # Test with the audio file we created
        test_audio = Path("/home/jorge/rumiaifinal/test_audio.wav")
        if not test_audio.exists():
            print("   ✗ Test audio file not found")
            return False
            
        print(f"2. Transcribing test audio: {test_audio}")
        result = await transcriber.transcribe(test_audio)
        
        print("3. Checking results...")
        if 'text' in result and 'segments' in result:
            print(f"   ✓ Transcription successful")
            print(f"   - Text: '{result['text'][:50]}...' (length: {len(result['text'])})")
            print(f"   - Segments: {len(result['segments'])}")
            print(f"   - Language: {result.get('language', 'unknown')}")
            print(f"   - Duration: {result.get('duration', 0):.2f}s")
            return True
        else:
            print(f"   ✗ Unexpected result format: {result.keys()}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_whisper_cpp())
    if success:
        print("\n✅ Whisper.cpp integration test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Whisper.cpp integration test FAILED")
        sys.exit(1)