#!/usr/bin/env python3
"""
Quick live test script for RumiAI v2.

Tests with a real TikTok URL if API keys are configured.
"""
import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from scripts.rumiai_runner import RumiAIRunner


async def test_live():
    """Run a live test with real APIs."""
    # Load environment
    load_dotenv()
    
    # Check API keys
    if not os.getenv('CLAUDE_API_KEY'):
        print("❌ Error: CLAUDE_API_KEY not set in .env file")
        return False
    
    if not os.getenv('APIFY_API_TOKEN'):
        print("❌ Error: APIFY_API_TOKEN not set in .env file")
        return False
    
    # Test URL (you can change this to any TikTok URL)
    test_url = "https://www.tiktok.com/@cristiano/video/7303262416014175489"
    
    print("🧪 RumiAI v2 Live Test")
    print("=" * 50)
    print(f"Testing with: {test_url}")
    print("=" * 50)
    
    try:
        runner = RumiAIRunner()
        result = await runner.process_video_url(test_url)
        
        if result['success']:
            print("\n✅ SUCCESS! Video processed successfully")
            print(f"\n📊 Results:")
            print(f"  - Video ID: {result['video_id']}")
            print(f"  - Unified Analysis: {result['outputs']['unified']}")
            print(f"  - Temporal Markers: {result['outputs']['temporal']}")
            
            if 'report' in result:
                report = result['report']
                print(f"\n📈 Processing Report:")
                print(f"  - Duration: {report['duration']}s")
                print(f"  - ML Analyses: {report['ml_completion_details']}")
                print(f"  - Temporal Markers: {'✅' if report['temporal_markers_generated'] else '❌'}")
                print(f"  - Prompts: {report['prompts_successful']}/{report['prompts_total']} succeeded")
                
                if 'processing_metrics' in report:
                    metrics = report['processing_metrics']
                    print(f"\n⏱️  Performance:")
                    print(f"  - Total Time: {metrics.get('uptime_seconds', 0):.1f}s")
                    print(f"  - Memory Used: {metrics.get('memory', {}).get('rss_mb', 0):.1f}MB")
            
            return True
        else:
            print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")
            print(f"Error Type: {result.get('error_type', 'Unknown')}")
            return False
            
    except Exception as e:
        print(f"\n💥 Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = asyncio.run(test_live())
    sys.exit(0 if success else 1)