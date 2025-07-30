#!/usr/bin/env python3
"""
Demo script to show Claude integration with temporal markers
"""

import json
import sys
import os
from pathlib import Path
from run_claude_insight import ClaudeInsightRunner


def demo_claude_temporal(video_id: str, enable_temporal: bool = True):
    """Demonstrate Claude analysis with and without temporal markers"""
    
    print(f"🤖 Claude Temporal Integration Demo")
    print(f"   Video ID: {video_id}")
    print(f"   Temporal Markers: {'ENABLED' if enable_temporal else 'DISABLED'}")
    print("=" * 60)
    
    # Set environment variable to control temporal markers
    os.environ['ENABLE_TEMPORAL_MARKERS'] = 'true' if enable_temporal else 'false'
    
    # Initialize runner
    runner = ClaudeInsightRunner()
    
    # Example prompts that benefit from temporal markers
    test_prompts = {
        'hook_effectiveness': {
            'prompt': """Analyze the hook effectiveness in the first 5 seconds of this video. 
            Consider:
            1. What elements appear in the first 2 seconds to grab attention?
            2. How does the content density progress through the first 5 seconds?
            3. Are there any early CTAs or urgency triggers?
            4. What emotional journey does the viewer experience?
            
            Provide specific timestamps and patterns that contribute to hook effectiveness.""",
            'context': {}
        },
        
        'cta_analysis': {
            'prompt': """Analyze the call-to-action strategy in this video, especially in the closing window.
            Consider:
            1. When do CTAs first appear?
            2. How are CTAs synchronized with gestures or visual emphasis?
            3. What objects or people are in focus during CTA delivery?
            4. Is there a pattern between content pacing and CTA placement?
            
            Identify specific temporal patterns that enhance CTA effectiveness.""",
            'context': {}
        },
        
        'viral_patterns': {
            'prompt': """Identify viral video patterns based on temporal analysis.
            Look for:
            1. "Wait for it" moments - when and how they're set up
            2. Density spikes that maintain attention
            3. Emotional rollercoasters and their timing
            4. Strategic reveals or payoffs
            
            Explain how the timing of these elements contributes to virality.""",
            'context': {}
        }
    }
    
    # Load some basic context from unified analysis if available
    unified_path = Path(f'unified_analysis_{video_id}.json')
    if unified_path.exists():
        with open(unified_path, 'r') as f:
            unified = json.load(f)
            
        # Add relevant context
        for prompt_name in test_prompts:
            test_prompts[prompt_name]['context'] = {
                'video_stats': unified.get('static_metadata', {}).get('stats', {}),
                'duration': unified.get('video_info', {}).get('duration', 0),
                'captions': unified.get('static_metadata', {}).get('captionText', '')
            }
    
    # Run test prompts
    print("\n📝 Running test prompts...")
    
    for prompt_name, prompt_data in test_prompts.items():
        print(f"\n🔄 Testing: {prompt_name}")
        print("-" * 40)
        
        result = runner.run_claude_prompt(
            video_id=video_id,
            prompt_name=f"demo_{prompt_name}_{'with' if enable_temporal else 'without'}_temporal",
            prompt_text=prompt_data['prompt'],
            context_data=prompt_data['context']
        )
        
        if result['success']:
            print(f"✅ Analysis complete!")
            print(f"   Saved to: {result['response_file']}")
            
            # Show preview
            with open(result['response_file'], 'r') as f:
                response = f.read()
                preview = response[:300] + "..." if len(response) > 300 else response
                print(f"\n   Preview: {preview}")
        else:
            print(f"❌ Analysis failed: {result['error']}")
    
    # Compare with/without temporal markers
    print("\n\n📊 Comparison Notes:")
    if enable_temporal:
        print("WITH temporal markers, Claude can:")
        print("  - Reference specific timestamps and density patterns")
        print("  - Identify moment-to-moment progression")
        print("  - Detect synchronization between elements")
        print("  - Discover temporal patterns (e.g., 'wait for it at 2s')")
    else:
        print("WITHOUT temporal markers, Claude:")
        print("  - Has general video statistics")
        print("  - Cannot reference specific timing")
        print("  - Misses density and progression patterns")
        print("  - Cannot identify temporal correlations")
    
    print("\n💡 To toggle temporal markers:")
    print("   - Set ENABLE_TEMPORAL_MARKERS=false in environment")
    print("   - Or modify config/temporal_markers.json")
    print("   - Or use rollout percentage for gradual deployment")


def main():
    if len(sys.argv) < 2:
        print("Usage: python demo_claude_temporal.py <video_id> [--no-temporal]")
        print("\nExample: python demo_claude_temporal.py 7142620042085264642")
        print("         python demo_claude_temporal.py 7142620042085264642 --no-temporal")
        sys.exit(1)
    
    video_id = sys.argv[1]
    enable_temporal = '--no-temporal' not in sys.argv
    
    demo_claude_temporal(video_id, enable_temporal)


if __name__ == "__main__":
    main()