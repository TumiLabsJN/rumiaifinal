#!/usr/bin/env python3
"""Manual test script for ML precompute functionality"""

import json
import sys
import os
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rumiai_v2.api.claude_client import ClaudeClient
from rumiai_v2.config.settings import Settings

def test_ml_precompute():
    """Test ML precompute functionality with detailed output analysis"""
    
    print("="*80)
    print("ML PRECOMPUTE TEST - RIGOROUS ANALYSIS")
    print("="*80)
    print(f"Test started at: {datetime.now()}")
    print()
    
    # Create config with ML precompute enabled
    config = Settings()
    config.claude_client.mock_api = True
    config.ml_precompute.enabled = True
    config.ml_precompute.extraction_batch_size = 5
    config.ml_precompute.block_request_order = ['analysis', 'insight', 'narrative', 'essay', 'haiku', 'reflection']
    config.claude_client.debug = True
    
    # Initialize client
    client = ClaudeClient(config)
    
    # Test prompts of different types
    test_prompts = [
        {
            "type": "analytical",
            "prompt": "Analyze the concept of artificial intelligence and its impact on society. Consider both benefits and challenges."
        },
        {
            "type": "creative",
            "prompt": "Write a story about a robot learning to understand human emotions."
        },
        {
            "type": "philosophical",
            "prompt": "What is the nature of consciousness and how does it relate to artificial intelligence?"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}: {test_case['type'].upper()}")
        print(f"{'='*60}")
        print(f"Prompt: {test_case['prompt'][:100]}...")
        print()
        
        start_time = time.time()
        
        # Check if precompute is enabled
        print("1. FEATURE FLAGS CHECK:")
        print(f"   - ML Precompute Enabled: {client.ml_precompute_enabled}")
        print(f"   - Mock API: {client.mock_api}")
        print()
        
        # Process request
        print("2. PROCESSING REQUEST...")
        try:
            response = client.generate_prompt(test_case['prompt'])
            processing_time = time.time() - start_time
            
            # Save response
            filename = f"test_response_{test_case['type']}.json"
            with open(filename, 'w') as f:
                json.dump(response, f, indent=2)
            
            print(f"   - Response saved to: {filename}")
            print(f"   - Processing time: {processing_time:.2f} seconds")
            print()
            
            # Analyze response structure
            print("3. RESPONSE STRUCTURE ANALYSIS:")
            if isinstance(response, dict):
                print(f"   - Response type: dict")
                print(f"   - Keys: {list(response.keys())}")
                
                if 'blocks' in response:
                    blocks = response['blocks']
                    print(f"   - Number of blocks: {len(blocks)}")
                    print(f"   - Block names: {[b.get('name', 'unnamed') for b in blocks]}")
                    
                    # Check for proper 6-block structure
                    expected_blocks = ['analysis', 'insight', 'narrative', 'essay', 'haiku', 'reflection']
                    actual_blocks = [b.get('name', '').lower() for b in blocks]
                    
                    print()
                    print("4. BLOCK VALIDATION:")
                    for expected in expected_blocks:
                        if expected in actual_blocks:
                            print(f"   ✓ {expected} block present")
                        else:
                            print(f"   ✗ {expected} block MISSING")
                    
                    # Check block content
                    print()
                    print("5. BLOCK CONTENT ANALYSIS:")
                    for block in blocks[:3]:  # Check first 3 blocks
                        name = block.get('name', 'unnamed')
                        content = block.get('content', '')
                        print(f"   - {name}:")
                        print(f"     * Content length: {len(content)} chars")
                        print(f"     * First 100 chars: {content[:100]}...")
                        print(f"     * Contains data: {'YES' if len(content) > 50 else 'NO'}")
                
                # Check metadata
                if 'metadata' in response:
                    print()
                    print("6. METADATA ANALYSIS:")
                    metadata = response['metadata']
                    print(f"   - Processing mode: {metadata.get('processing_mode', 'unknown')}")
                    print(f"   - Total tokens: {metadata.get('total_tokens', 0)}")
                    print(f"   - Cost: ${metadata.get('cost', 0):.4f}")
                    
                    if 'timing' in metadata:
                        timing = metadata['timing']
                        print(f"   - Extraction time: {timing.get('extraction_time', 0):.2f}s")
                        print(f"   - Generation time: {timing.get('generation_time', 0):.2f}s")
                        print(f"   - Total time: {timing.get('total_time', 0):.2f}s")
            
            else:
                print(f"   ERROR: Response is not a dict, got {type(response)}")
            
            # Store result
            results.append({
                'test_case': test_case['type'],
                'success': True,
                'response_file': filename,
                'processing_time': processing_time,
                'block_count': len(response.get('blocks', [])) if isinstance(response, dict) else 0
            })
            
        except Exception as e:
            print(f"   ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                'test_case': test_case['type'],
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total test cases: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.get('success'))}")
    print(f"Failed: {sum(1 for r in results if not r.get('success'))}")
    print()
    
    for result in results:
        status = "✓" if result.get('success') else "✗"
        print(f"{status} {result['test_case']}: ", end="")
        if result.get('success'):
            print(f"{result['block_count']} blocks in {result['processing_time']:.2f}s")
        else:
            print(f"FAILED - {result.get('error', 'Unknown error')}")
    
    print(f"\nTest completed at: {datetime.now()}")
    
    return results

if __name__ == "__main__":
    # Run the test
    results = test_ml_precompute()
    
    # Read and analyze the generated files
    print(f"\n{'='*80}")
    print("DETAILED FILE CONTENT ANALYSIS")
    print(f"{'='*80}")
    
    for test_type in ['analytical', 'creative', 'philosophical']:
        filename = f"test_response_{test_type}.json"
        if os.path.exists(filename):
            print(f"\n{test_type.upper()} RESPONSE FILE:")
            with open(filename, 'r') as f:
                data = json.load(f)
            
            if 'blocks' in data:
                for block in data['blocks']:
                    name = block.get('name', 'unnamed')
                    content = block.get('content', '')
                    print(f"\n  Block: {name}")
                    print(f"  Length: {len(content)} chars")
                    if len(content) > 0:
                        # Show meaningful excerpt
                        excerpt = content[:200].strip()
                        if len(content) > 200:
                            excerpt += "..."
                        print(f"  Content: {excerpt}")
                    else:
                        print("  Content: [EMPTY]")