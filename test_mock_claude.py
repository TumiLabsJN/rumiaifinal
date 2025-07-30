#!/usr/bin/env python3
"""Test mock Claude API functionality"""

import json
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rumiai_v2.api.claude_client import ClaudeClient

# Set environment to use mock API
os.environ['USE_MOCK_CLAUDE'] = 'true'
os.environ['USE_ML_PRECOMPUTE'] = 'true'

def test_mock_responses():
    """Test the mock Claude API responses"""
    
    print("="*80)
    print("MOCK CLAUDE API TEST")
    print("="*80)
    
    # Initialize client with mock API key
    client = ClaudeClient(api_key="mock_api_key", model="claude-3-haiku-20240307")
    
    # Test prompts
    test_prompts = [
        "Analyze the concept of artificial intelligence.",
        "Write a creative story about robots.",
        "Explain quantum computing."
    ]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nTest {i+1}: {prompt[:50]}...")
        
        try:
            # Send prompt with context
            result = client.send_prompt(
                prompt=prompt,
                context_data={"test_id": i},
                timeout=30,
                max_tokens=1000
            )
            
            print(f"Success: {result.success}")
            print(f"Content length: {len(result.content)}")
            print(f"Model: {result.model}")
            print(f"Total tokens: {result.total_tokens}")
            print(f"Cost: ${result.cost:.4f}")
            print(f"First 200 chars: {result.content[:200]}...")
            
            # Save response
            filename = f"mock_response_{i+1}.json"
            with open(filename, 'w') as f:
                json.dump({
                    "prompt": prompt,
                    "response": result.content,
                    "metadata": {
                        "model": result.model,
                        "total_tokens": result.total_tokens,
                        "cost": result.cost,
                        "success": result.success
                    }
                }, f, indent=2)
            print(f"Saved to: {filename}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_mock_responses()