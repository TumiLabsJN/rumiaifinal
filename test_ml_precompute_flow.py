#!/usr/bin/env python3
"""Test ML precompute flow with mock data"""

import os
import sys
import json
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up environment
os.environ['CLAUDE_API_KEY'] = 'test-key'
os.environ['USE_ML_PRECOMPUTE'] = 'true'
os.environ['LOG_LEVEL'] = 'DEBUG'

def create_mock_claude_responses():
    """Create mock responses in 6-block format"""
    return {
        "analysis": {
            "blocks": [
                {
                    "name": "analysis",
                    "content": json.dumps({
                        "CoreMetrics": {
                            "understanding_depth": 8.5,
                            "clarity_score": 9.0,
                            "completeness": 0.85
                        },
                        "Dynamics": {
                            "flow": "progressive",
                            "transitions": "smooth",
                            "coherence": 0.9
                        },
                        "Interactions": {
                            "concept_links": ["AI", "society", "ethics"],
                            "interdependencies": 0.8
                        },
                        "KeyEvents": {
                            "main_points": [
                                "AI benefits in healthcare",
                                "Automation challenges",
                                "Ethical considerations"
                            ]
                        },
                        "Patterns": {
                            "recurring_themes": ["progress", "caution", "adaptation"],
                            "pattern_strength": 0.85
                        },
                        "Quality": {
                            "overall_quality": 8.7,
                            "confidence": 0.92
                        }
                    })
                }
            ],
            "metadata": {
                "processing_mode": "ml_precompute",
                "total_tokens": 250,
                "cost": 0.0001
            }
        },
        "insight": {
            "blocks": [
                {
                    "name": "insight", 
                    "content": "Key insight: AI represents both tremendous opportunity and significant challenges. The balance between innovation and responsible development will shape our future."
                }
            ],
            "metadata": {
                "processing_mode": "ml_precompute",
                "total_tokens": 150,
                "cost": 0.00008
            }
        },
        "narrative": {
            "blocks": [
                {
                    "name": "narrative",
                    "content": "The story of artificial intelligence is one of human ambition meeting technological possibility. From its early conceptual stages to today's deep learning systems, AI has evolved from science fiction to everyday reality."
                }
            ],
            "metadata": {
                "processing_mode": "ml_precompute",
                "total_tokens": 200,
                "cost": 0.0001
            }
        },
        "essay": {
            "blocks": [
                {
                    "name": "essay",
                    "content": "Artificial Intelligence: Reshaping Our World\\n\\nArtificial intelligence stands at the forefront of technological innovation, promising to revolutionize every aspect of human life. From healthcare diagnostics that catch diseases early to autonomous vehicles that could eliminate traffic accidents, AI's potential benefits are immense.\\n\\nHowever, this technological revolution also brings profound challenges. Job displacement through automation threatens traditional employment structures. Algorithmic bias can perpetuate and amplify societal inequalities. Privacy concerns grow as AI systems require vast amounts of personal data.\\n\\nThe path forward requires careful balance. We must harness AI's capabilities while implementing robust ethical frameworks, ensuring transparency in decision-making processes, and maintaining human agency. Education and reskilling programs will be essential to help workers adapt.\\n\\nUltimately, AI is a tool that reflects our values and choices. By approaching its development thoughtfully, we can create a future where technology enhances rather than replaces human potential."
                }
            ],
            "metadata": {
                "processing_mode": "ml_precompute",
                "total_tokens": 400,
                "cost": 0.0002
            }
        },
        "haiku": {
            "blocks": [
                {
                    "name": "haiku",
                    "content": "Silicon dreams wake\\nHuman wisdom guides the code\\nFuture blooms with care"
                }
            ],
            "metadata": {
                "processing_mode": "ml_precompute",
                "total_tokens": 50,
                "cost": 0.00003
            }
        },
        "reflection": {
            "blocks": [
                {
                    "name": "reflection",
                    "content": "Reflecting on artificial intelligence reveals our deepest hopes and fears about technology. We see in AI a mirror of human intelligence, yet also something fundamentally different. The questions it raises - about consciousness, creativity, and what makes us human - are as philosophical as they are practical. As we stand at this technological crossroads, we must remember that the choices we make today will echo through generations."
                }
            ],
            "metadata": {
                "processing_mode": "ml_precompute",
                "total_tokens": 180,
                "cost": 0.00009
            }
        }
    }

def test_ml_precompute_processing():
    """Test the complete ML precompute processing flow"""
    
    print("="*80)
    print("ML PRECOMPUTE PROCESSING TEST")
    print("="*80)
    print()
    
    # Create mock responses
    mock_responses = create_mock_claude_responses()
    
    # Test 1: Validate each response
    print("1. VALIDATING 6-BLOCK RESPONSES:")
    from rumiai_v2.validators.response_validator import ResponseValidator
    
    for block_type, response_data in mock_responses.items():
        print(f"\n   Validating {block_type}:")
        
        # Get the content from the first block
        if 'blocks' in response_data and response_data['blocks']:
            content = response_data['blocks'][0]['content']
            
            # For analysis block, validate as JSON
            if block_type == 'analysis':
                try:
                    json_data = json.loads(content)
                    print(f"   - Valid JSON: Yes")
                    print(f"   - Keys: {list(json_data.keys())}")
                    
                    # Check for 6-block structure
                    expected_blocks = ['CoreMetrics', 'Dynamics', 'Interactions', 'KeyEvents', 'Patterns', 'Quality']
                    missing = [b for b in expected_blocks if b not in json_data]
                    if missing:
                        print(f"   - Missing blocks: {missing}")
                    else:
                        print(f"   - All 6 blocks present: Yes")
                except json.JSONDecodeError as e:
                    print(f"   - JSON Error: {e}")
            else:
                print(f"   - Content length: {len(content)} chars")
                print(f"   - First 100 chars: {content[:100]}...")
        
        # Check metadata
        if 'metadata' in response_data:
            meta = response_data['metadata']
            print(f"   - Processing mode: {meta.get('processing_mode')}")
            print(f"   - Tokens: {meta.get('total_tokens')}")
            print(f"   - Cost: ${meta.get('cost'):.5f}")
    
    # Test 2: Calculate total cost and tokens
    print("\n\n2. AGGREGATE METRICS:")
    total_tokens = sum(r['metadata']['total_tokens'] for r in mock_responses.values())
    total_cost = sum(r['metadata']['cost'] for r in mock_responses.values())
    print(f"   - Total blocks: {len(mock_responses)}")
    print(f"   - Total tokens: {total_tokens}")
    print(f"   - Total cost: ${total_cost:.5f}")
    print(f"   - Average tokens per block: {total_tokens / len(mock_responses):.1f}")
    
    # Test 3: Simulate precompute timing
    print("\n\n3. PRECOMPUTE TIMING SIMULATION:")
    print("   Running precompute functions...")
    
    start_time = time.time()
    
    # Simulate extraction batch
    print("\n   Extraction Phase (parallel):")
    extraction_start = time.time()
    time.sleep(0.1)  # Simulate API call
    extraction_time = time.time() - extraction_start
    print(f"   - Extracted 3 blocks in {extraction_time:.3f}s")
    
    # Simulate generation batch
    print("\n   Generation Phase (parallel):")
    generation_start = time.time()
    time.sleep(0.1)  # Simulate API call
    generation_time = time.time() - generation_start
    print(f"   - Generated 3 blocks in {generation_time:.3f}s")
    
    total_time = time.time() - start_time
    print(f"\n   Total processing time: {total_time:.3f}s")
    print(f"   Time saved vs sequential: ~{(0.6 - total_time):.3f}s")
    
    # Test 4: Memory usage
    print("\n\n4. MEMORY USAGE:")
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"   - RSS Memory: {memory_info.rss / 1024 / 1024:.1f} MB")
    print(f"   - VMS Memory: {memory_info.vms / 1024 / 1024:.1f} MB")
    
    # Test 5: Save combined output
    print("\n\n5. SAVING COMBINED OUTPUT:")
    combined_output = {
        "prompt": "Analyze the concept of artificial intelligence and its impact on society.",
        "blocks": mock_responses,
        "metadata": {
            "total_blocks": len(mock_responses),
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "processing_time": total_time,
            "processing_mode": "ml_precompute"
        }
    }
    
    output_file = "test_ml_precompute_output.json"
    with open(output_file, 'w') as f:
        json.dump(combined_output, f, indent=2)
    
    print(f"   - Saved to: {output_file}")
    print(f"   - File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    
    # Test 6: Verify output structure
    print("\n\n6. OUTPUT STRUCTURE VERIFICATION:")
    with open(output_file, 'r') as f:
        saved_data = json.load(f)
    
    print(f"   - Top-level keys: {list(saved_data.keys())}")
    print(f"   - Block types: {list(saved_data['blocks'].keys())}")
    print(f"   - All blocks have metadata: {all('metadata' in b for b in saved_data['blocks'].values())}")
    print(f"   - All blocks have content: {all('blocks' in b and b['blocks'] for b in saved_data['blocks'].values())}")
    
    # Test 7: Validate block order
    print("\n\n7. BLOCK ORDER VALIDATION:")
    expected_order = ['analysis', 'insight', 'narrative', 'essay', 'haiku', 'reflection']
    actual_order = list(saved_data['blocks'].keys())
    print(f"   - Expected order: {expected_order}")
    print(f"   - Actual order: {actual_order}")
    print(f"   - Order correct: {expected_order == actual_order}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    return saved_data

if __name__ == "__main__":
    result = test_ml_precompute_processing()
    
    # Additional analysis
    print("\n\nDETAILED CONTENT ANALYSIS:")
    print("="*80)
    
    for block_type, block_data in result['blocks'].items():
        content = block_data['blocks'][0]['content']
        print(f"\n{block_type.upper()}:")
        print("-" * 40)
        
        if block_type == 'analysis':
            # Parse and show structure
            try:
                analysis_data = json.loads(content)
                for key, value in analysis_data.items():
                    print(f"{key}:")
                    if isinstance(value, dict):
                        for k, v in value.items():
                            print(f"  - {k}: {v}")
                    else:
                        print(f"  {value}")
            except:
                print("Error parsing analysis JSON")
        else:
            # Show text content
            if len(content) > 300:
                print(content[:300] + "...")
            else:
                print(content)
        
        print(f"\nMetadata: {block_data['metadata']}")