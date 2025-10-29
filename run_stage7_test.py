#!/usr/bin/env python3
"""
Wrapper to run Stage 7 with .env loaded
"""
import os
import sys
from pathlib import Path

# Load .env file
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
    print(f"✓ Loaded .env file")
else:
    print(f"⚠ .env file not found at {env_file}")

# Verify API key is set
api_key = os.environ.get('ANTHROPIC_API_KEY')
if api_key:
    print(f"✓ ANTHROPIC_API_KEY set ({api_key[:20]}...)")
else:
    print(f"✗ ANTHROPIC_API_KEY not set!")
    sys.exit(1)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run Stage 7
from ml_pipeline.stage7_llm_analysis.stage7_llm_analysis import main as stage7_main

bucket_path = 'data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s'
bucket = '18-33s'
hashtag = 'test_vitamin'

print('\nStarting Stage 7 LLM Analysis...')
print('(Calling Claude API, may take 1-2 minutes)\n')

stage7_main(bucket_path, bucket, hashtag)

print('\n✓ Stage 7 Complete')
