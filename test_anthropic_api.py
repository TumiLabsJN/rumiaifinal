#!/usr/bin/env python3
"""
Quick test to verify Anthropic API key works.
Tests basic API connectivity without running the full pipeline.
"""

import os
import sys
from pathlib import Path

# Load .env file
def load_env():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value
        print("✓ Loaded .env file")
    else:
        print("✗ .env file not found")
        sys.exit(1)

load_env()

# Check API key is set
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("✗ ANTHROPIC_API_KEY not set in environment")
    sys.exit(1)

print(f"✓ ANTHROPIC_API_KEY found: {api_key[:20]}...{api_key[-4:]}")

# Try importing anthropic
try:
    import anthropic
    print(f"✓ anthropic package imported (version: {anthropic.__version__})")
except ImportError as e:
    print(f"✗ Failed to import anthropic: {e}")
    sys.exit(1)

# Test API connection with a minimal request
print("\nTesting API connection...")
print("Sending minimal test request to Claude API...")

try:
    client = anthropic.Anthropic(api_key=api_key)

    # Minimal test message
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=50,
        messages=[
            {"role": "user", "content": "Say 'API test successful' in exactly 3 words."}
        ]
    )

    response_text = message.content[0].text
    print(f"\n✅ API Test Successful!")
    print(f"   Response: {response_text}")
    print(f"   Model: {message.model}")
    print(f"   Tokens used: {message.usage.input_tokens} input, {message.usage.output_tokens} output")
    print(f"   Estimated cost: ${(message.usage.input_tokens * 0.000003 + message.usage.output_tokens * 0.000015):.6f}")

except anthropic.AuthenticationError as e:
    print(f"\n✗ Authentication Failed: {e}")
    print("   Check that your API key is valid")
    print("   Get a new key from: https://console.anthropic.com/settings/keys")
    sys.exit(1)

except anthropic.RateLimitError as e:
    print(f"\n⚠️  Rate Limit Exceeded: {e}")
    print("   Wait a moment and try again")
    sys.exit(1)

except anthropic.APIError as e:
    print(f"\n✗ API Error: {e}")
    sys.exit(1)

except Exception as e:
    print(f"\n✗ Unexpected Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("ANTHROPIC API READY FOR STAGE 7")
print("="*80)
print("\nYou can now run the full pipeline:")
print("  python3 rumiai_ml_batch.py --client test --target '#nutrition'")
print()
