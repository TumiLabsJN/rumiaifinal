#!/usr/bin/env python3
"""
Minimal test to see what Anthropic API actually returns.
This will help us understand the response structure.
"""

import os
import json
from anthropic import Anthropic

# Load API key
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("ERROR: ANTHROPIC_API_KEY not set")
    exit(1)

print(f"API Key loaded: {api_key[:20]}...")

# Create client
client = Anthropic(api_key=api_key)

# Simple test prompt that should return JSON
prompt = """Please return the following JSON structure exactly:

{
  "test": "success",
  "message": "This is a test response"
}

Return ONLY the JSON, no other text."""

print("\nCalling Anthropic API...")

try:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"\n✓ API call successful!")
    print(f"Response type: {type(response)}")
    print(f"Response attributes: {dir(response)}")

    # Try to access content
    print(f"\nresponse.content type: {type(response.content)}")
    print(f"response.content: {response.content}")

    # Try to access text
    if hasattr(response, 'content') and len(response.content) > 0:
        print(f"\nresponse.content[0] type: {type(response.content[0])}")
        print(f"response.content[0]: {response.content[0]}")

        if hasattr(response.content[0], 'text'):
            text = response.content[0].text
            print(f"\nresponse.content[0].text type: {type(text)}")
            print(f"response.content[0].text length: {len(text)}")
            print(f"response.content[0].text content:\n{text}")

            # Try to parse as JSON
            try:
                parsed = json.loads(text)
                print(f"\n✓ JSON parsing successful!")
                print(f"Parsed content: {json.dumps(parsed, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"\n✗ JSON parsing failed: {e}")
                print(f"First 200 chars: {text[:200]}")
        else:
            print("✗ response.content[0] has no 'text' attribute")
    else:
        print("✗ response.content is empty or not a list")

except Exception as e:
    print(f"\n✗ Error: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
