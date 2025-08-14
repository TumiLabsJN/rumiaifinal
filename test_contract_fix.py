#!/usr/bin/env python3
"""
Test if the contract fix allows '46-46s' timestamps
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rumiai_v2.processors.service_contracts import validate_compute_contract

# Test the problematic timestamp that was failing
test_timelines = {
    'expressionTimeline': {
        '45-46s': {'emotion': 'happy', 'confidence': 0.8},
        '46-46s': {'emotion': 'neutral', 'confidence': 0.7},  # This was failing
    }
}

try:
    validate_compute_contract(test_timelines, 46.0)
    print("✅ SUCCESS: Contract validation passed with '46-46s' timestamp")
    print("🎉 Creative Density should now work!")
except Exception as e:
    print(f"❌ FAILED: Contract still rejects '46-46s': {e}")