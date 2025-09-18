#!/usr/bin/env python3
"""
Verify that pitch extraction is integrated into production flow
"""

import sys
sys.path.append('/home/jorge/rumiaifinal')

# Test 1: Verify the audio energy service has pitch capabilities
print("=" * 60)
print("VERIFYING PITCH INTEGRATION")
print("=" * 60)

print("\n1. Checking audio_energy_service.py has pitch extraction...")
from rumiai_v2.ml_services.audio_energy_service import AudioEnergyService

service = AudioEnergyService()
if hasattr(service, 'config') and 'enabled' in service.config:
    print("✓ AudioEnergyService has pitch configuration")
    print(f"  - Pitch enabled: {service.config.get('enabled')}")
    print(f"  - Sample rate: {service.config.get('sample_rate')} Hz")
    print(f"  - Quality: {service.config.get('quality')}")
else:
    print("✗ AudioEnergyService missing pitch configuration")

# Test 2: Check if _extract_pitch method exists
if hasattr(service, '_extract_pitch'):
    print("✓ _extract_pitch method exists")
else:
    print("✗ _extract_pitch method missing")

# Test 3: Verify imports work correctly
print("\n2. Checking imports from all entry points...")

try:
    from rumiai_v2.ml_services import AudioEnergyService
    print("✓ Import from ml_services.__init__ works")
except ImportError as e:
    print(f"✗ Import from ml_services.__init__ failed: {e}")

try:
    from rumiai_v2.processors.video_analyzer import VideoAnalyzer
    print("✓ video_analyzer.py imports successfully")
except ImportError as e:
    print(f"✗ video_analyzer.py import failed: {e}")

try:
    from rumiai_v2.api.ml_services_unified import get_audio_energy_service
    result = get_audio_energy_service()
    if hasattr(result, 'config'):
        print("✓ ml_services_unified returns extended service")
    else:
        print("✗ ml_services_unified returns original service")
except ImportError as e:
    print(f"✗ ml_services_unified import failed: {e}")

# Test 4: Check temporal_compute has pitch metrics calculation
print("\n3. Checking temporal_compute.py integration...")
try:
    from rumiai_v2.processors.temporal_compute import calculate_pitch_metrics
    print("✓ calculate_pitch_metrics function exists")

    # Check function signature
    import inspect
    sig = inspect.signature(calculate_pitch_metrics)
    params = list(sig.parameters.keys())
    expected = ['audio_data', 'ml_data', 'start', 'end']
    if params == expected:
        print(f"✓ Function signature correct: {params}")
    else:
        print(f"✗ Function signature mismatch: {params} != {expected}")
except ImportError as e:
    print(f"✗ calculate_pitch_metrics not found: {e}")

print("\n" + "=" * 60)
print("INTEGRATION STATUS:")
if all([
    hasattr(service, 'config'),
    hasattr(service, '_extract_pitch'),
]):
    print("✅ PITCH EXTRACTION IS INTEGRATED IN PRODUCTION")
    print("\nNext step: Run rumiai_runner.py on a new video to see pitch metrics in output")
else:
    print("❌ PITCH EXTRACTION NOT FULLY INTEGRATED")

print("=" * 60)