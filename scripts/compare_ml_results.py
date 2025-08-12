#!/usr/bin/env python3
"""
Compare ML results between production and test for the same video
Usage: python3 compare_ml_results.py <video_id_1> <video_id_2> [analysis_type]
"""

import json
import sys
from pathlib import Path
from glob import glob

# Get arguments
if len(sys.argv) < 3:
    print("Usage: python3 compare_ml_results.py <video_id_1> <video_id_2> [analysis_type]")
    print("Example: python3 compare_ml_results.py 7518107389090876727 TestVideoTester12.08pt2 creative_density")
    sys.exit(1)

video1_id = sys.argv[1]
video2_id = sys.argv[2]
analysis_type = sys.argv[3] if len(sys.argv) > 3 else 'creative_density'

# Find the ML files
video1_files = glob(f'/home/jorge/rumiaifinal/insights/{video1_id}/{analysis_type}/{analysis_type}_ml_*.json')
video2_files = glob(f'/home/jorge/rumiaifinal/insights/{video2_id}/{analysis_type}/{analysis_type}_ml_*.json')

if not video1_files:
    print(f"Error: No {analysis_type} ML file found for video {video1_id}")
    sys.exit(1)
if not video2_files:
    print(f"Error: No {analysis_type} ML file found for video {video2_id}")
    sys.exit(1)

# Load the files
with open(video1_files[0]) as f:
    prod = json.load(f)

with open(video2_files[0]) as f:
    test = json.load(f)

print(f"\n{'='*60}")
print(f"Comparing {analysis_type} results")
print(f"Video 1: {video1_id}")
print(f"Video 2: {video2_id}")
print('='*60)

# Compare CoreMetrics if available
if 'CoreMetrics' in prod and 'CoreMetrics' in test:
    print('\nCORE METRICS COMPARISON:')
    for key in prod['CoreMetrics']:
        if key in test['CoreMetrics']:
            prod_val = prod['CoreMetrics'][key]
            test_val = test['CoreMetrics'][key]
            if isinstance(prod_val, (int, float)) and isinstance(test_val, (int, float)):
                diff = test_val - prod_val
                pct = (diff / prod_val * 100) if prod_val != 0 else 0
                print(f"  {key:25} {prod_val:10.2f} → {test_val:10.2f}  ({pct:+.1f}%)")

# For creative_density, show density curve comparison
if analysis_type == 'creative_density' and 'Dynamics' in prod and 'Dynamics' in test:
    prod_curve = prod['Dynamics'].get('densityCurve', [])
    test_curve = test['Dynamics'].get('densityCurve', [])
    
    print('\nDENSITY PER SECOND DIFFERENCES:')
    print('Sec  Vid1  Vid2  Diff  Primary(Vid1)  Primary(Vid2)')
    print('-' * 55)

differences = []
for i in range(min(len(prod_curve), len(test_curve))):
    prod_density = prod_curve[i]['density']
    test_density = test_curve[i]['density']
    diff = test_density - prod_density
    
    if diff != 0:
        differences.append((i, diff))
        
    # Show first 10 seconds and any with differences
    if i < 10 or diff != 0:
        prod_primary = prod_curve[i]['primaryElement']
        test_primary = test_curve[i]['primaryElement']
        symbol = '  ' if diff == 0 else '❌'
        print(f'{symbol}{i:2}   {prod_density:3}   {test_density:3}   {diff:+3}   {prod_primary:10}     {test_primary:10}')

print(f'\nTotal seconds with differences: {len(differences)} out of {len(prod_curve)}')
avg_diff = sum(d for _, d in differences) / len(differences) if differences else 0
print(f'Average difference: {avg_diff:.2f}')

# Check where text/object detection differs
print('\n\nCOMPARING PEAK MOMENTS:')
prod_peaks = prod['KeyEvents']['peakMoments']
test_peaks = test['KeyEvents']['peakMoments']

print(f'Production peaks: {len(prod_peaks)}')
print(f'Test peaks: {len(test_peaks)}')

# Compare first few peaks
for i in range(min(3, len(prod_peaks), len(test_peaks))):
    print(f'\nPeak {i+1}:')
    print(f'  Prod: {prod_peaks[i]["timestamp"]}, elements: {prod_peaks[i]["totalElements"]}')
    print(f'  Test: {test_peaks[i]["timestamp"]}, elements: {test_peaks[i]["totalElements"]}')