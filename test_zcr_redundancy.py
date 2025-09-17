#!/usr/bin/env python3
"""
Test script to analyze redundancy between Zero-Crossing Rate and Energy metrics
Determines if ZCR adds unique information or causes collinearity issues
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import json
from pathlib import Path

def analyze_zcr_vs_energy(audio_path):
    """
    Analyze relationship between ZCR and energy metrics
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Calculate metrics with same hop_length for alignment
    hop_length = 512
    
    # 1. RMS Energy (what we currently have)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    
    # 2. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        y, 
        frame_length=2048,
        hop_length=hop_length
    )[0]
    
    # 3. Energy Variance (what we currently calculate per window)
    # Calculate in sliding windows of 100 frames (~2.3 seconds)
    window_size = 100
    energy_variance_windows = []
    zcr_mean_windows = []
    
    for i in range(0, len(rms) - window_size, window_size // 2):
        window_rms = rms[i:i+window_size]
        window_zcr = zcr[i:i+window_size]
        
        energy_variance_windows.append(np.var(window_rms))
        zcr_mean_windows.append(np.mean(window_zcr))
    
    # 4. Calculate correlations
    print("\n=== CORRELATION ANALYSIS ===")
    print("\n1. Frame-level correlations:")
    
    # RMS vs ZCR (frame-level)
    pearson_rms_zcr, p_val_rms = pearsonr(rms, zcr)
    spearman_rms_zcr, _ = spearmanr(rms, zcr)
    print(f"   RMS Energy vs ZCR:")
    print(f"   - Pearson correlation: {pearson_rms_zcr:.3f} (p={p_val_rms:.4f})")
    print(f"   - Spearman correlation: {spearman_rms_zcr:.3f}")
    
    # 5. Window-level correlations
    print("\n2. Window-level correlations:")
    if len(energy_variance_windows) > 1:
        pearson_var_zcr, p_val_var = pearsonr(energy_variance_windows, zcr_mean_windows)
        spearman_var_zcr, _ = spearmanr(energy_variance_windows, zcr_mean_windows)
        print(f"   Energy Variance vs Mean ZCR:")
        print(f"   - Pearson correlation: {pearson_var_zcr:.3f} (p={p_val_var:.4f})")
        print(f"   - Spearman correlation: {spearman_var_zcr:.3f}")
    
    # 6. What each metric captures
    print("\n=== SIGNAL CHARACTERISTICS ===")
    print("\n1. RMS Energy captures:")
    print("   - Overall loudness/amplitude")
    print("   - Power of the signal")
    print("   - Speaking intensity")
    
    print("\n2. Energy Variance captures:")
    print("   - Dynamic range within windows")
    print("   - Consistency of speaking volume")
    print("   - Emotional intensity changes")
    
    print("\n3. Zero-Crossing Rate captures:")
    print("   - Frequency content (high freq = more crossings)")
    print("   - Voiced vs unvoiced speech")
    print("   - Consonants (high ZCR) vs vowels (low ZCR)")
    print("   - Noisiness/breathiness of voice")
    
    # 7. Specific speech analysis
    print("\n=== SPEECH-SPECIFIC ANALYSIS ===")
    
    # Detect voiced segments (where pitch exists)
    pitches, magnitudes = librosa.piptrack(
        y=y, sr=sr, hop_length=hop_length,
        fmin=80, fmax=400
    )
    
    # Get frames with detected pitch (voiced speech)
    voiced_frames = np.max(pitches, axis=0) > 0
    
    if np.any(voiced_frames):
        # Compare ZCR in voiced vs unvoiced segments
        zcr_voiced = zcr[voiced_frames]
        zcr_unvoiced = zcr[~voiced_frames]
        
        print(f"\n   Voiced segments (vowels):")
        print(f"   - Mean ZCR: {np.mean(zcr_voiced):.4f}")
        print(f"   - Mean RMS: {np.mean(rms[voiced_frames]):.4f}")
        
        if len(zcr_unvoiced) > 0:
            print(f"\n   Unvoiced segments (consonants/silence):")
            print(f"   - Mean ZCR: {np.mean(zcr_unvoiced):.4f}")
            print(f"   - Mean RMS: {np.mean(rms[~voiced_frames]):.4f}")
        
        # Check if ZCR differentiates where energy doesn't
        print(f"\n   Discriminative power:")
        print(f"   - ZCR ratio (unvoiced/voiced): {np.mean(zcr_unvoiced)/np.mean(zcr_voiced):.2f}x")
        print(f"   - RMS ratio (unvoiced/voiced): {np.mean(rms[~voiced_frames])/np.mean(rms[voiced_frames]):.2f}x")
    
    # 8. Collinearity assessment
    print("\n=== COLLINEARITY ASSESSMENT ===")
    
    threshold = 0.7  # Common threshold for concerning collinearity
    
    if abs(pearson_rms_zcr) > threshold:
        print(f"\n⚠️  HIGH collinearity detected between RMS and ZCR (r={pearson_rms_zcr:.3f})")
        print("   Recommendation: EXCLUDE ZCR to avoid redundancy")
    elif abs(pearson_rms_zcr) > 0.5:
        print(f"\n⚡ MODERATE correlation between RMS and ZCR (r={pearson_rms_zcr:.3f})")
        print("   Recommendation: ZCR adds some unique information")
    else:
        print(f"\n✅ LOW correlation between RMS and ZCR (r={pearson_rms_zcr:.3f})")
        print("   Recommendation: ZCR captures different signal aspects")
    
    # Return results for further analysis
    return {
        'pearson_rms_zcr': pearson_rms_zcr,
        'spearman_rms_zcr': spearman_rms_zcr,
        'pearson_var_zcr': pearson_var_zcr if len(energy_variance_windows) > 1 else None,
        'zcr_discrimination': np.mean(zcr_unvoiced)/np.mean(zcr_voiced) if np.any(voiced_frames) and len(zcr_unvoiced) > 0 else None
    }

if __name__ == "__main__":
    # Test with the wellness video audio
    video_id = "7515687288257465630"
    audio_path = f"/tmp/rumiai_shared_audio/{video_id}_audio.wav"
    
    if not Path(audio_path).exists():
        print(f"Audio file not found at {audio_path}")
        print("Attempting to use test audio from downloads...")
        audio_path = f"downloads/{video_id}/{video_id}_audio.wav"
    
    if Path(audio_path).exists():
        print(f"Analyzing: {audio_path}")
        results = analyze_zcr_vs_energy(audio_path)
        
        print("\n" + "="*50)
        print("FINAL RECOMMENDATION")
        print("="*50)
        
        if results['pearson_rms_zcr'] is not None:
            if abs(results['pearson_rms_zcr']) > 0.7:
                print("\n❌ DO NOT include Zero-Crossing Rate")
                print("   Reason: High collinearity with existing energy metrics")
                print("   Impact: Would add noise without new information")
            elif results['zcr_discrimination'] and results['zcr_discrimination'] > 2.0:
                print("\n✅ INCLUDE Zero-Crossing Rate")
                print("   Reason: Strong discriminative power for speech characteristics")
                print("   Impact: Captures consonant/vowel patterns energy misses")
            else:
                print("\n⚡ OPTIONAL - Zero-Crossing Rate")
                print("   Reason: Moderate unique information")
                print("   Suggestion: Test with/without in ML model")
    else:
        print(f"\nError: Could not find audio file at {audio_path}")
        print("\nTheoretical Analysis:")
        print("="*50)
        print("\nBased on signal processing theory:")
        print("\n1. Energy (RMS) measures:")
        print("   - Signal amplitude/power")
        print("   - Correlates with perceived loudness")
        print("\n2. Zero-Crossing Rate measures:")
        print("   - Frequency content")
        print("   - Spectral characteristics")
        print("\n3. Expected correlation: LOW to MODERATE")
        print("   - Different physical properties")
        print("   - Energy = amplitude domain")
        print("   - ZCR = frequency domain proxy")
        print("\n4. Recommendation: ZCR likely adds value for speech")
        print("   - Distinguishes fricatives from vowels")
        print("   - Captures articulation clarity")
        print("   - Not redundant with energy")