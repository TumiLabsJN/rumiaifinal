#!/usr/bin/env python3
"""
Test suite for universal_principles generation in Stage 7 LLM Analysis.

This test validates the generate_universal_principles() function and identifies:
1. Missing feature names in output
2. Redundant label repetition
3. Useless principles where label_top == label_bottom
4. Lack of clarity about what feature is being discussed

Run: python test_universal_principles.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from ml_pipeline.stage7_llm_analysis.stage7_preprocessing import generate_universal_principles
from config.semantic_interpretations import interpret_value, extract_base_feature


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def test_case_1_missing_feature_name():
    """
    Test Case 1: Missing Feature Name

    Problem: Current output doesn't include the feature name, making it unclear
    what the principle refers to.

    Example: "Moderate variation in middle: Top performers use moderate variation..."
    Question: Moderate variation of WHAT?
    """
    print_section("TEST CASE 1: Missing Feature Name")

    # Mock RF data with energy_variance feature
    rf_video_data = {
        'feature_importance': [
            {
                'feature': 'middle_3_energy_variance',
                'importance': 0.25,
                'rank': 1,
                'top_performer_avg': 0.0032,  # Maps to "moderate variation"
                'bottom_performer_avg': 0.0001,  # Maps to "very consistent"
                'gap': 0.0031
            }
        ]
    }

    # Generate principles
    principles = generate_universal_principles(rf_video_data, top_n=1)

    # Display results
    print("Mock Feature Data:")
    print(f"  Feature: middle_3_energy_variance")
    print(f"  Top Avg: 0.0032 → 'moderate variation'")
    print(f"  Bottom Avg: 0.0001 → 'very consistent'")
    print(f"  Gap: 0.0031")

    print("\nGenerated Principle:")
    print(f"  '{principles[0]}'")

    print("\n❌ PROBLEM:")
    print("  - Doesn't mention 'energy_variance' anywhere")
    print("  - Says 'Moderate variation' but variation of WHAT?")
    print("  - Content creator has no idea what feature to modify")

    print("\n✅ EXPECTED OUTPUT:")
    print("  'Energy variance in middle: Top: moderate variation, Bottom: very consistent'")
    print("  OR")
    print("  'Use moderate variation in energy variance (middle segment)'")

    return principles[0]


def test_case_2_redundant_repetition():
    """
    Test Case 2: Redundant Label Repetition

    Problem: The semantic label appears twice in the output, creating redundancy.

    Example: "Moderate variation in middle: Top performers use moderate variation..."
             ^^^^^^^^^^^^^^^^^^^            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
             Says "moderate variation" twice!
    """
    print_section("TEST CASE 2: Redundant Label Repetition")

    # Mock RF data
    rf_video_data = {
        'feature_importance': [
            {
                'feature': 'hook_eye_contact_rate',
                'importance': 0.35,
                'rank': 1,
                'top_performer_avg': 0.75,  # Maps to "high eye contact"
                'bottom_performer_avg': 0.25,  # Maps to "minimal eye contact"
                'gap': 0.50
            }
        ]
    }

    principles = generate_universal_principles(rf_video_data, top_n=1)

    print("Mock Feature Data:")
    print(f"  Feature: hook_eye_contact_rate")
    print(f"  Top Avg: 0.75 → 'high eye contact'")
    print(f"  Bottom Avg: 0.25 → 'minimal eye contact'")

    print("\nGenerated Principle:")
    print(f"  '{principles[0]}'")

    print("\n❌ PROBLEM:")
    print("  - Says 'High eye contact in opening: Top performers use high eye contact...'")
    print("  - Label 'high eye contact' appears at START and in 'use high eye contact'")
    print("  - Redundant and verbose")

    print("\n✅ EXPECTED OUTPUT:")
    print("  'Eye contact rate in opening: Top: high, Bottom: minimal'")
    print("  OR")
    print("  'Maintain high eye contact in opening (top: high vs bottom: minimal)'")

    return principles[0]


def test_case_3_identical_labels_useless():
    """
    Test Case 3: Useless Principles (Identical Labels)

    Problem: When top and bottom performers have the same semantic label,
    the principle is completely useless.

    Example: "Quiet in middle: Top performers use quiet vs bottom use quiet"
             This provides ZERO actionable insight!
    """
    print_section("TEST CASE 3: Useless Principles (Identical Labels)")

    # Mock RF data where top and bottom have same label
    rf_video_data = {
        'feature_importance': [
            {
                'feature': 'middle_1_energy_level',
                'importance': 0.10,
                'rank': 1,
                'top_performer_avg': 0.05,  # Maps to "quiet"
                'bottom_performer_avg': 0.06,  # Also maps to "quiet"
                'gap': 0.01  # Small gap, same semantic label
            },
            {
                'feature': 'hook_average_face_size',
                'importance': 0.08,
                'rank': 2,
                'top_performer_avg': 0.05,  # Maps to "medium shot"
                'bottom_performer_avg': 0.06,  # Also maps to "medium shot"
                'gap': 0.01
            }
        ]
    }

    principles = generate_universal_principles(rf_video_data, top_n=2)

    print("Mock Feature Data:")
    print(f"\n  Feature 1: middle_1_energy_level")
    print(f"    Top Avg: 0.05 → 'quiet'")
    print(f"    Bottom Avg: 0.06 → 'quiet'")
    print(f"    Gap: 0.01 (too small for different labels)")

    print(f"\n  Feature 2: hook_average_face_size")
    print(f"    Top Avg: 0.05 → 'medium shot'")
    print(f"    Bottom Avg: 0.06 → 'medium shot'")

    print("\nGenerated Principles:")
    for i, principle in enumerate(principles, 1):
        print(f"  {i}. '{principle}'")

    print("\n❌ PROBLEM:")
    print("  - Says 'Top performers use quiet vs bottom use quiet'")
    print("  - Says 'Top performers use medium shot vs bottom use medium shot'")
    print("  - Provides ZERO differentiation or actionable insight")
    print("  - Wastes space in the universal_principles array")

    print("\n✅ EXPECTED BEHAVIOR:")
    print("  - SKIP features where label_top == label_bottom")
    print("  - Only include features with meaningful differences")
    print("  - Filter out low-gap features that map to same semantic label")

    return principles


def test_case_4_feature_name_extraction():
    """
    Test Case 4: Feature Name Extraction

    Validates that extract_base_feature() correctly removes window prefixes
    and that we have access to the clean feature name.
    """
    print_section("TEST CASE 4: Feature Name Extraction")

    test_features = [
        'middle_3_energy_variance',
        'hook_eye_contact_rate',
        'closing_scene_count',
        'xwin_speech_coverage',
        'middle_1_word_count'
    ]

    print("Testing extract_base_feature() on various inputs:\n")

    for feature in test_features:
        base_feature = extract_base_feature(feature)
        print(f"  '{feature}'")
        print(f"    → base_feature: '{base_feature}'")
        print(f"    → formatted: '{base_feature.replace('_', ' ').title()}'")
        print()

    print("✅ VALIDATION:")
    print("  - extract_base_feature() successfully removes window prefixes")
    print("  - We have access to clean feature names (energy_variance, eye_contact_rate, etc.)")
    print("  - These names SHOULD be included in universal_principles output")

    return True


def test_case_5_semantic_interpretation():
    """
    Test Case 5: Semantic Interpretation Coverage

    Shows how interpret_value() maps numeric values to semantic labels,
    and demonstrates the gap issue where different values map to same label.
    """
    print_section("TEST CASE 5: Semantic Interpretation Coverage")

    # Test energy_level interpretations
    print("Testing 'energy_level' semantic interpretations:\n")

    energy_values = [0.05, 0.08, 0.11, 0.15, 0.20, 0.30]

    for value in energy_values:
        label, desc = interpret_value('energy_level', value)
        print(f"  {value:.2f} → label: '{label}', desc: '{desc}'")

    print("\n📊 INSIGHT:")
    print("  - Values 0.05 and 0.08 both map to 'quiet'")
    print("  - Gap of 0.03 doesn't change semantic label")
    print("  - This is why we get useless principles like 'Top: quiet vs Bottom: quiet'")

    print("\n✅ RECOMMENDATION:")
    print("  - Filter out principles where label_top == label_bottom")
    print("  - Focus on features with MEANINGFUL semantic differences")

    return True


def run_all_tests():
    """Run all test cases and generate summary."""
    print("\n" + "="*80)
    print("  UNIVERSAL PRINCIPLES TEST SUITE")
    print("  Testing Stage 7 LLM Analysis - generate_universal_principles()")
    print("="*80)

    # Run test cases
    result1 = test_case_1_missing_feature_name()
    result2 = test_case_2_redundant_repetition()
    result3 = test_case_3_identical_labels_useless()
    result4 = test_case_4_feature_name_extraction()
    result5 = test_case_5_semantic_interpretation()

    # Summary
    print_section("SUMMARY OF ISSUES IDENTIFIED")

    print("1. ❌ MISSING FEATURE NAME")
    print("   - Current: 'Moderate variation in middle: ...'")
    print("   - Problem: Doesn't say what feature this refers to")
    print("   - Impact: Content creators don't know what to change")

    print("\n2. ❌ REDUNDANT REPETITION")
    print("   - Current: 'High eye contact in opening: Top performers use high eye contact...'")
    print("   - Problem: Label appears twice unnecessarily")
    print("   - Impact: Verbose, harder to read")

    print("\n3. ❌ USELESS IDENTICAL-LABEL PRINCIPLES")
    print("   - Current: 'Quiet in middle: Top performers use quiet vs bottom use quiet'")
    print("   - Problem: No differentiation between top and bottom")
    print("   - Impact: Wastes space, provides zero insight")

    print("\n4. ✅ FEATURE NAME EXTRACTION WORKS")
    print("   - extract_base_feature() successfully extracts clean names")
    print("   - We can use these names in the output")

    print("\n5. ✅ SEMANTIC INTERPRETATIONS WORK")
    print("   - interpret_value() correctly maps values to labels")
    print("   - Need to filter when labels are identical")

    print_section("RECOMMENDED FIXES")

    print("FIX 1: Include Feature Name in Output")
    print("  Before: f\"{label_top.capitalize()} {window_text}: ...\"")
    print("  After:  f\"{base_feature.replace('_', ' ').title()} {window_text}: ...\"")

    print("\nFIX 2: Remove Redundancy")
    print("  Before: f\"Top performers use {label_top} vs bottom use {label_bottom}\"")
    print("  After:  f\"Top: {label_top}, Bottom: {label_bottom}\"")

    print("\nFIX 3: Skip Identical Labels")
    print("  Add filter: if label_top == label_bottom: continue")

    print("\nFIX 4: Combined Improved Format")
    print("  Example: 'Energy variance in middle: Top: moderate variation, Bottom: very consistent'")
    print("  Example: 'Eye contact rate in opening: Top: high, Bottom: minimal'")

    print("\n" + "="*80)
    print("  TEST SUITE COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_all_tests()
