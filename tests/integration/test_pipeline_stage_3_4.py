#!/usr/bin/env python3
"""
Integration Tests for Stage 3.4: Review CSV Generation

Source: PipelineIntegration_ReviewCSVGeneration.md Section 5
Test Data: /home/jorge/rumiaifinal/data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s/
"""

import pytest
import pandas as pd
from pathlib import Path

# Test bucket path
TEST_BUCKET = Path("/home/jorge/rumiaifinal/data/clients/test_run/hashtags/fitness/top_contrastive/buckets/bucket_18-33s/")


def test_stage_3_4_happy_path():
    """
    Test Stage 3.4 integration (happy path).
    Source: ReviewCSVGenerationCHILD.md Section 8.2
    """
    # Prerequisites: Stage 3 must have completed
    aggregated_csv = TEST_BUCKET / "ml_analysis" / "aggregated_features.csv"

    if not aggregated_csv.exists():
        pytest.skip(f"Stage 3 must complete before Stage 3.4. Missing: {aggregated_csv}")

    # Execute Stage 3.4
    from ml_pipeline.stage3_aggregation.review_csv_generator import generate_review_csv_for_bucket
    generate_review_csv_for_bucket(TEST_BUCKET)

    # Verify output
    output_path = TEST_BUCKET / "validation" / "video_review.csv"
    assert output_path.exists(), "video_review.csv not created"

    # Load and validate
    df_review = pd.read_csv(output_path)
    df_aggregated = pd.read_csv(aggregated_csv)

    print(f"\n✓ Review CSV: {len(df_review)} rows × {len(df_review.columns)} columns")
    print(f"✓ Aggregated CSV: {len(df_aggregated)} rows × {len(df_aggregated.columns)} columns")

    # Row count validation (HLD Section 8.2 line 700)
    assert len(df_review) <= len(df_aggregated), \
        f"Review CSV has more rows ({len(df_review)}) than aggregated CSV ({len(df_aggregated)})"

    # Allow 10% missing urls
    assert len(df_review) >= len(df_aggregated) * 0.9, \
        f"Review CSV has too few rows: {len(df_review)} < {len(df_aggregated) * 0.9} (90% threshold)"

    # Column count validation (HLD Section 8.2 line 704)
    expected_col_count = len(df_aggregated.columns) + 1  # +1 for url
    assert len(df_review.columns) == expected_col_count, \
        f"Column count mismatch: expected {expected_col_count} (aggregated + url), got {len(df_review.columns)}"

    # url column position validation (HLD Section 8.2 line 707)
    assert df_review.columns[1] == 'url', \
        f"url column not at position 2. Found: {df_review.columns[1]}"

    # url validity check (HLD Section 8.2 line 710)
    assert df_review['url'].str.startswith('https://').all(), \
        "Not all urls start with https://"

    print("✓ Stage 3.4 Integration Test PASSED")


def test_stage_3_4_skip_on_no_urls(tmp_path):
    """
    Test Stage 3.4 skips CSV generation when all videos missing url.
    Source: ReviewCSVGenerationCHILD.md Section 8.3 Edge Case 3
    """
    # Setup: Create test bucket with videos missing url
    test_bucket = tmp_path / "bucket_test_no_urls"
    insights_dir = test_bucket / "analysis" / "insights"
    insights_dir.mkdir(parents=True)

    # Create 3 test JSONs with null url
    for i in range(3):
        test_json = {
            "temporal_windows": {
                "hook": {"scene_count": 2, "word_count": 10},
                "middle_segments": None,
                "closing": {"scene_count": 1, "word_count": 5}
            },
            "metadata": {
                "video_id": f"742859641370714448{i}",
                "url": None,  # All videos have null url
                "duration": 7.5
            }
        }

        json_path = insights_dir / f"742859641370714448{i}_temporal_windows_updated.json"
        with open(json_path, 'w') as f:
            import json
            json.dump(test_json, f)

    # Execute: Should raise ValueError (all videos missing url)
    from ml_pipeline.stage3_aggregation.review_csv_generator import generate_review_csv_for_bucket

    with pytest.raises(ValueError, match="No videos with valid url"):
        generate_review_csv_for_bucket(test_bucket)

    # Verify: No output file created
    output_path = test_bucket / "validation" / "video_review.csv"
    assert not output_path.exists(), "video_review.csv should not be created when all urls missing"

    print("✓ Stage 3.4 Skip Test PASSED")


def test_stage_3_4_idempotent():
    """Test Stage 3.4 can re-run without side effects."""

    if not TEST_BUCKET.exists():
        pytest.skip(f"Test bucket does not exist: {TEST_BUCKET}")

    from ml_pipeline.stage3_aggregation.review_csv_generator import generate_review_csv_for_bucket

    # Run Stage 3.4 twice
    generate_review_csv_for_bucket(TEST_BUCKET)
    output_path = TEST_BUCKET / "validation" / "video_review.csv"

    if not output_path.exists():
        pytest.skip("video_review.csv not generated (all videos may be missing url)")

    # Load first result
    df1 = pd.read_csv(output_path)

    # Re-run Stage 3.4
    generate_review_csv_for_bucket(TEST_BUCKET)

    # Load second result
    df2 = pd.read_csv(output_path)

    # Verify identical results
    assert df1.equals(df2), "Stage 3.4 is not idempotent (results differ on re-run)"

    print("✓ Stage 3.4 Idempotency Test PASSED")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
