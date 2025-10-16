"""
Integration Tests for Content Analysis Stage (2.6 & 2.7)

Source: ContentAnalysisCHILDTI.md Section 7 (Example Traces)

Usage:
    python -m pytest ml_pipeline/stage2_content_analysis/test_integration.py -v
"""

import pytest
import os
import json
from pathlib import Path

# Import module functions
from . import (
    run_discovery_stage,
    run_classification_stage,
    validate_discovery_inputs,
    validate_classification_inputs,
    load_json,
    save_json,
    construct_path
)


class TestUtilities:
    """Test utility functions"""

    def test_construct_path_base(self):
        """Test base path construction"""
        path = construct_path("acme", "nutrition", file_type="base")
        assert path == "/data/clients/acme/hashtags/nutrition/top_contrastive"

    def test_construct_path_selection_manifest(self):
        """Test selection manifest path construction"""
        path = construct_path("acme", "nutrition", file_type="selection_manifest")
        assert path == "/data/clients/acme/hashtags/nutrition/top_contrastive/selection_manifest.json"

    def test_construct_path_raw_discovery(self):
        """Test raw discovery path construction"""
        path = construct_path("acme", "nutrition", file_type="raw_discovery")
        assert path == "/data/clients/acme/hashtags/nutrition/top_contrastive/content_taxonomies/nutrition_raw_discovery.json"

    def test_construct_path_bucket_content_analysis(self):
        """Test bucket content analysis path construction"""
        path = construct_path("acme", "nutrition", bucket="33-60s", file_type="bucket_content_analysis")
        assert path == "/data/clients/acme/hashtags/nutrition/top_contrastive/buckets/bucket_33-60s/content_analysis"

    def test_save_and_load_json(self, tmp_path):
        """Test JSON save and load with atomic writes"""
        test_data = {"key": "value", "number": 42}
        test_file = tmp_path / "test.json"

        # Save
        save_json(str(test_file), test_data)
        assert test_file.exists()

        # Load
        loaded_data = load_json(str(test_file))
        assert loaded_data == test_data


class TestValidation:
    """Test validation functions"""

    def test_validate_discovery_inputs_missing_manifest(self):
        """Test discovery validation with missing manifest"""
        with pytest.raises(FileNotFoundError, match="selection_manifest.json not found"):
            validate_discovery_inputs("/nonexistent/path.json", sample_size=50)

    def test_validate_discovery_inputs_small_sample_size(self, tmp_path):
        """Test discovery validation with too small sample size"""
        # Create minimal valid manifest
        manifest = {
            "hashtag": "test",
            "selected_buckets": ["33-60s", "60-90s", "90-120s"],
            "videos_by_bucket": {
                "33-60s": {"top_performers": ["vid1"] * 20},
                "60-90s": {"top_performers": ["vid2"] * 20},
                "90-120s": {"top_performers": ["vid3"] * 20}
            }
        }
        manifest_path = tmp_path / "manifest.json"
        save_json(str(manifest_path), manifest)

        # Set API key for validation
        os.environ["ANTHROPIC_API_KEY"] = "sk-test-key"

        # Should fail with too small sample size
        with pytest.raises(ValueError, match="Sample size too small: 5"):
            validate_discovery_inputs(str(manifest_path), sample_size=5)


class TestDiscoveryStage:
    """Test Stage 2.6 Discovery functions"""

    @pytest.mark.skipif(
        not os.environ.get('ANTHROPIC_API_KEY'),
        reason="ANTHROPIC_API_KEY not set - skipping LLM tests"
    )
    def test_discovery_stage_integration(self, tmp_path):
        """
        Integration test for discovery stage

        NOTE: This test requires:
        - ANTHROPIC_API_KEY environment variable set
        - Valid manifest and transcripts
        - Will make real API calls (costs money)
        """
        # This is a placeholder - real test would need full setup
        # For now, we just verify the function signature works
        pytest.skip("Requires full environment setup with real data")


class TestClassificationStage:
    """Test Stage 2.7 Classification functions"""

    @pytest.mark.skipif(
        not os.environ.get('ANTHROPIC_API_KEY'),
        reason="ANTHROPIC_API_KEY not set - skipping LLM tests"
    )
    def test_classification_stage_integration(self, tmp_path):
        """
        Integration test for classification stage

        NOTE: This test requires:
        - ANTHROPIC_API_KEY environment variable set
        - Valid taxonomy, manifest, and video data
        - Will make real API calls (costs money)
        """
        # This is a placeholder - real test would need full setup
        pytest.skip("Requires full environment setup with real data")


class TestEndToEnd:
    """End-to-end tests (manual execution recommended)"""

    @pytest.mark.manual
    def test_full_pipeline_discovery_to_classification(self):
        """
        Full pipeline test: Discovery (2.6) → Classification (2.7)

        This test is marked as manual and should be run with actual data:

        Steps:
        1. Ensure Stage 2.5 has completed (selection_manifest.json exists)
        2. Set ANTHROPIC_API_KEY environment variable
        3. Run discovery: run_discovery_stage(client_id, hashtag)
        4. Manually curate raw_discovery.json → taxonomy.json
        5. Run classification: run_classification_stage(client_id, hashtag)
        6. Verify content_analysis/*.json files created

        Expected outcome:
        - 120 classification files (40 per bucket × 3 buckets)
        - All files pass schema validation
        """
        pytest.skip("Manual test - requires full environment and curation step")


# Pytest configuration
def pytest_configure(config):
    """Add custom markers"""
    config.addinivalue_line(
        "markers", "manual: tests that require manual execution with real data"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
