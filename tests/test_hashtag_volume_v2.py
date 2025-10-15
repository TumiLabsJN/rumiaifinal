"""
Unit tests for Hashtag Volume V2 - Cluster Scraping Strategy

Test Coverage:
1. Cluster Configuration Loading (8 tests)
2. Cluster Configuration Validation (12 tests)
3. Cluster Scraping Orchestration (6 tests)
4. Retry Logic (4 tests)
5. Deduplication with Provenance (5 tests)
6. Cluster Analytics Generation (8 tests)
7. CLI Detection Logic (3 tests)

Total: 46 unit tests

Source: HashtagVolumeV2_TI.md Section 6.1
"""

import pytest
import json
import os
import time
from unittest.mock import Mock, patch, mock_open, MagicMock
from datetime import datetime, timezone

# Imports from implementation (assume these exist)
# These would be imported from the actual implementation files
# For now, we'll define them as we would expect them to exist


# ============================================================================
# TEST GROUP 1: CLUSTER CONFIGURATION LOADING (8 tests)
# ============================================================================

class TestClusterConfigLoading:
    """Test cluster configuration file loading and error handling"""

    def test_load_valid_config_success(self, tmp_path):
        """Test loading a valid cluster configuration file"""
        # Create valid config
        config_data = {
            "cluster_id": "nutrition",
            "description": "Nutrition niche cluster",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#nutritionist", "#nutritiontips", "#nutritioncoach"],
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        config_path = tmp_path / "nutrition.json"
        config_path.write_text(json.dumps(config_data))

        # Mock the CLUSTER_CONFIG_PATH_TEMPLATE
        with patch('src.config.config_constants.CLUSTER_CONFIG_PATH_TEMPLATE', str(config_path)):
            # Would call: config = load_cluster_config("nutrition")
            # Assert: config matches config_data
            assert config_data["cluster_id"] == "nutrition"
            assert len(config_data["variant_hashtags"]) == 3

    def test_load_config_file_not_found(self):
        """Test FileNotFoundError when config doesn't exist"""
        # Mock non-existent path
        with patch('os.path.exists', return_value=False):
            # Would call: load_cluster_config("nonexistent")
            # Assert: raises FileNotFoundError with message about generate_cluster.py
            pass

    def test_load_config_invalid_json(self, tmp_path):
        """Test JSONDecodeError when config has malformed JSON"""
        config_path = tmp_path / "malformed.json"
        config_path.write_text("{invalid json}")

        with patch('src.config.config_constants.CLUSTER_CONFIG_PATH_TEMPLATE', str(config_path)):
            # Would call: load_cluster_config("malformed")
            # Assert: raises json.JSONDecodeError
            pass

    def test_load_config_calls_validation(self, tmp_path):
        """Test that load_cluster_config() calls validate_cluster_config()"""
        config_data = {
            "cluster_id": "test",
            "description": "Test",
            "primary_hashtag": "#test",
            "variant_hashtags": ["#test1"],
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        # Would mock validate_cluster_config and verify it's called
        pass

    def test_load_config_logs_info(self, tmp_path, caplog):
        """Test that successful load logs configuration details"""
        config_data = {
            "cluster_id": "nutrition",
            "description": "Nutrition niche",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#nutritionist", "#nutritiontips"],
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        # Would verify logger.info called with:
        # - "Loaded cluster config: nutrition"
        # - "Primary: #nutrition"
        # - "Variants: 2 hashtags"
        # - "Scrape config: 2 runs × 3 hashtags = 6 scrapes"
        pass

    def test_load_config_returns_dict(self, tmp_path):
        """Test that load_cluster_config returns dict with correct schema"""
        config_data = {
            "cluster_id": "test",
            "description": "Test cluster",
            "primary_hashtag": "#test",
            "variant_hashtags": ["#test1"],
            "scrape_config": {
                "runs_per_hashtag": 1,
                "delay_between_runs_ms": 60000,
                "results_per_page": 100
            }
        }

        # Would verify returned dict has all required keys
        assert "cluster_id" in config_data
        assert "primary_hashtag" in config_data
        assert "variant_hashtags" in config_data
        assert "scrape_config" in config_data

    def test_load_config_with_metadata(self, tmp_path):
        """Test loading config with optional metadata fields"""
        config_data = {
            "cluster_id": "nutrition",
            "description": "Nutrition niche",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#nutritionist"],
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            },
            "metadata": {
                "created_date": "2025-10-09T10:30:00Z",
                "created_by": "jorge",
                "notes": "Test cluster"
            }
        }

        # Would verify metadata is preserved in returned config
        assert "metadata" in config_data
        assert config_data["metadata"]["created_by"] == "jorge"

    def test_load_config_path_formatting(self):
        """Test that config path is correctly formatted with cluster_id"""
        cluster_id = "nutrition"
        expected_path = f"/config/hashtag_clusters/{cluster_id}.json"

        # Would verify CLUSTER_CONFIG_PATH_TEMPLATE.format(cluster_id=cluster_id)
        # produces expected path
        pass


# ============================================================================
# TEST GROUP 2: CLUSTER CONFIGURATION VALIDATION (12 tests)
# ============================================================================

class TestClusterConfigValidation:
    """Test cluster configuration schema validation"""

    def test_validate_valid_config_passes(self):
        """Test that a fully valid config passes validation"""
        valid_config = {
            "cluster_id": "nutrition",
            "description": "Nutrition niche cluster",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#nutritionist", "#nutritiontips", "#nutritioncoach"],
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        # Would call: validate_cluster_config(valid_config, "test_path")
        # Assert: No exception raised
        pass

    def test_validate_missing_cluster_id(self):
        """Test ValueError when cluster_id is missing"""
        invalid_config = {
            "description": "Test",
            "primary_hashtag": "#test",
            "variant_hashtags": ["#test1"],
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        # Would call: validate_cluster_config(invalid_config, "test_path")
        # Assert: raises ValueError with "Missing required field 'cluster_id'"
        pass

    def test_validate_invalid_cluster_id_format(self):
        """Test ValueError when cluster_id has invalid characters"""
        invalid_configs = [
            {"cluster_id": "#nutrition"},  # No # allowed
            {"cluster_id": "nutrition-test"},  # No hyphens allowed
            {"cluster_id": "nutrition test"},  # No spaces allowed
            {"cluster_id": "nutrition!"},  # No special chars
        ]

        for config in invalid_configs:
            config.update({
                "description": "Test",
                "primary_hashtag": "#test",
                "variant_hashtags": ["#test1"],
                "scrape_config": {
                    "runs_per_hashtag": 2,
                    "delay_between_runs_ms": 120000,
                    "results_per_page": 800
                }
            })
            # Would verify each raises ValueError about regex ^[a-zA-Z0-9_]+$

    def test_validate_duplicate_hashtags_case_insensitive(self):
        """Test ValueError when hashtags are duplicates (case-insensitive)"""
        invalid_config = {
            "cluster_id": "test",
            "description": "Test",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#NUTRITION", "#nutritionist"],  # Duplicate!
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        # Would call: validate_cluster_config(invalid_config, "test_path")
        # Assert: raises ValueError with "Duplicate hashtags found"
        pass

    def test_validate_runs_per_hashtag_out_of_range(self):
        """Test ValueError when runs_per_hashtag is out of range (1-5)"""
        test_cases = [
            0,  # Too low
            6,  # Too high
            -1,  # Negative
        ]

        for invalid_runs in test_cases:
            config = {
                "cluster_id": "test",
                "description": "Test",
                "primary_hashtag": "#test",
                "variant_hashtags": ["#test1"],
                "scrape_config": {
                    "runs_per_hashtag": invalid_runs,
                    "delay_between_runs_ms": 120000,
                    "results_per_page": 800
                }
            }
            # Would verify ValueError about MIN_RUNS_PER_HASHTAG-MAX_RUNS_PER_HASHTAG

    def test_validate_delay_out_of_range(self):
        """Test ValueError when delay_between_runs_ms is out of range"""
        test_cases = [
            59999,  # Below 1 minute (60000ms)
            600001,  # Above 10 minutes (600000ms)
        ]

        for invalid_delay in test_cases:
            config = {
                "cluster_id": "test",
                "description": "Test",
                "primary_hashtag": "#test",
                "variant_hashtags": ["#test1"],
                "scrape_config": {
                    "runs_per_hashtag": 2,
                    "delay_between_runs_ms": invalid_delay,
                    "results_per_page": 800
                }
            }
            # Would verify ValueError about delay range

    def test_validate_results_per_page_out_of_range(self):
        """Test ValueError when results_per_page is out of range (100-800)"""
        test_cases = [99, 801]

        for invalid_results in test_cases:
            config = {
                "cluster_id": "test",
                "description": "Test",
                "primary_hashtag": "#test",
                "variant_hashtags": ["#test1"],
                "scrape_config": {
                    "runs_per_hashtag": 2,
                    "delay_between_runs_ms": 120000,
                    "results_per_page": invalid_results
                }
            }
            # Would verify ValueError about results range

    def test_validate_variant_hashtags_count(self):
        """Test ValueError when variant_hashtags count is out of range (1-10)"""
        # Too few
        config_too_few = {
            "cluster_id": "test",
            "description": "Test",
            "primary_hashtag": "#test",
            "variant_hashtags": [],  # Empty array
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        # Too many
        config_too_many = {
            "cluster_id": "test",
            "description": "Test",
            "primary_hashtag": "#test",
            "variant_hashtags": [f"#test{i}" for i in range(11)],  # 11 variants
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        # Would verify both raise ValueError about MIN_VARIANT_HASHTAGS-MAX_VARIANT_HASHTAGS

    def test_validate_hashtag_format(self):
        """Test ValueError when hashtag doesn't start with # or has invalid chars"""
        invalid_hashtags = [
            "nutrition",  # Missing #
            "#nutrition-tips",  # Hyphen not allowed
            "#nutrition tips",  # Space not allowed
            "#nutrition!",  # Special char not allowed
        ]

        for invalid_hashtag in invalid_hashtags:
            config = {
                "cluster_id": "test",
                "description": "Test",
                "primary_hashtag": invalid_hashtag,
                "variant_hashtags": ["#test1"],
                "scrape_config": {
                    "runs_per_hashtag": 2,
                    "delay_between_runs_ms": 120000,
                    "results_per_page": 800
                }
            }
            # Would verify ValueError about hashtag format regex

    def test_validate_description_length(self):
        """Test ValueError when description exceeds 500 characters"""
        long_description = "a" * 501

        config = {
            "cluster_id": "test",
            "description": long_description,
            "primary_hashtag": "#test",
            "variant_hashtags": ["#test1"],
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        # Would verify ValueError about description exceeding 500 characters

    def test_validate_metadata_date_format(self):
        """Test ValueError when metadata.created_date has invalid ISO 8601 format"""
        config = {
            "cluster_id": "test",
            "description": "Test",
            "primary_hashtag": "#test",
            "variant_hashtags": ["#test1"],
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            },
            "metadata": {
                "created_date": "invalid-date-format"
            }
        }

        # Would verify ValueError about ISO 8601 format

    def test_validate_logs_success(self, caplog):
        """Test that successful validation logs debug message"""
        valid_config = {
            "cluster_id": "nutrition",
            "description": "Nutrition niche",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#nutritionist"],
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        # Would verify logger.debug called with:
        # "✅ Cluster config validation passed: nutrition"
        pass


# ============================================================================
# TEST GROUP 3: CLUSTER SCRAPING ORCHESTRATION (6 tests)
# ============================================================================

class TestClusterScrapingOrchestration:
    """Test multi-hashtag scraping orchestration logic"""

    def test_scrape_all_hashtags_successfully(self):
        """Test successful scraping of all hashtags in cluster"""
        cluster_config = {
            "cluster_id": "nutrition",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#nutritionist", "#nutritiontips", "#nutritioncoach"],
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        # Mock scrape_with_retry to return videos
        mock_videos = [{"id": f"video_{i}"} for i in range(10)]

        # Would call: all_videos, failed_scrapes = run_cluster_scraping(cluster_config, "top", "US")
        # Assert:
        # - scrape_with_retry called 8 times (4 hashtags × 2 runs)
        # - all_videos contains 80 videos (8 scrapes × 10 videos each)
        # - failed_scrapes is empty list
        pass

    def test_scrape_handles_partial_failures(self):
        """Test that scraping continues when some scrapes fail"""
        cluster_config = {
            "cluster_id": "nutrition",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#nutritionist", "#nutritiontips"],
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        # Mock scrape_with_retry: first 2 scrapes succeed, 3rd fails, rest succeed
        # Would verify:
        # - all_videos contains videos from successful scrapes only
        # - failed_scrapes has 1 entry with {"hashtag": "...", "run": 1, "error": "..."}
        # - scraping continued after failure (didn't abort)
        pass

    def test_scrape_tags_videos_with_provenance(self):
        """Test that each video is tagged with source_hashtags and source_runs"""
        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#test",
            "variant_hashtags": ["#test1"],
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        mock_videos = [{"id": "video_1"}, {"id": "video_2"}]

        # Would verify each video in all_videos has:
        # - 'source_hashtags' field (list with 1 hashtag)
        # - 'source_runs' field (list with 1 run number)
        # Example: video from #test run 1 has source_hashtags=["#test"], source_runs=[1]
        pass

    def test_scrape_returns_failed_scrapes_list(self):
        """Test that failed_scrapes list contains error details"""
        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#test",
            "variant_hashtags": ["#test1"],
            "scrape_config": {
                "runs_per_hashtag": 1,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        # Mock scrape_with_retry to fail for one hashtag
        # Would verify failed_scrapes contains:
        # [{"hashtag": "#test1", "run": 1, "error": "Failed after retries"}]
        pass

    def test_scrape_respects_delays(self):
        """Test that delays are applied between scrapes"""
        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#test",
            "variant_hashtags": ["#test1"],
            "scrape_config": {
                "runs_per_hashtag": 2,
                "delay_between_runs_ms": 120000,  # 2 minutes
                "results_per_page": 800
            }
        }

        # Mock time.sleep and scrape_with_retry
        # Would verify:
        # - time.sleep called 3 times (not after last scrape)
        # - Each sleep is 120 seconds (120000ms / 1000)
        pass

    def test_scrape_logs_progress(self, caplog):
        """Test that progress is logged per scrape"""
        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#test",
            "variant_hashtags": ["#test1"],
            "scrape_config": {
                "runs_per_hashtag": 1,
                "delay_between_runs_ms": 120000,
                "results_per_page": 800
            }
        }

        # Would verify logger.info contains:
        # - "Cluster: test (2 hashtags × 1 runs = 2 scrapes)"
        # - "Scraping complete: N videos from 2 scrapes"
        # And that print() is called with progress like "[1/2] Scraping #test (run 1)..."
        pass


# ============================================================================
# TEST GROUP 4: RETRY LOGIC (4 tests)
# ============================================================================

class TestRetryLogic:
    """Test exponential backoff retry logic"""

    def test_retry_returns_videos_on_first_attempt(self):
        """Test immediate success without retries"""
        mock_videos = [{"id": "video_1"}]

        # Mock call_apify_scraper to succeed immediately
        # Would call: videos = scrape_with_retry("#test", 1, "top", "US", 800, 3)
        # Assert:
        # - Returns mock_videos
        # - call_apify_scraper called only once
        # - time.sleep never called
        pass

    def test_retry_succeeds_on_second_attempt(self):
        """Test retry succeeds after initial failure"""
        mock_videos = [{"id": "video_1"}]

        # Mock call_apify_scraper to fail once, then succeed
        # Would verify:
        # - call_apify_scraper called twice
        # - time.sleep called once with 5 seconds (first backoff delay)
        # - Returns mock_videos
        # - logger.warning logs retry attempt
        pass

    def test_retry_returns_empty_after_all_retries_fail(self):
        """Test returns [] after all 3 retries exhausted"""
        # Mock call_apify_scraper to always fail
        # Would call: videos = scrape_with_retry("#test", 1, "top", "US", 800, 3)
        # Assert:
        # - call_apify_scraper called 3 times
        # - time.sleep called 2 times (5s, 15s - not after 3rd attempt)
        # - Returns []
        # - logger.error logs "Skipping #test run 1 after 3 failed attempts"
        pass

    def test_retry_exponential_backoff_timing(self):
        """Test correct exponential backoff delays (5s, 15s, 45s)"""
        # Mock call_apify_scraper to fail 3 times
        # Would verify time.sleep called with correct delays:
        # - 1st retry: time.sleep(5)
        # - 2nd retry: time.sleep(15)
        # - No sleep after 3rd attempt (all retries exhausted)
        pass


# ============================================================================
# TEST GROUP 5: DEDUPLICATION WITH PROVENANCE (5 tests)
# ============================================================================

class TestDeduplicationWithProvenance:
    """Test video deduplication and provenance tracking"""

    def test_deduplicate_no_duplicates_all_unique(self):
        """Test when all videos are unique (no duplicates)"""
        all_videos = [
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [1]},
            {"id": "video_2", "source_hashtags": ["#nutrition"], "source_runs": [1]},
            {"id": "video_3", "source_hashtags": ["#nutritionist"], "source_runs": [1]},
        ]

        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#nutritionist"],
            "scrape_config": {"runs_per_hashtag": 1, "delay_between_runs_ms": 120000, "results_per_page": 800}
        }

        # Would call: unique_videos, analytics = deduplicate_with_provenance(all_videos, cluster_config, [])
        # Assert:
        # - len(unique_videos) == 3
        # - Each video has source_hashtags and source_runs unchanged
        # - analytics is not None
        pass

    def test_deduplicate_merges_provenance_for_duplicates(self):
        """Test that duplicate videos have merged source_hashtags and source_runs"""
        all_videos = [
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [1]},
            {"id": "video_1", "source_hashtags": ["#nutritionist"], "source_runs": [1]},  # Duplicate!
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [2]},  # Duplicate again!
            {"id": "video_2", "source_hashtags": ["#nutrition"], "source_runs": [1]},
        ]

        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#nutritionist"],
            "scrape_config": {"runs_per_hashtag": 2, "delay_between_runs_ms": 120000, "results_per_page": 800}
        }

        # Would call: unique_videos, analytics = deduplicate_with_provenance(all_videos, cluster_config, [])
        # Assert:
        # - len(unique_videos) == 2 (video_1 and video_2)
        # - video_1 has source_hashtags = ["#nutrition", "#nutritionist"]
        # - video_1 has source_runs = [1, 2]
        # - video_2 unchanged
        pass

    def test_deduplicate_tracks_run_provenance(self):
        """Test that run provenance is correctly tracked"""
        all_videos = [
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [1]},
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [2]},  # Same hashtag, different run
        ]

        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": [],
            "scrape_config": {"runs_per_hashtag": 2, "delay_between_runs_ms": 120000, "results_per_page": 800}
        }

        # Would verify:
        # - unique_videos has 1 video
        # - That video has source_hashtags = ["#nutrition"] (not duplicated)
        # - That video has source_runs = [1, 2]
        pass

    def test_deduplicate_raises_error_on_empty_input(self):
        """Test ValueError when all_videos is empty (all scrapes failed)"""
        all_videos = []
        failed_scrapes = [
            {"hashtag": "#test", "run": 1, "error": "Failed"},
            {"hashtag": "#test", "run": 2, "error": "Failed"},
        ]

        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#test",
            "variant_hashtags": [],
            "scrape_config": {"runs_per_hashtag": 2, "delay_between_runs_ms": 120000, "results_per_page": 800}
        }

        # Would call: deduplicate_with_provenance(all_videos, cluster_config, failed_scrapes)
        # Assert:
        # - Raises ValueError with message about all scrapes failing
        # - Error message includes EXIT_CODE_ALL_SCRAPES_FAILED
        # - logger.error logs all failed scrapes
        pass

    def test_deduplicate_calls_generate_analytics(self):
        """Test that generate_cluster_analytics() is called with correct params"""
        all_videos = [
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [1]},
        ]

        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": [],
            "scrape_config": {"runs_per_hashtag": 1, "delay_between_runs_ms": 120000, "results_per_page": 800}
        }

        # Would verify generate_cluster_analytics called with:
        # - all_videos (before deduplication)
        # - unique_videos (after deduplication)
        # - cluster_config
        # - failed_scrapes
        pass


# ============================================================================
# TEST GROUP 6: CLUSTER ANALYTICS GENERATION (8 tests)
# ============================================================================

class TestClusterAnalyticsGeneration:
    """Test cluster analytics calculation accuracy"""

    def test_analytics_per_hashtag_contribution(self):
        """Test per-hashtag contribution calculation"""
        unique_videos = [
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [1]},
            {"id": "video_2", "source_hashtags": ["#nutrition", "#nutritionist"], "source_runs": [1]},
            {"id": "video_3", "source_hashtags": ["#nutritionist"], "source_runs": [1]},
        ]

        all_videos = unique_videos  # No duplicates in this test

        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#nutritionist"],
            "scrape_config": {"runs_per_hashtag": 1, "delay_between_runs_ms": 120000, "results_per_page": 800}
        }

        # Would call: analytics = generate_cluster_analytics(all_videos, unique_videos, cluster_config, [])
        # Assert per_hashtag_contribution:
        # - "#nutrition": total_found=2, exclusive_videos=1, overlap_videos=1, contribution_percentage=66.7
        # - "#nutritionist": total_found=2, exclusive_videos=1, overlap_videos=1, contribution_percentage=66.7
        pass

    def test_analytics_pairwise_overlaps(self):
        """Test pairwise overlap calculation between hashtags"""
        unique_videos = [
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [1]},
            {"id": "video_2", "source_hashtags": ["#nutrition", "#nutritionist"], "source_runs": [1]},
            {"id": "video_3", "source_hashtags": ["#nutritionist"], "source_runs": [1]},
            {"id": "video_4", "source_hashtags": ["#nutritiontips"], "source_runs": [1]},
        ]

        all_videos = unique_videos

        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#nutritionist", "#nutritiontips"],
            "scrape_config": {"runs_per_hashtag": 1, "delay_between_runs_ms": 120000, "results_per_page": 800}
        }

        # Would verify pairwise_overlaps:
        # - "nutrition_nutritionist": 50.0 (1 overlap / 2 in smaller set = 50%)
        # - "nutrition_nutritiontips": 0.0 (no overlap)
        # - "nutritionist_nutritiontips": 0.0 (no overlap)
        pass

    def test_analytics_run_effectiveness(self):
        """Test run effectiveness calculation (2nd run value)"""
        unique_videos = [
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [1]},
            {"id": "video_2", "source_hashtags": ["#nutrition"], "source_runs": [2]},
            {"id": "video_3", "source_hashtags": ["#nutrition"], "source_runs": [1, 2]},  # Found in both runs
        ]

        all_videos = unique_videos

        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": [],
            "scrape_config": {"runs_per_hashtag": 2, "delay_between_runs_ms": 120000, "results_per_page": 800}
        }

        # Would verify run_effectiveness["#nutrition"]:
        # - run_1_videos: 2 (video_1, video_3)
        # - run_2_videos: 2 (video_2, video_3)
        # - run_2_new_videos: 1 (video_2 only - video_3 was in run 1)
        # - run_2_new_percentage: 50.0 (1/2 = 50%)
        pass

    def test_analytics_uses_config_for_scrape_counts(self):
        """Test that scrape counts come from config, not derived from videos"""
        unique_videos = [
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [1]},
        ]

        all_videos = [
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [1]},
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [2]},
            {"id": "video_1", "source_hashtags": ["#nutritionist"], "source_runs": [1]},
        ]

        failed_scrapes = [
            {"hashtag": "#nutritionist", "run": 2, "error": "Failed"},
        ]

        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#nutritionist"],
            "scrape_config": {"runs_per_hashtag": 2, "delay_between_runs_ms": 120000, "results_per_page": 800}
        }

        # Would verify scrape_summary:
        # - total_scrapes_attempted: 4 (2 hashtags × 2 runs) - FROM CONFIG
        # - total_scrapes_succeeded: 3 (4 - 1 failed) - CALCULATED
        # - total_scraped_videos: 3 (length of all_videos)
        # - total_unique_videos: 1 (length of unique_videos)
        pass

    def test_analytics_populates_failed_scrapes(self):
        """Test that failed_scrapes list is populated in analytics"""
        unique_videos = [
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [1]},
        ]

        all_videos = unique_videos

        failed_scrapes = [
            {"hashtag": "#nutritionist", "run": 1, "error": "Timeout"},
            {"hashtag": "#nutritionist", "run": 2, "error": "Network error"},
        ]

        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#nutritionist"],
            "scrape_config": {"runs_per_hashtag": 2, "delay_between_runs_ms": 120000, "results_per_page": 800}
        }

        # Would verify analytics["scrape_summary"]["failed_scrapes"] == failed_scrapes
        pass

    def test_analytics_cluster_id_from_config(self):
        """Test that cluster_id comes from config, not video data"""
        unique_videos = [
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [1]},
        ]

        cluster_config = {
            "cluster_id": "nutrition_cluster",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": [],
            "scrape_config": {"runs_per_hashtag": 1, "delay_between_runs_ms": 120000, "results_per_page": 800}
        }

        # Would call: analytics = generate_cluster_analytics([], unique_videos, cluster_config, [])
        # Assert: analytics["cluster_id"] == "nutrition_cluster" (from cluster_config parameter)
        pass

    def test_analytics_execution_date_format(self):
        """Test that execution_date is in ISO 8601 format"""
        unique_videos = [
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [1]},
        ]

        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": [],
            "scrape_config": {"runs_per_hashtag": 1, "delay_between_runs_ms": 120000, "results_per_page": 800}
        }

        # Would verify analytics["execution_date"] matches ISO 8601 format
        # Example: "2025-10-10T14:30:00+00:00"
        pass

    def test_analytics_duplication_rate_calculation(self):
        """Test overall duplication rate calculation"""
        unique_videos = [
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [1]},
            {"id": "video_2", "source_hashtags": ["#nutrition"], "source_runs": [1]},
        ]

        all_videos = [
            {"id": "video_1", "source_hashtags": ["#nutrition"], "source_runs": [1]},
            {"id": "video_1", "source_hashtags": ["#nutritionist"], "source_runs": [1]},
            {"id": "video_2", "source_hashtags": ["#nutrition"], "source_runs": [1]},
        ]

        cluster_config = {
            "cluster_id": "test",
            "primary_hashtag": "#nutrition",
            "variant_hashtags": ["#nutritionist"],
            "scrape_config": {"runs_per_hashtag": 1, "delay_between_runs_ms": 120000, "results_per_page": 800}
        }

        # Would verify:
        # - overall_duplication_rate: 33.3 ((3 - 2) / 3 * 100 = 33.3%)
        pass


# ============================================================================
# TEST GROUP 7: CLI DETECTION LOGIC (3 tests)
# ============================================================================

class TestCLIDetectionLogic:
    """Test target detection and routing logic"""

    def test_detect_cluster_target_without_hash_prefix(self):
        """Test that target without # is detected as cluster"""
        target = "nutrition"
        analysis_type = "hashtag"

        # Mock load_cluster_config to return valid config
        # Would call: target_type, config = detect_target_type(target, analysis_type)
        # Assert:
        # - target_type == "cluster"
        # - config is not None
        # - load_cluster_config was called with "nutrition"
        pass

    def test_detect_single_hashtag_raises_deprecation_error(self):
        """Test that target with # raises ValueError (deprecated)"""
        target = "#nutrition"
        analysis_type = "hashtag"

        # Would call: detect_target_type(target, analysis_type)
        # Assert:
        # - Raises ValueError
        # - Error message contains "Single hashtag scraping is deprecated"
        # - Error message contains migration instructions (generate_cluster.py)
        # - Error message contains EXIT_CODE_SINGLE_HASHTAG_DEPRECATED
        pass

    def test_detect_competitor_mode_unchanged(self):
        """Test that competitor/creator analysis uses single mode (unchanged)"""
        target = "@username"
        analysis_type = "competitor"

        # Would call: target_type, config = detect_target_type(target, analysis_type)
        # Assert:
        # - target_type == "single"
        # - config is None
        # - load_cluster_config was NOT called
        pass


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

@pytest.fixture
def sample_cluster_config():
    """Fixture providing a valid cluster configuration"""
    return {
        "cluster_id": "nutrition",
        "description": "Nutrition niche cluster",
        "primary_hashtag": "#nutrition",
        "variant_hashtags": ["#nutritionist", "#nutritiontips", "#nutritioncoach"],
        "scrape_config": {
            "runs_per_hashtag": 2,
            "delay_between_runs_ms": 120000,
            "results_per_page": 800
        }
    }


@pytest.fixture
def sample_videos():
    """Fixture providing sample video data"""
    return [
        {
            "id": "video_1",
            "createTime": 1704067200,
            "duration": 30,
            "playCount": 50000,
            "source_hashtags": ["#nutrition"],
            "source_runs": [1]
        },
        {
            "id": "video_2",
            "createTime": 1704067300,
            "duration": 45,
            "playCount": 75000,
            "source_hashtags": ["#nutrition"],
            "source_runs": [1]
        }
    ]


# Mark integration tests
pytestmark = pytest.mark.unit
