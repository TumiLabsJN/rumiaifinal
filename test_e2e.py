#!/usr/bin/env python3
"""
End-to-End test for RumiAI v2.

Tests the complete flow from URL to insights.
AUTOMATED - NO HUMAN INTERVENTION REQUIRED.
"""
import sys
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import logging

sys.path.insert(0, str(Path(__file__)))

from rumiai_v2.core.models import VideoMetadata, MLAnalysisResult, PromptResult, PromptType
from scripts.rumiai_runner import RumiAIRunner


async def test_full_pipeline():
    """Test complete pipeline with mocked external services."""
    print("🧪 RumiAI v2 End-to-End Test")
    print("=" * 50)
    
    # Mock all external services
    with patch('rumiai_v2.api.ApifyClient') as mock_apify, \
         patch('rumiai_v2.api.ClaudeClient') as mock_claude, \
         patch('rumiai_v2.api.MLServices') as mock_ml, \
         patch('rumiai_v2.config.Settings') as mock_settings:
        
        # Configure settings
        mock_settings.return_value.output_dir = Path("test_output")
        mock_settings.return_value.unified_dir = Path("test_output/unified")
        mock_settings.return_value.insights_dir = Path("test_output/insights")
        mock_settings.return_value.temporal_dir = Path("test_output/temporal")
        mock_settings.return_value.temp_dir = Path("test_output/temp")
        mock_settings.return_value.prompt_delay = 0
        mock_settings.return_value.apify_token = "test_token"
        mock_settings.return_value.claude_api_key = "test_key"
        mock_settings.return_value.claude_model = "claude-3-opus"
        mock_settings.return_value._prompt_templates = {}
        mock_settings.return_value.prompt_timeouts = {}
        
        # Mock Apify
        print("\n📹 Mocking video scraping...")
        mock_apify_instance = AsyncMock()
        mock_apify_instance.scrape_video = AsyncMock(return_value=VideoMetadata(
            video_id="e2e_test_123",
            url="https://tiktok.com/@testuser/video/e2e_test_123",
            download_url="https://fake-download.url/video.mp4",
            duration=47.5,
            description="End-to-end test video #test #rumiai",
            author="testuser",
            created_at="2024-07-08T10:00:00Z",
            stats={
                "plays": 1000000,
                "likes": 50000,
                "comments": 1000,
                "shares": 500
            }
        ))
        
        # Mock video download
        video_path = Path("test_output/temp/e2e_test_123.mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.touch()
        mock_apify_instance.download_video = AsyncMock(return_value=video_path)
        mock_apify.return_value = mock_apify_instance
        
        # Mock ML Services
        print("🤖 Mocking ML analysis...")
        mock_ml_instance = AsyncMock()
        
        # Create comprehensive ML results
        ml_results = {
            'scene_detection': MLAnalysisResult(
                model_name='scene_detection',
                success=True,
                data={
                    'scenes': [
                        {'start': 0.0, 'end': 5.0, 'type': 'intro'},
                        {'start': 5.0, 'end': 40.0, 'type': 'main'},
                        {'start': 40.0, 'end': 47.5, 'type': 'outro'}
                    ]
                },
                processing_time=1.5
            ),
            'speech_recognition': MLAnalysisResult(
                model_name='speech_recognition',
                success=True,
                data={
                    'transcripts': [
                        {'timestamp': 2.0, 'text': 'Welcome to the test', 'confidence': 0.95},
                        {'timestamp': 10.0, 'text': 'This is the main content', 'confidence': 0.92},
                        {'timestamp': 45.0, 'text': 'Thanks for watching', 'confidence': 0.94}
                    ],
                    'language': 'en'
                },
                processing_time=3.2
            ),
            'emotion_recognition': MLAnalysisResult(
                model_name='emotion_recognition',
                success=True,
                data={
                    'emotions': [
                        {'timestamp': 5.0, 'emotion': 'neutral', 'confidence': 0.85},
                        {'timestamp': 15.0, 'emotion': 'happy', 'confidence': 0.92},
                        {'timestamp': 30.0, 'emotion': 'excited', 'confidence': 0.88}
                    ]
                },
                processing_time=2.1
            ),
            'object_detection': MLAnalysisResult(
                model_name='object_detection',
                success=True,
                data={
                    'objects': [
                        {'timestamp': 10.0, 'object': 'person', 'confidence': 0.98},
                        {'timestamp': 20.0, 'object': 'product', 'confidence': 0.91},
                        {'timestamp': 35.0, 'object': 'logo', 'confidence': 0.87}
                    ]
                },
                processing_time=2.8
            )
        }
        
        mock_ml_instance.analyze_video = AsyncMock(return_value=ml_results)
        mock_ml.return_value = mock_ml_instance
        
        # Mock Claude responses
        print("🧠 Mocking Claude prompts...")
        mock_claude_instance = Mock()
        
        # Create different responses for each prompt type
        prompt_responses = {
            PromptType.CREATIVE_DENSITY: "High creative density with rapid transitions",
            PromptType.EMOTIONAL_JOURNEY: "Emotional arc from neutral to excitement",
            PromptType.SPEECH_ANALYSIS: "Clear speech with strong call-to-action",
            PromptType.VISUAL_OVERLAY: "Dynamic visual overlays enhance engagement",
            PromptType.METADATA_ANALYSIS: "Viral potential due to trending topic",
            PromptType.PERSON_FRAMING: "Centered framing with good eye contact",
            PromptType.SCENE_PACING: "Fast-paced editing maintains attention"
        }
        
        def mock_send_prompt(prompt_text, metadata, timeout=60):
            prompt_type_str = metadata.get('prompt_type', 'unknown')
            prompt_type = PromptType(prompt_type_str)
            
            return PromptResult(
                prompt_type=prompt_type,
                success=True,
                response=prompt_responses.get(prompt_type, "Generic analysis"),
                tokens_used=100 + len(prompt_text) // 10,
                estimated_cost=0.01,
                processing_time=1.0
            )
        
        mock_claude_instance.send_prompt = Mock(side_effect=mock_send_prompt)
        mock_claude.return_value = mock_claude_instance
        
        # Run the pipeline
        print("\n🚀 Running full pipeline...")
        runner = RumiAIRunner(legacy_mode=False)
        
        try:
            result = await runner.process_video_url("https://tiktok.com/@testuser/video/e2e_test_123")
            
            # Verify results
            print("\n📊 Verifying results...")
            assert result['success'], "Pipeline should succeed"
            assert result['video_id'] == 'e2e_test_123', "Video ID should match"
            
            # Check outputs
            assert 'outputs' in result, "Should have outputs"
            assert 'unified' in result['outputs'], "Should have unified analysis"
            assert 'temporal' in result['outputs'], "Should have temporal markers"
            
            # Check report
            assert 'report' in result, "Should have report"
            report = result['report']
            assert report['ml_analyses_complete'], "ML analyses should be complete"
            assert report['temporal_markers_generated'], "Temporal markers should be generated"
            assert report['prompts_successful'] > 0, "Some prompts should succeed"
            
            # Load and verify unified analysis
            unified_path = Path(result['outputs']['unified'])
            assert unified_path.exists(), "Unified analysis file should exist"
            
            with open(unified_path, 'r') as f:
                unified_data = json.load(f)
            
            assert unified_data['video_id'] == 'e2e_test_123'
            assert len(unified_data['timeline']['events']) > 0
            assert unified_data['temporal_markers'] is not None
            
            # Load and verify temporal markers
            temporal_path = Path(result['outputs']['temporal'])
            assert temporal_path.exists(), "Temporal markers file should exist"
            
            with open(temporal_path, 'r') as f:
                temporal_data = json.load(f)
            
            assert temporal_data['video_id'] == 'e2e_test_123'
            assert len(temporal_data['markers']) > 0
            
            print("\n✅ All verifications passed!")
            
            # Print summary
            print("\n📈 Pipeline Summary:")
            print(f"  - Video Duration: {report['duration']}s")
            print(f"  - ML Analyses: {len(ml_results)}")
            print(f"  - Timeline Events: {len(unified_data['timeline']['events'])}")
            print(f"  - Temporal Markers: {len(temporal_data['markers'])}")
            print(f"  - Prompts Successful: {report['prompts_successful']}/{report['prompts_total']}")
            print(f"  - Processing Time: {result['metrics'].get('uptime_seconds', 0):.1f}s")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Cleanup
            import shutil
            if Path("test_output").exists():
                shutil.rmtree("test_output")


async def test_legacy_mode():
    """Test legacy video ID mode."""
    print("\n\n🔄 Testing Legacy Mode")
    print("=" * 50)
    
    # Create test unified analysis
    from rumiai_v2.core.models import UnifiedAnalysis, Timeline
    
    test_dir = Path("test_legacy")
    unified_dir = test_dir / "unified"
    unified_dir.mkdir(parents=True, exist_ok=True)
    
    analysis = UnifiedAnalysis(
        video_id="legacy_test_456",
        video_metadata={
            "video_id": "legacy_test_456",
            "duration": 30.0,
            "url": "https://tiktok.com/legacy"
        },
        timeline=Timeline(duration=30.0),
        ml_results={},
        temporal_markers=None
    )
    
    analysis_path = unified_dir / "legacy_test_456.json"
    analysis.save_to_file(str(analysis_path))
    
    with patch('rumiai_v2.config.Settings') as mock_settings, \
         patch('rumiai_v2.api.ClaudeClient') as mock_claude:
        
        mock_settings.return_value.output_dir = test_dir
        mock_settings.return_value.unified_dir = unified_dir
        mock_settings.return_value.insights_dir = test_dir / "insights"
        mock_settings.return_value.temporal_dir = test_dir / "temporal"
        mock_settings.return_value._prompt_templates = {}
        mock_settings.return_value.prompt_delay = 0
        
        # Mock Claude
        mock_claude_instance = Mock()
        mock_claude_instance.send_prompt = Mock(return_value=PromptResult(
            prompt_type=PromptType.CREATIVE_DENSITY,
            success=True,
            response="Legacy mode analysis",
            tokens_used=50,
            estimated_cost=0.005,
            processing_time=0.5
        ))
        mock_claude.return_value = mock_claude_instance
        
        # Run in legacy mode
        runner = RumiAIRunner(legacy_mode=True)
        
        try:
            result = await runner.process_video_id("legacy_test_456")
            
            assert result['success'], "Legacy mode should succeed"
            assert result['video_id'] == 'legacy_test_456'
            assert result['prompts_completed'] > 0
            
            print("✅ Legacy mode test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Legacy mode failed: {e}")
            return False
            
        finally:
            # Cleanup
            import shutil
            if test_dir.exists():
                shutil.rmtree(test_dir)


async def test_error_handling():
    """Test error handling scenarios."""
    print("\n\n⚠️  Testing Error Handling")
    print("=" * 50)
    
    scenarios = [
        ("Network Error", Exception("Connection timeout")),
        ("API Error", Exception("API rate limit exceeded")),
        ("ML Processing Error", Exception("Model failed to load"))
    ]
    
    for scenario_name, error in scenarios:
        print(f"\n🔸 Testing: {scenario_name}")
        
        with patch('rumiai_v2.api.ApifyClient') as mock_apify, \
             patch('rumiai_v2.config.Settings') as mock_settings:
            
            mock_settings.return_value.output_dir = Path("test_error")
            mock_settings.return_value.unified_dir = Path("test_error/unified")
            mock_settings.return_value.insights_dir = Path("test_error/insights")
            mock_settings.return_value.temporal_dir = Path("test_error/temporal")
            
            # Make Apify fail
            mock_apify_instance = AsyncMock()
            mock_apify_instance.scrape_video = AsyncMock(side_effect=error)
            mock_apify.return_value = mock_apify_instance
            
            runner = RumiAIRunner()
            result = await runner.process_video_url("https://tiktok.com/fail")
            
            assert not result['success'], f"{scenario_name} should fail"
            assert 'error' in result, "Should have error message"
            assert result['error'] == str(error), "Error message should match"
            
            print(f"  ✅ {scenario_name} handled correctly")
    
    print("\n✅ All error scenarios handled correctly!")
    return True


async def main():
    """Run all end-to-end tests."""
    print("🎯 RumiAI v2 End-to-End Test Suite")
    print("=" * 70)
    print("This test validates the complete Pure Big Bang implementation")
    print("NO HUMAN INTERVENTION REQUIRED - Everything is automated!")
    print("=" * 70)
    
    all_passed = True
    
    # Test 1: Full pipeline
    if not await test_full_pipeline():
        all_passed = False
    
    # Test 2: Legacy mode
    if not await test_legacy_mode():
        all_passed = False
    
    # Test 3: Error handling
    if not await test_error_handling():
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ 🎉 ALL END-TO-END TESTS PASSED! 🎉")
        print("The Pure Big Bang implementation is ready for deployment!")
        print("\nNext steps:")
        print("1. Run unit tests: ./run_tests.sh")
        print("2. Deploy the new system")
        print("3. Run migration if needed: python scripts/migrate_to_v2.py")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == '__main__':
    # Suppress most logging for cleaner output
    logging.getLogger().setLevel(logging.ERROR)
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)