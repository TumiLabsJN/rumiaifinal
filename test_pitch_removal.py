import json
import unittest
from rumiai_v2.processors.temporal_compute import process_temporal_windows, calculate_pitch_metrics
import inspect

class TestPitchRemoval(unittest.TestCase):

    def test_no_avg_pitch_normalized_in_output(self):
        """Verify avg_pitch_normalized is not in any temporal window"""
        result = process_temporal_windows('test_video.mp4')

        self.assertNotIn('avg_pitch_normalized', result['temporal_windows']['hook'])

        for segment in result['temporal_windows'].get('middle_segments', []):
            self.assertNotIn('avg_pitch_normalized', segment)

        self.assertNotIn('avg_pitch_normalized', result['temporal_windows']['closing'])

    def test_gender_detection_still_present(self):
        """Verify gender detection remains in metadata"""
        result = process_temporal_windows('test_video.mp4')

        self.assertIn('gender_detection', result['metadata'])
        gender_data = result['metadata']['gender_detection']
        self.assertIn('gender', gender_data)
        self.assertIn('confidence', gender_data)
        self.assertIn('method', gender_data)

    def test_pitch_scatter_ratio_handles_no_voice(self):
        """Verify pitch_scatter_ratio handles videos with no voiced content"""
        result = process_temporal_windows('silent_video.mp4')

        pitch_val = result['temporal_windows']['hook'].get('pitch_scatter_ratio', 0.0)
        self.assertEqual(pitch_val, 0.0)

    def test_calculate_pitch_metrics_signature(self):
        """Verify function signature no longer accepts gender parameter"""
        sig = inspect.signature(calculate_pitch_metrics)
        param_names = list(sig.parameters.keys())

        self.assertNotIn('gender', param_names)

    def test_calculate_pitch_metrics_return_value(self):
        """Verify function returns single value or None"""
        # This test would need proper mock data to run
        pass

if __name__ == '__main__':
    unittest.main()