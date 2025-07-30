#!/usr/bin/env python3
"""
Test suite for Timestamp model.

CRITICAL: Tests all edge cases and formats that caused failures.
"""
import unittest
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rumiai_v2.core.models import Timestamp


class TestTimestamp(unittest.TestCase):
    """Test timestamp parsing and operations."""
    
    def test_valid_formats(self):
        """Test all valid timestamp formats."""
        test_cases = [
            # Format: (input, expected_seconds)
            ("00:00:12", 12.0),
            ("01:23:45", 5025.0),
            ("2:34", 154.0),
            ("0:05", 5.0),
            ("123", 123.0),
            ("45.67", 45.67),
            (123, 123.0),
            (45.67, 45.67),
            ("1:00:00", 3600.0),
            ("00:00:00", 0.0),
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input=input_val):
                ts = Timestamp.from_value(input_val)
                self.assertIsNotNone(ts)
                self.assertAlmostEqual(ts.seconds, expected, places=2)
    
    def test_invalid_formats(self):
        """Test invalid formats return None (not raise)."""
        test_cases = [
            None,
            "",
            "invalid",
            "12:60:00",  # Invalid minutes
            "12:00:60",  # Invalid seconds
            "-5",
            "1:2:3:4",  # Too many parts
            {},
            [],
            "NaN",
            float('inf'),
            float('-inf'),
        ]
        
        for input_val in test_cases:
            with self.subTest(input=input_val):
                ts = Timestamp.from_value(input_val)
                self.assertIsNone(ts)
    
    def test_comparisons(self):
        """Test timestamp comparisons."""
        ts1 = Timestamp(10.0)
        ts2 = Timestamp(20.0)
        ts3 = Timestamp(10.0)
        
        # Equality
        self.assertEqual(ts1, ts3)
        self.assertNotEqual(ts1, ts2)
        
        # Ordering
        self.assertLess(ts1, ts2)
        self.assertLessEqual(ts1, ts2)
        self.assertLessEqual(ts1, ts3)
        self.assertGreater(ts2, ts1)
        self.assertGreaterEqual(ts2, ts1)
        self.assertGreaterEqual(ts1, ts3)
    
    def test_none_comparisons(self):
        """Test comparisons with None values."""
        ts = Timestamp(10.0)
        none_ts = None
        
        # These should not raise
        self.assertNotEqual(ts, none_ts)
        self.assertIsNone(Timestamp.from_value(None))
    
    def test_string_representation(self):
        """Test string formatting."""
        test_cases = [
            (0.0, "00:00:00"),
            (5.0, "00:00:05"),
            (65.0, "00:01:05"),
            (3665.5, "01:01:05"),
            (7200.0, "02:00:00"),
        ]
        
        for seconds, expected in test_cases:
            with self.subTest(seconds=seconds):
                ts = Timestamp(seconds)
                self.assertEqual(str(ts), expected)
    
    def test_serialization(self):
        """Test JSON serialization."""
        ts = Timestamp(123.45)
        data = ts.to_dict()
        
        self.assertEqual(data['seconds'], 123.45)
        self.assertEqual(data['formatted'], "00:02:03")
        
        # Test round-trip
        ts2 = Timestamp.from_dict(data)
        self.assertEqual(ts, ts2)
    
    def test_edge_cases(self):
        """Test edge cases that caused real failures."""
        # Case 1: Parsing from visual_overlay prompt
        ts = Timestamp.from_value("0:12")
        self.assertIsNotNone(ts)
        self.assertEqual(ts.seconds, 12.0)
        
        # Case 2: Comparison with None (from parse_timestamp_to_seconds)
        ts = Timestamp.from_value("invalid")
        self.assertIsNone(ts)
        
        # Case 3: Very large timestamps
        ts = Timestamp.from_value("99:59:59")
        self.assertIsNotNone(ts)
        self.assertEqual(ts.seconds, 359999.0)


if __name__ == '__main__':
    unittest.main()