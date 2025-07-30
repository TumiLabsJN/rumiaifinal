#!/usr/bin/env python3
"""
Test suite for Claude API Client.

CRITICAL: Tests connection pool fix for RemoteDisconnected errors.
"""
import unittest
from unittest.mock import patch, Mock, MagicMock
import sys
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from rumiai_v2.api import ClaudeClient
from rumiai_v2.core.exceptions import APIError


class TestClaudeClient(unittest.TestCase):
    """Test Claude API client functionality."""
    
    def setUp(self):
        """Set up test client."""
        self.client = ClaudeClient(
            api_key="test-key",
            model="claude-3-opus-20240229"
        )
    
    @patch('requests.Session')
    def test_fresh_session_per_request(self, mock_session_class):
        """Test that each request uses a fresh session."""
        # Create mock session instance
        mock_session = MagicMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'content': [{'text': 'Test response'}],
            'usage': {'input_tokens': 10, 'output_tokens': 20}
        }
        mock_session.post.return_value = mock_response
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = None
        
        # Configure class to return our mock
        mock_session_class.return_value = mock_session
        
        # Make two requests
        result1 = self.client.send_prompt("Test 1", {})
        result2 = self.client.send_prompt("Test 2", {})
        
        # Verify fresh session created each time
        self.assertEqual(mock_session_class.call_count, 2)
        
        # Verify results
        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
    
    @patch('requests.Session')
    def test_connection_error_handling(self, mock_session_class):
        """Test handling of connection errors."""
        # Create session that raises RemoteDisconnected
        mock_session = MagicMock()
        mock_session.post.side_effect = requests.exceptions.ConnectionError(
            "RemoteDisconnected('Remote end closed connection without response')"
        )
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = None
        mock_session_class.return_value = mock_session
        
        # Should not raise, returns error result
        result = self.client.send_prompt("Test", {})
        
        self.assertFalse(result.success)
        self.assertIn("API request failed", result.error)
    
    @patch('requests.Session')
    def test_retry_logic(self, mock_session_class):
        """Test retry on transient failures."""
        # Create session that fails then succeeds
        mock_session = MagicMock()
        mock_response_fail = Mock()
        mock_response_fail.status_code = 503
        mock_response_fail.text = "Service unavailable"
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            'content': [{'text': 'Success after retry'}],
            'usage': {'input_tokens': 10, 'output_tokens': 20}
        }
        
        # First call fails, second succeeds
        mock_session.post.side_effect = [
            mock_response_fail,
            mock_response_success
        ]
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = None
        mock_session_class.return_value = mock_session
        
        result = self.client.send_prompt("Test", {})
        
        # Should succeed after retry
        self.assertTrue(result.success)
        self.assertEqual(result.response, "Success after retry")
    
    @patch('requests.Session')
    def test_timeout_handling(self, mock_session_class):
        """Test request timeout handling."""
        mock_session = MagicMock()
        mock_session.post.side_effect = requests.exceptions.Timeout("Request timed out")
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = None
        mock_session_class.return_value = mock_session
        
        result = self.client.send_prompt("Test", {}, timeout=5)
        
        self.assertFalse(result.success)
        self.assertIn("timed out", result.error)
    
    @patch('requests.Session')
    def test_rate_limit_handling(self, mock_session_class):
        """Test rate limit error handling."""
        mock_session = MagicMock()
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            'error': {'message': 'Rate limit exceeded'}
        }
        mock_response.headers = {'retry-after': '60'}
        mock_session.post.return_value = mock_response
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = None
        mock_session_class.return_value = mock_session
        
        result = self.client.send_prompt("Test", {})
        
        self.assertFalse(result.success)
        self.assertIn("Rate limit", result.error)
    
    @patch('requests.Session')
    def test_successful_request(self, mock_session_class):
        """Test successful API request."""
        mock_session = MagicMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'content': [{'text': 'Test successful response'}],
            'usage': {
                'input_tokens': 100,
                'output_tokens': 200
            }
        }
        mock_session.post.return_value = mock_response
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = None
        mock_session_class.return_value = mock_session
        
        result = self.client.send_prompt(
            "Test prompt",
            {"video_id": "test123"},
            timeout=30
        )
        
        # Verify request structure
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        
        # Check URL
        self.assertEqual(call_args[0][0], "https://api.anthropic.com/v1/messages")
        
        # Check headers
        headers = call_args[1]['headers']
        self.assertEqual(headers['x-api-key'], 'test-key')
        self.assertEqual(headers['anthropic-version'], '2023-06-01')
        
        # Check timeout
        self.assertEqual(call_args[1]['timeout'], 30)
        
        # Verify result
        self.assertTrue(result.success)
        self.assertEqual(result.response, "Test successful response")
        self.assertEqual(result.tokens_used, 300)
        self.assertGreater(result.estimated_cost, 0)
    
    def test_prompt_building(self):
        """Test prompt message building."""
        # This tests internal method directly
        messages = self.client._build_messages(
            "Test prompt text",
            {"video_id": "123", "type": "test"}
        )
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]['role'], 'user')
        self.assertIn("Test prompt text", messages[0]['content'])
        self.assertIn("Video ID: 123", messages[0]['content'])
    
    @patch('requests.Session')  
    def test_cost_calculation(self, mock_session_class):
        """Test API cost calculation."""
        mock_session = MagicMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'content': [{'text': 'Response'}],
            'usage': {
                'input_tokens': 1000,
                'output_tokens': 2000
            }
        }
        mock_session.post.return_value = mock_response
        mock_session.__enter__.return_value = mock_session
        mock_session.__exit__.return_value = None
        mock_session_class.return_value = mock_session
        
        result = self.client.send_prompt("Test", {})
        
        # Verify cost calculation (based on Claude 3 Opus pricing)
        # Input: $15/M tokens, Output: $75/M tokens
        expected_cost = (1000 * 15 / 1_000_000) + (2000 * 75 / 1_000_000)
        self.assertAlmostEqual(result.estimated_cost, expected_cost, places=4)


if __name__ == '__main__':
    unittest.main()