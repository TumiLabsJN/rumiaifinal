"""
Claude API client for RumiAI v2.

CRITICAL: This client handles all Claude API interactions with retry logic,
connection pooling fixes, and cost tracking.
"""
import requests
import time
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict

from ..core.exceptions import APIError
from ..core.models import PromptResult

logger = logging.getLogger(__name__)


class APIMetrics:
    """Track API usage and costs."""
    
    def __init__(self):
        self.api_calls = defaultdict(int)
        self.api_costs = defaultdict(float)
        self.api_errors = defaultdict(list)
        self.api_tokens = defaultdict(int)
    
    def track_call(self, service: str, success: bool, cost: float = 0, tokens: int = 0):
        """Track an API call."""
        self.api_calls[service] += 1
        self.api_costs[service] += cost
        self.api_tokens[service] += tokens
        if not success:
            self.api_errors[service].append(datetime.utcnow().isoformat())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        return {
            'calls': dict(self.api_calls),
            'costs': dict(self.api_costs),
            'tokens': dict(self.api_tokens),
            'error_count': {k: len(v) for k, v in self.api_errors.items()},
            'recent_errors': {k: v[-5:] for k, v in self.api_errors.items()}  # Last 5 errors
        }


class ClaudeClient:
    """
    Claude API client with retry logic and error handling.
    
    CRITICAL: Uses fresh sessions to avoid connection pool issues.
    """
    
    # Pricing per million tokens by model
    MODEL_PRICING = {
        "claude-3-haiku-20240307": {
            "input": 0.25,
            "output": 1.25
        },
        "claude-3-sonnet-20241022": {
            "input": 3.00,
            "output": 15.00
        },
        "claude-3-5-sonnet-20241022": {
            "input": 3.00,
            "output": 15.00
        },
        "claude-3-5-sonnet-20240620": {
            "input": 3.00,
            "output": 15.00
        }
    }
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.max_retries = 3
        self.base_delay = 5  # seconds
        self.metrics = APIMetrics()
        
        # Set default pricing based on initial model
        if model in self.MODEL_PRICING:
            pricing = self.MODEL_PRICING[model]
            self.PRICE_PER_MILLION_INPUT_TOKENS = pricing["input"]
            self.PRICE_PER_MILLION_OUTPUT_TOKENS = pricing["output"]
        else:
            # Default to Haiku pricing if model unknown
            self.PRICE_PER_MILLION_INPUT_TOKENS = 0.25
            self.PRICE_PER_MILLION_OUTPUT_TOKENS = 1.25
    
    def send_prompt(self, prompt: str, context_data: Dict[str, Any], 
                   timeout: int = 60, max_tokens: int = 1500) -> PromptResult:
        """
        Send prompt to Claude with context data.
        
        CRITICAL: Creates fresh session for each request to avoid stale connections.
        """
        prompt_type = context_data.get('prompt_type', 'unknown')
        video_id = context_data.get('video_id', 'unknown')
        
        # Allow model override in context_data
        model_to_use = context_data.get('model', self.model)
        
        # Update pricing if using different model
        if model_to_use != self.model and model_to_use in self.MODEL_PRICING:
            pricing = self.MODEL_PRICING[model_to_use]
            input_price = pricing["input"]
            output_price = pricing["output"]
        else:
            input_price = self.PRICE_PER_MILLION_INPUT_TOKENS
            output_price = self.PRICE_PER_MILLION_OUTPUT_TOKENS
        
        # Import PromptType enum for creating result
        from ..core.models.prompt import PromptType
        
        # Convert string prompt_type back to enum if needed
        if isinstance(prompt_type, str):
            # Find the matching enum value
            prompt_type_enum = None
            for pt in PromptType:
                if pt.value == prompt_type:
                    prompt_type_enum = pt
                    break
            if not prompt_type_enum:
                # Default fallback - just use the string
                prompt_type_enum = prompt_type
        else:
            prompt_type_enum = prompt_type
        
        # Build complete prompt
        full_prompt = self._build_prompt_content(prompt, context_data)
        
        # Prepare request
        messages = [{
            "role": "user",
            "content": full_prompt
        }]
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        request_data = {
            "model": model_to_use,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        # Log request size
        request_json = json.dumps(request_data)
        request_size = len(request_json)
        logger.info(f"Sending {prompt_type} prompt for {video_id} using {model_to_use}: {request_size} bytes")
        
        # CRITICAL: Track retry attempts for monitoring
        retry_metadata = {
            'video_id': video_id,
            'prompt_type': prompt_type,
            'attempts': []
        }
        
        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            attempt_start = time.time()
            
            try:
                # CRITICAL: Create fresh session for each request
                with requests.Session() as session:
                    response = session.post(
                        self.api_url,
                        headers=headers,
                        json=request_data,
                        timeout=timeout
                    )
                
                attempt_duration = time.time() - attempt_start
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract response and usage
                    response_text = ''
                    if 'content' in result and result['content']:
                        response_text = result['content'][0].get('text', '')
                    
                    usage = result.get('usage', {})
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    total_tokens = input_tokens + output_tokens
                    
                    # Calculate cost
                    input_cost = (input_tokens / 1_000_000) * input_price
                    output_cost = (output_tokens / 1_000_000) * output_price
                    total_cost = input_cost + output_cost
                    
                    # Track successful call
                    self.metrics.track_call(
                        prompt_type,
                        success=True,
                        cost=total_cost,
                        tokens=total_tokens
                    )
                    
                    retry_metadata['attempts'].append({
                        'attempt': attempt + 1,
                        'status': 200,
                        'duration': attempt_duration,
                        'tokens': total_tokens,
                        'cost': total_cost
                    })
                    
                    logger.info(
                        f"Claude API success for {prompt_type}: "
                        f"{total_tokens} tokens, ${total_cost:.4f}"
                    )
                    
                    return PromptResult(
                        prompt_type=prompt_type_enum,
                        success=True,
                        response=response_text,
                        processing_time=attempt_duration,
                        tokens_used=total_tokens,
                        estimated_cost=total_cost,
                        retry_attempts=attempt
                    )
                
                elif response.status_code == 429:  # Rate limit
                    retry_metadata['attempts'].append({
                        'attempt': attempt + 1,
                        'status': 429,
                        'duration': attempt_duration,
                        'error': 'Rate limited'
                    })
                    
                    if attempt < self.max_retries - 1:
                        delay = self.base_delay * (2 ** attempt)
                        logger.warning(f"Rate limited, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    else:
                        error_msg = f"Rate limited after {self.max_retries} attempts"
                        self.metrics.track_call(prompt_type, success=False)
                        raise APIError('Claude', 429, error_msg, video_id)
                
                else:
                    # Other HTTP errors
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    retry_metadata['attempts'].append({
                        'attempt': attempt + 1,
                        'status': response.status_code,
                        'duration': attempt_duration,
                        'error': error_msg
                    })
                    
                    self.metrics.track_call(prompt_type, success=False)
                    raise APIError('Claude', response.status_code, error_msg, video_id)
                    
            except requests.exceptions.ConnectionError as e:
                # Handle connection errors (including RemoteDisconnected)
                error_msg = f"Connection error: {str(e)}"
                retry_metadata['attempts'].append({
                    'attempt': attempt + 1,
                    'status': 0,
                    'duration': time.time() - attempt_start,
                    'error': error_msg
                })
                
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"🔌 Connection error: {error_msg}")
                    logger.warning(f"⏳ Retrying in {delay}s (attempt {attempt + 1}/{self.max_retries})...")
                    time.sleep(delay)
                    continue
                else:
                    self.metrics.track_call(prompt_type, success=False)
                    return PromptResult(
                        prompt_type=prompt_type,
                        success=False,
                        error=f"Connection failed after {self.max_retries} attempts: {error_msg}",
                        retry_attempts=self.max_retries
                    )
            
            except requests.exceptions.Timeout:
                error_msg = f"Request timed out after {timeout}s"
                retry_metadata['attempts'].append({
                    'attempt': attempt + 1,
                    'status': 0,
                    'duration': timeout,
                    'error': error_msg
                })
                
                self.metrics.track_call(prompt_type, success=False)
                return PromptResult(
                    prompt_type=prompt_type_enum,
                    success=False,
                    error=error_msg,
                    retry_attempts=attempt
                )
            
            except Exception as e:
                error_msg = f"Unexpected error: {type(e).__name__}: {str(e)}"
                retry_metadata['attempts'].append({
                    'attempt': attempt + 1,
                    'status': 0,
                    'duration': time.time() - attempt_start,
                    'error': error_msg
                })
                
                logger.error(f"Unexpected error in Claude API: {error_msg}", exc_info=True)
                self.metrics.track_call(prompt_type, success=False)
                
                return PromptResult(
                    prompt_type=prompt_type_enum,
                    success=False,
                    error=error_msg,
                    retry_attempts=attempt
                )
        
        # If we get here, all retries failed
        self.metrics.track_call(prompt_type, success=False)
        return PromptResult(
            prompt_type=prompt_type_enum,
            success=False,
            error=f"Failed after {self.max_retries} attempts",
            retry_attempts=self.max_retries
        )
    
    def _build_prompt_content(self, prompt: str, context_data: Dict[str, Any]) -> str:
        """Build complete prompt with context."""
        # Remove internal fields from context
        context_copy = context_data.copy()
        context_copy.pop('video_id', None)
        context_copy.pop('prompt_type', None)
        
        # Format context sections
        context_sections = []
        
        # Add each context section
        for key, value in context_copy.items():
            if value:  # Skip empty values
                if isinstance(value, dict):
                    section_text = json.dumps(value, indent=2)
                elif isinstance(value, list):
                    section_text = json.dumps(value, indent=2)
                else:
                    section_text = str(value)
                
                context_sections.append(f"{key.upper()}:\n{section_text}")
        
        # Combine prompt and context
        if context_sections:
            context_text = "\n\n".join(context_sections)
            return f"{prompt}\n\nCONTEXT DATA:\n{context_text}"
        else:
            return prompt
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get API usage metrics."""
        return self.metrics.get_summary()
    
    def estimate_prompt_cost(self, prompt_text: str, expected_output_tokens: int = 500) -> float:
        """Estimate cost for a prompt."""
        # Rough token estimation: ~4 characters per token
        input_tokens = len(prompt_text) // 4
        
        input_cost = (input_tokens / 1_000_000) * self.PRICE_PER_MILLION_INPUT_TOKENS
        output_cost = (expected_output_tokens / 1_000_000) * self.PRICE_PER_MILLION_OUTPUT_TOKENS
        
        return input_cost + output_cost