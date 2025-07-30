"""
Apify API client for RumiAI v2.

This module handles TikTok video scraping via Apify.
"""
import aiohttp
import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import time

from ..core.exceptions import APIError
from ..core.models import VideoMetadata

logger = logging.getLogger(__name__)


class ApifyClient:
    """
    Apify API client for TikTok scraping.
    
    Handles video metadata scraping and video downloads.
    """
    
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.actor_id = "GdWCkxBtKWOsKjdch"  # TikTok scraper actor
        self.base_url = "https://api.apify.com/v2"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}"
        }
    
    async def scrape_video(self, video_url: str) -> VideoMetadata:
        """
        Scrape single video metadata from TikTok.
        
        Returns VideoMetadata object with all video information.
        """
        logger.info(f"Scraping video: {video_url}")
        
        # Prepare actor input
        actor_input = {
            "postURLs": [video_url],
            "resultsPerPage": 1,
            "shouldDownloadVideos": True,
            "shouldDownloadCovers": True,
            "shouldDownloadSubtitles": True,
            "proxyConfiguration": {
                "useApifyProxy": True
            }
        }
        
        try:
            # Start actor run
            run_info = await self._start_actor_run(actor_input)
            # Debug: log the response structure
            logger.info(f"Apify run response: {run_info}")
            # Handle both 'id' and 'data' response formats
            if 'data' in run_info and 'id' in run_info['data']:
                run_id = run_info['data']['id']
            elif 'id' in run_info:
                run_id = run_info['id']
            else:
                raise KeyError(f"No run ID found in response: {run_info}")
            logger.info(f"Apify run started: {run_id}")
            
            # Wait for completion
            run_result = await self._wait_for_run(run_id)
            
            if run_result['status'] != 'SUCCEEDED':
                raise APIError(
                    'Apify',
                    0,
                    f"Actor run failed with status: {run_result['status']}",
                    video_url
                )
            
            # Get dataset items
            dataset_id = run_result['defaultDatasetId']
            items = await self._get_dataset_items(dataset_id)
            
            if not items:
                raise APIError('Apify', 0, "No video data returned", video_url)
            
            # Convert to VideoMetadata
            video_data = items[0]
            logger.info(f"Apify data keys: {list(video_data.keys())}")
            logger.info(f"Sample data: {json.dumps(video_data, indent=2)[:500]}...")
            metadata = VideoMetadata.from_apify_data(video_data)
            
            logger.info(f"Successfully scraped video {metadata.video_id}")
            return metadata
            
        except Exception as e:
            if isinstance(e, APIError):
                raise
            else:
                raise APIError('Apify', 0, f"Scraping failed: {str(e)}", video_url)
    
    async def scrape_multiple_videos(self, video_urls: List[str]) -> List[VideoMetadata]:
        """Scrape multiple videos in one run."""
        logger.info(f"Scraping {len(video_urls)} videos")
        
        # Prepare actor input
        actor_input = {
            "postURLs": video_urls,
            "resultsPerPage": len(video_urls),
            "shouldDownloadVideos": True,
            "shouldDownloadCovers": True,
            "shouldDownloadSubtitles": True,
            "proxyConfiguration": {
                "useApifyProxy": True
            }
        }
        
        try:
            # Start actor run
            run_info = await self._start_actor_run(actor_input)
            run_id = run_info['id']
            
            # Wait for completion
            run_result = await self._wait_for_run(run_id)
            
            if run_result['status'] != 'SUCCEEDED':
                raise APIError(
                    'Apify',
                    0,
                    f"Actor run failed with status: {run_result['status']}"
                )
            
            # Get dataset items
            dataset_id = run_result['defaultDatasetId']
            items = await self._get_dataset_items(dataset_id)
            
            # Convert to VideoMetadata objects
            videos = []
            for item in items:
                try:
                    metadata = VideoMetadata.from_apify_data(item)
                    videos.append(metadata)
                except Exception as e:
                    logger.error(f"Failed to parse video data: {e}")
            
            logger.info(f"Successfully scraped {len(videos)} videos")
            return videos
            
        except Exception as e:
            if isinstance(e, APIError):
                raise
            else:
                raise APIError('Apify', 0, f"Batch scraping failed: {str(e)}")
    
    async def download_video(self, download_url: str, video_id: str, 
                           output_dir: Path = Path("temp")) -> Path:
        """
        Download video file from Apify storage.
        
        Returns path to downloaded video file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_id}.mp4"
        
        # Check if already downloaded
        if output_path.exists():
            logger.info(f"Video already downloaded: {output_path}")
            return output_path
        
        logger.info(f"Downloading video {video_id} from {download_url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(download_url, headers=self.headers) as response:
                    if response.status != 200:
                        raise APIError(
                            'Apify',
                            response.status,
                            f"Failed to download video: {response.status}",
                            video_id
                        )
                    
                    # Download in chunks
                    with open(output_path, 'wb') as f:
                        chunk_size = 8192
                        downloaded = 0
                        
                        async for chunk in response.content.iter_chunked(chunk_size):
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Log progress every 10MB
                            if downloaded % (10 * 1024 * 1024) == 0:
                                logger.info(f"Downloaded {downloaded / (1024*1024):.1f}MB")
            
            logger.info(f"Successfully downloaded video to {output_path}")
            return output_path
            
        except Exception as e:
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            
            if isinstance(e, APIError):
                raise
            else:
                raise APIError('Apify', 0, f"Download failed: {str(e)}", video_id)
    
    async def _start_actor_run(self, actor_input: Dict[str, Any]) -> Dict[str, Any]:
        """Start an Apify actor run."""
        url = f"{self.base_url}/acts/{self.actor_id}/runs"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=self.headers,
                json=actor_input
            ) as response:
                text = await response.text()
                logger.info(f"Apify API response status: {response.status}")
                
                if response.status not in [200, 201]:
                    raise APIError('Apify', response.status, f"Failed to start actor: {text}")
                
                try:
                    data = json.loads(text)
                    return data
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON response: {text[:500]}")
                    raise APIError('Apify', response.status, f"Invalid JSON response")
    
    async def _wait_for_run(self, run_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for actor run to complete."""
        url = f"{self.base_url}/actor-runs/{run_id}"
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while True:
                # Check timeout
                if time.time() - start_time > timeout:
                    raise APIError('Apify', 0, f"Run {run_id} timed out after {timeout}s")
                
                # Get run status
                async with session.get(url, headers=self.headers) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise APIError('Apify', response.status, f"Failed to get run status: {text}")
                    
                    run_data = await response.json()
                    status = run_data['data']['status']
                    
                    logger.info(f"Run {run_id} status: {status}")
                    
                    if status in ['SUCCEEDED', 'FAILED', 'ABORTED']:
                        return run_data['data']
                
                # Wait before next check
                await asyncio.sleep(3)
    
    async def _get_dataset_items(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get items from Apify dataset."""
        url = f"{self.base_url}/datasets/{dataset_id}/items"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    text = await response.text()
                    raise APIError('Apify', response.status, f"Failed to get dataset items: {text}")
                
                return await response.json()
    
    async def check_quota(self) -> Dict[str, Any]:
        """Check Apify account quota and usage."""
        url = f"{self.base_url}/users/me"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    text = await response.text()
                    raise APIError('Apify', response.status, f"Failed to get user info: {text}")
                
                user_data = await response.json()
                
                # Extract quota information
                return {
                    'monthly_usage': user_data.get('monthlyUsage', 0),
                    'monthly_limit': user_data.get('monthlyLimit', 0),
                    'remaining': user_data.get('monthlyLimit', 0) - user_data.get('monthlyUsage', 0),
                    'usage_percentage': (
                        user_data.get('monthlyUsage', 0) / user_data.get('monthlyLimit', 1) * 100
                    )
                }