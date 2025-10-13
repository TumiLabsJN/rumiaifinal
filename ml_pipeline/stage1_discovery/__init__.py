"""
Stage 1: Video Discovery & Selection

This module handles the discovery and selection of TikTok videos for ML analysis.
It implements the complete Stage 1 pipeline as specified in VideoDiscoveryCHILDTI.md.

Pipeline Stages:
- Stage 1.1: Apify Scraping - Scrape 800 videos from TikTok
- Stage 1.2: Date Filtering - Filter by publication date
- Stage 1.3: Winner Analysis - Identify top 3 winning buckets
- Stage 1.4: Video Selection - Select videos per bucket (contrastive/top)
- Stage 1.5: Interactive Confirmation - Display summary and get user approval

Output:
- selected_videos.json per winning bucket
- winner_analysis.json for analysis run
"""

from .video_discovery import VideoDiscovery
from .apify_scraper import ApifyScraper
from .date_filter import DateFilter
from .winner_analyzer import WinnerAnalyzer
from .video_selector import VideoSelector
from .confirmation import InteractiveConfirmation

__all__ = [
    'VideoDiscovery',
    'ApifyScraper',
    'DateFilter',
    'WinnerAnalyzer',
    'VideoSelector',
    'InteractiveConfirmation',
]

__version__ = '1.0.0'
