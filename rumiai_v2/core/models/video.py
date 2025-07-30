"""
Video metadata model for RumiAI v2.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class VideoMetadata:
    """
    Video metadata from TikTok scraping.
    
    This maintains compatibility with Apify output format.
    """
    video_id: str
    url: str
    username: str
    description: str
    duration: int  # seconds
    views: int
    likes: int
    comments: int
    shares: int
    saves: int
    create_time: datetime
    download_url: str
    cover_url: str
    hashtags: List[Dict[str, Any]] = field(default_factory=list)
    music: Dict[str, Any] = field(default_factory=dict)
    author: Dict[str, Any] = field(default_factory=dict)
    engagement_rate: float = 0.0
    
    @classmethod
    def from_apify_data(cls, data: Dict[str, Any]) -> 'VideoMetadata':
        """Create from Apify scraper output."""
        # Parse create time
        create_time_str = data.get('createTimeISO', data.get('createTime', ''))
        try:
            create_time = datetime.fromisoformat(create_time_str.replace('Z', '+00:00'))
        except:
            create_time = datetime.now()
        
        # Get download URL - try multiple possible field names
        download_url = ''
        # First try videoUrl (as shown in JS implementation)
        if data.get('videoUrl'):
            download_url = data.get('videoUrl')
        # Then try downloadAddr
        elif data.get('downloadAddr'):
            download_url = data.get('downloadAddr')
        # Then try mediaUrls array
        elif data.get('mediaUrls') and len(data.get('mediaUrls', [])) > 0:
            download_url = data.get('mediaUrls')[0]
        # Finally fallback to downloadUrl
        else:
            download_url = data.get('downloadUrl', '')
        
        # Get username from authorMeta
        author_meta = data.get('authorMeta', {})
        username = author_meta.get('name', '')
        
        # Get video stats
        video_meta = data.get('videoMeta', {})
        
        return cls(
            video_id=data.get('id', ''),
            url=data.get('webVideoUrl', data.get('url', '')),
            username=username,
            description=data.get('text', data.get('description', '')),
            duration=video_meta.get('duration', data.get('duration', 0)),
            views=data.get('playCount', data.get('views', 0)),
            likes=data.get('diggCount', data.get('likes', 0)),
            comments=data.get('commentCount', data.get('comments', 0)),
            shares=data.get('shareCount', data.get('shares', 0)),
            saves=data.get('collectCount', data.get('saves', 0)),
            create_time=create_time,
            download_url=download_url,
            cover_url=video_meta.get('cover', data.get('coverUrl', '')),
            hashtags=data.get('hashtags', []),
            music=data.get('musicMeta', data.get('music', {})),
            author=author_meta,
            engagement_rate=data.get('engagementRate', 0.0)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching expected format."""
        return {
            'id': self.video_id,
            'url': self.url,
            'description': self.description,
            'duration': self.duration,
            'views': self.views,
            'likes': self.likes,
            'comments': self.comments,
            'shares': self.shares,
            'saves': self.saves,
            'createTime': self.create_time.isoformat(),
            'downloadUrl': self.download_url,
            'coverUrl': self.cover_url,
            'hashtags': self.hashtags,
            'music': self.music,
            'author': self.author,
            'engagementRate': self.engagement_rate
        }
    
    @property
    def display_name(self) -> str:
        """Get display name for video."""
        return f"@{self.username}_{self.video_id}"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return f"VideoMetadata(id={self.video_id}, user=@{self.username}, duration={self.duration}s)"