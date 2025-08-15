"""
System constants for RumiAI v2.
"""

# Version
SYSTEM_VERSION = "2.0.0"

# ML Model versions
ML_MODELS = {
    'yolo': {
        'name': 'YOLOv8',
        'version': 'v8',
        'supported_classes': [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    },
    'whisper': {
        'name': 'OpenAI Whisper',
        'version': 'base',
        'model_size': 'base'
    },
    'mediapipe': {
        'name': 'Google MediaPipe',
        'version': '0.10',
        'solutions': ['pose', 'face_mesh', 'hands']
    },
    'ocr': {
        'name': 'Tesseract OCR',
        'version': '5.0'
    },
    'scene_detection': {
        'name': 'PySceneDetect',
        'version': '0.6'
    }
}

# Prompt types
PROMPT_TYPES = [
    'creative_density',
    'emotional_journey',
    'speech_analysis',
    'visual_overlay_analysis',
    'metadata_analysis',
    'person_framing',
    'scene_pacing'
]

# File naming patterns
FILE_PATTERNS = {
    'yolo': '{video_id}_yolo_detections.json',
    'whisper': '{video_id}_whisper.json',
    'mediapipe': '{video_id}_human_analysis.json',
    'ocr': '{video_id}_creative_analysis.json',
    'scene': '{video_id}_scenes.json',
    'temporal': '{video_id}_temporal_markers.json',
    'unified': '{video_id}.json',
    'analysis_result': '{analysis_type}_result_{timestamp}.json',
    'analysis_complete': '{analysis_type}_complete_{timestamp}.json'
}

# Timeline entry types
TIMELINE_ENTRY_TYPES = [
    'object',      # Object detection
    'speech',      # Speech segment
    'text',        # Text overlay
    'sticker',     # Sticker/emoji
    'scene_change', # Scene transition
    'scene',       # Scene segment
    'pose',        # Human pose
    'face',        # Face detection
    'gesture',     # Hand gesture
    'action',      # Action recognition
]

# Emotion categories
EMOTION_CATEGORIES = [
    'neutral',
    'positive',
    'negative',
    'excited',
    'calm',
    'urgent',
    'warning',
    'humorous',
    'serious',
    'inspirational'
]

# CTA keywords
CTA_KEYWORDS = [
    'follow', 'like', 'subscribe', 'comment', 'share',
    'link', 'bio', 'check out', 'click', 'tap', 'swipe',
    'buy', 'shop', 'order', 'get', 'join', 'sign up',
    'download', 'save', 'watch', 'learn more', 'dm',
    'message', 'visit', 'discover', 'explore'
]

# Video quality thresholds
VIDEO_QUALITY = {
    'min_duration': 3,      # Minimum video duration in seconds
    'max_duration': 600,    # Maximum video duration in seconds (10 min)
    'min_resolution': 480,  # Minimum height in pixels
    'max_file_size': 500 * 1024 * 1024,  # 500MB
}

# Processing limits
PROCESSING_LIMITS = {
    'max_concurrent_videos': 5,
    'max_frames_per_video': 3000,
    'max_timeline_entries': 10000,
    'max_prompt_context_size': 100 * 1024,  # 100KB
    'max_api_retries': 3,
    'api_timeout': 300  # 5 minutes
}

# Cost tracking (legacy - kept for reference)
COST_ESTIMATES = {
    # 'claude_haiku_input': 0.25 / 1_000_000,   # $ per token (removed - Python-only)
    # 'claude_haiku_output': 1.25 / 1_000_000,  # $ per token (removed - Python-only)
    'apify_video_scrape': 0.001,              # $ per video
    'storage_gb_month': 0.02                  # $ per GB per month
}