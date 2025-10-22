"""
Stage 4: Feature Transformation - Configuration Constants

Source: FeatureTransformationTI.md Section 9.2 (Lines 1647-1730)

Single source of truth for all Stage 4 transformation parameters.

Last Updated: 2025-01-28
"""

# ===== File Paths (relative to bucket directory) =====
AGGREGATED_CSV_PATH = "ml_analysis/aggregated_features.csv"
OUTPUT_DIR = "ml_analysis/"
CHECKPOINT_DIR = "checkpoints/"
CHECKPOINT_FILE = "stage_4_checkpoint.json"

# ===== Base Features (21 total) =====
BASE_FEATURES = [
    # Visual features
    'average_face_size', 'overlay_unique_count', 'scene_count', 'shortest_scene',
    'longest_scene', 'scene_duration_variance', 'object_count', 'person_count',
    'eye_contact_rate', 'gaze_variance', 'gesture_count',
    # Audio features
    'speech_coverage', 'word_count', 'energy_level', 'energy_variance',
    'energy_max', 'pitch_scatter_ratio',
    # Emotion features
    'dominant_emotion_id', 'emotional_valence', 'emotion_consistency',
    # Text features
    'has_captions'
]

# ===== Transformation Categories =====
LOG_SCALE_FEATURES = [
    'scene_count', 'word_count', 'gesture_count', 'object_count', 'person_count',
    'overlay_unique_count', 'shortest_scene', 'longest_scene',
    'scene_duration_variance', 'energy_variance', 'gaze_variance'
]

SCALE_FEATURES = [
    'average_face_size', 'speech_coverage', 'energy_level', 'energy_max',
    'pitch_scatter_ratio', 'eye_contact_rate', 'emotion_consistency'
]

CATEGORICAL_FEATURES = {
    'has_captions': 'boolean',
    'dominant_emotion_id': 'ordinal_1_7'
}

# ===== Cross-Window Features (NEW) =====
CROSS_WINDOW_FEATURES = [
    'hook_to_middle_energy_delta',
    'middle_to_closing_delta',
    'eye_contact_consistency',
    'word_density_std',
    'energy_progression_slope'
]

# ===== Performance Thresholds (Relative to Baseline) =====
# Baseline: Trace 1 shows 7.2s for N=100 videos on reference hardware
# Use relative thresholds to adapt to different hardware/batch sizes

BASELINE_TIME_SECONDS = 7.2    # Reference time from Trace 1 (N=100, bucket 18-33s)
WARNING_TIME_MULTIPLIER = 2.0  # Warn if >2x baseline (e.g., >14.4s for N=100)
TIMEOUT_MULTIPLIER = 10.0      # Fail if >10x baseline (e.g., >72s for N=100)
MAX_TIMEOUT_SECONDS = 300      # Absolute maximum (5 minutes) regardless of baseline

# Memory thresholds scale with video count: ~5MB per video
BASELINE_MEMORY_MB_PER_VIDEO = 5.0  # Measured from Trace 1
WARNING_MEMORY_MULTIPLIER = 2.0     # Warn if >2x expected
FAIL_MEMORY_MULTIPLIER = 4.0        # Fail if >4x expected
MAX_MEMORY_MB = 2048                # Absolute maximum regardless of count

MINIMUM_VIDEO_COUNT = 10       # TEMPORARY: Lowered for testing (TODO: Revert to 50 for production)

# ===== Expected Column Counts =====
EXPECTED_INPUT_COLUMNS = {
    '0-3s': 24,      # 21 × 1 + 3 metadata
    '3-9s': 45,      # 21 × 2 + 3 metadata
    '9-13s': 66,     # 21 × 3 + 3 metadata
    '13-18s': 66,    # 21 × 3 + 3 metadata
    '18-33s': 129,   # 21 × 6 + 3 metadata
    '33-60s': 150,   # 21 × 7 + 3 metadata
    '60-90s': 150,
    '90-120s': 150,
}

# ===== Logging Configuration =====
LOG_PERFORMANCE = True         # Log per-operation timing
LOG_MEMORY_USAGE = True        # Log peak memory
