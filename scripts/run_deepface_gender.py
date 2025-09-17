#!/usr/bin/env python3
"""
Standalone DeepFace gender detection script.
This works around the memory corruption issues in the integrated service.

Usage:
    python scripts/run_deepface_gender.py <video_path> [output_path]
"""

import sys
import json
import time
import cv2
from pathlib import Path
import numpy as np

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from deepface import DeepFace


def analyze_video(video_path: str) -> dict:
    """Analyze video for gender detection."""

    start_time = time.time()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            'gender': None,
            'confidence': 0.0,
            'error': 'Cannot open video file'
        }

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    # Determine frame count based on duration
    if duration < 5:
        num_frames = 2
    elif duration < 15:
        num_frames = 3
    elif duration < 30:
        num_frames = 5
    else:
        num_frames = 7

    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    if not frames:
        return {
            'gender': None,
            'confidence': 0.0,
            'error': 'No frames extracted'
        }

    # Analyze frames
    gender_votes = []
    multi_person_count = 0

    for frame in frames:
        try:
            result = DeepFace.analyze(
                frame,
                actions=['gender'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )

            if result and len(result) > 0:
                # Filter out small faces (likely false positives from logos/watermarks)
                valid_faces = []
                for face in result:
                    region = face.get('region', {})
                    width = region.get('w', 0)
                    height = region.get('h', 0)
                    # Only keep faces larger than 120x120 pixels (filter out logos/watermarks)
                    if width >= 120 and height >= 120:
                        valid_faces.append(face)

                if len(valid_faces) > 1:
                    # Multiple valid people detected
                    multi_person_count += 1
                elif len(valid_faces) == 1:
                    # Single person
                    face = valid_faces[0]
                    dominant = face['dominant_gender']
                    confidence = face['gender'][dominant] / 100.0

                    # Map to our format
                    gender = 'male' if dominant.lower() == 'man' else 'female'
                    gender_votes.append({
                        'gender': gender,
                        'confidence': confidence
                    })
        except Exception as e:
            # Skip frame on error
            continue

    # Check for multi-person scenario
    if multi_person_count > 0:
        return {
            'gender': 'multiple_people',
            'confidence': 0.0,
            'method': 'deepface',
            'frames_analyzed': len(frames),
            'multi_person_frames': multi_person_count,
            'processing_ms': int((time.time() - start_time) * 1000),
            'note': 'Multiple people detected - use self-normalization for pitch'
        }

    # Aggregate votes
    if not gender_votes:
        return {
            'gender': None,
            'confidence': 0.0,
            'method': 'deepface',
            'frames_analyzed': len(frames),
            'processing_ms': int((time.time() - start_time) * 1000),
            'error': 'no_faces_detected'
        }

    # Count gender occurrences
    male_votes = [v for v in gender_votes if v['gender'] == 'male']
    female_votes = [v for v in gender_votes if v['gender'] == 'female']

    if len(male_votes) > len(female_votes):
        final_gender = 'male'
        final_confidence = np.mean([v['confidence'] for v in male_votes])
    elif len(female_votes) > len(male_votes):
        final_gender = 'female'
        final_confidence = np.mean([v['confidence'] for v in female_votes])
    else:
        # Tie - use highest confidence
        max_male = max(male_votes, key=lambda x: x['confidence']) if male_votes else {'confidence': 0}
        max_female = max(female_votes, key=lambda x: x['confidence']) if female_votes else {'confidence': 0}

        if max_male['confidence'] > max_female['confidence']:
            final_gender = 'male'
            final_confidence = max_male['confidence']
        else:
            final_gender = 'female'
            final_confidence = max_female['confidence']

    return {
        'gender': final_gender,
        'confidence': float(final_confidence),
        'method': 'deepface',
        'frames_analyzed': len(frames),
        'detector_backend': 'opencv',
        'processing_ms': int((time.time() - start_time) * 1000)
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_deepface_gender.py <video_path> [output_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print(f"Analyzing: {video_path}")
    result = analyze_video(video_path)

    # Save to file if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to: {output_path}")

    # Print result
    print(json.dumps(result, indent=2))

    return 0 if result.get('gender') else 1


if __name__ == "__main__":
    exit(main())