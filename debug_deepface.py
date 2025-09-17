#!/usr/bin/env python3
"""
Debug version of DeepFace gender detection to see frame-by-frame results.
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


def debug_analyze_video(video_path: str):
    """Analyze video with detailed frame-by-frame output."""

    print(f"Analyzing: {video_path}")
    start_time = time.time()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s duration")

    # Determine frame count based on duration
    if duration < 5:
        num_frames = 2
    elif duration < 15:
        num_frames = 3
    elif duration < 30:
        num_frames = 5
    else:
        num_frames = 7

    print(f"Will sample {num_frames} frames")

    # Sample frames evenly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    print(f"Frame indices: {frame_indices}")

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))

    cap.release()
    print(f"Extracted {len(frames)} frames")

    if not frames:
        print("No frames extracted")
        return

    # Analyze frames with detailed output
    gender_votes = []
    multi_person_count = 0

    for frame_idx, frame in frames:
        print(f"\n--- Frame {frame_idx} ---")
        try:
            result = DeepFace.analyze(
                frame,
                actions=['gender'],
                enforce_detection=False,
                detector_backend='opencv',
                silent=True
            )

            print(f"DeepFace found {len(result)} face(s)")

            if result and len(result) > 0:
                if len(result) > 1:
                    print(f"Multiple people detected: {len(result)} faces")
                    multi_person_count += 1
                    # Print details about each face
                    for i, face in enumerate(result):
                        gender_data = face.get('gender', {})
                        dominant = face.get('dominant_gender', 'unknown')
                        region = face.get('region', {})
                        print(f"  Face {i+1}: {dominant} at region {region}")
                        print(f"    Gender scores: {gender_data}")
                else:
                    # Single person
                    face = result[0]
                    gender_data = face.get('gender', {})
                    dominant = face.get('dominant_gender', 'unknown')
                    confidence = gender_data.get(dominant, 0) / 100.0 if gender_data else 0
                    region = face.get('region', {})

                    print(f"Single person: {dominant} (confidence: {confidence:.2f})")
                    print(f"  Region: {region}")
                    print(f"  Gender scores: {gender_data}")

                    # Map to our format
                    gender = 'male' if dominant.lower() == 'man' else 'female'
                    gender_votes.append({
                        'gender': gender,
                        'confidence': confidence,
                        'frame': frame_idx
                    })
            else:
                print("No faces detected in this frame")

        except Exception as e:
            print(f"Error analyzing frame: {e}")
            continue

    print(f"\n--- Summary ---")
    print(f"Multi-person frames: {multi_person_count}/{len(frames)}")
    print(f"Gender votes: {gender_votes}")

    # Final decision logic
    if multi_person_count > 0:
        print("RESULT: multiple_people (due to multi-person frames)")
    elif not gender_votes:
        print("RESULT: no faces detected")
    else:
        male_votes = [v for v in gender_votes if v['gender'] == 'male']
        female_votes = [v for v in gender_votes if v['gender'] == 'female']

        print(f"Male votes: {len(male_votes)}")
        print(f"Female votes: {len(female_votes)}")

        if len(male_votes) > len(female_votes):
            avg_conf = np.mean([v['confidence'] for v in male_votes])
            print(f"RESULT: male (confidence: {avg_conf:.2f})")
        else:
            avg_conf = np.mean([v['confidence'] for v in female_votes])
            print(f"RESULT: female (confidence: {avg_conf:.2f})")

    processing_time = time.time() - start_time
    print(f"Processing took {processing_time:.2f} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_deepface.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    debug_analyze_video(video_path)