#!/usr/bin/env python3
import json
from difflib import SequenceMatcher
import re

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fix_ocr_errors(text):
    if not text:
        return text
    text = re.sub(r'(\w)([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\w)\|', r'\1 ', text)
    return text

def calculate_speech_overlap(text, timestamp, speech_segments):
    """Calculate % overlap between text and speech at given timestamp."""
    if not speech_segments:
        return 0.0

    # Find speech segments overlapping this timestamp
    for segment in speech_segments:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', seg_start + 1)
        if seg_start <= timestamp <= seg_end:

            # TIMING STRICTNESS: Prevent false positives from thematic overlays
            time_alignment = min(abs(timestamp - seg_start), abs(timestamp - seg_end))
            if time_alignment > 2.0:  # Text >2s from speech boundaries
                return 0.0  # Too far from speech timing - likely thematic overlay

            # Get segment text
            segment_text = segment.get('text', '').lower()
            if not segment_text:
                continue

            # Apply OCR error correction before normalization
            text_fixed = fix_ocr_errors(text)
            segment_text_fixed = fix_ocr_errors(segment_text)

            # Normalize both texts
            text_normalized = normalize_text(text_fixed)
            segment_normalized = normalize_text(segment_text_fixed)

            # Calculate overlap using fuzzy matching to handle OCR errors
            if not text_normalized:
                return 0.0

            # Use fuzzy similarity to handle OCR variations
            char_similarity = SequenceMatcher(None, text_normalized, segment_normalized).ratio()

            # Also calculate word overlap for additional confidence
            text_words = set(text_normalized.split())
            segment_words = set(segment_normalized.split())

            if text_words and segment_words:
                overlap_words = text_words.intersection(segment_words)
                word_overlap_ratio = len(overlap_words) / len(text_words)
            else:
                word_overlap_ratio = 0.0

            # Use the higher of character similarity or word overlap
            overlap_ratio = max(char_similarity, word_overlap_ratio)
            return overlap_ratio

    return 0.0

# Load the raw data
with open('unified_analysis/7480428850522950920.json') as f:
    data = json.load(f)

# Get text entries for segment_3 (26.6 to 38.4)
text_entries = []
for entry in data['timeline']['entries']:
    if entry['entry_type'] == 'text':
        start = entry.get('start', 0)
        if 26.6 <= start <= 38.4:
            text_entries.append({
                'time': start,
                'text': entry['data']['text'],
                'entry': entry
            })

# Get speech segments
speech_segments = []
for entry in data['timeline']['entries']:
    if entry['entry_type'] == 'speech':
        start = entry.get('start', 0)
        if 26.6 <= start <= 38.4:
            speech_segments.append({
                'start': entry.get('start'),
                'end': entry.get('end'),
                'text': entry['data']['text']
            })

print('Speech segments in segment_3:')
for seg in speech_segments:
    print(f"  {seg['start']:.2f}-{seg['end']:.2f}: \"{seg['text']}\"")
print()

# Calculate speech overlap for each text
HIGH_SPEECH_THRESHOLD = 0.7
LOW_SPEECH_THRESHOLD = 0.3

captions = []
overlays = []
uncertain = []

for entry in text_entries:
    text = entry['text']
    timestamp = entry['time']

    overlap = calculate_speech_overlap(text, timestamp, speech_segments)

    if overlap > HIGH_SPEECH_THRESHOLD:
        captions.append((text, timestamp, overlap))
    elif overlap < LOW_SPEECH_THRESHOLD:
        overlays.append((text, timestamp, overlap))
    else:
        uncertain.append((text, timestamp, overlap))

print(f'Classification results:')
print(f'  Captions: {len(captions)}')
print(f'  Overlays: {len(overlays)}')
print(f'  Uncertain: {len(uncertain)}')
print()

print('CAPTIONS (overlap > 0.7):')
for text, ts, score in captions:
    print(f'  {score:.2f} @ {ts:.1f}s: "{text}"')
print()

print('OVERLAYS (overlap < 0.3):')
for text, ts, score in overlays:
    print(f'  {score:.2f} @ {ts:.1f}s: "{text}"')
print()

if uncertain:
    print('UNCERTAIN (0.3-0.7):')
    for text, ts, score in uncertain:
        print(f'  {score:.2f} @ {ts:.1f}s: "{text}"')
    print()

# Count unique texts in each category
unique_captions = set([t[0] for t in captions])
unique_overlays = set([t[0] for t in overlays])
unique_uncertain = set([t[0] for t in uncertain])

print(f'Unique counts:')
print(f'  Unique captions: {len(unique_captions)}')
print(f'  Unique overlays: {len(unique_overlays)}')
print(f'  Unique uncertain: {len(unique_uncertain)}')
