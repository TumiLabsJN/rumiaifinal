#!/usr/bin/env python3
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

# Test texts from segment_3
texts = [
    'Iremember when | first',
    'wanted to lose weight',
    'Ikept tellingmyselflike oh',
    'Ineed to start going to the',
    'gvm',
    'Ineed to start going to the',
    'gym',
    'Ineed to start doing this',
    'and this and this',
    'should be approaching',
    'weight loss'
]

# Speech segments
speeches = [
    'I kept telling myself like,',
    'oh, I need to start going to the gym,',
    'I need to start doing this and this and this and this.',
    'And that is not how you should be approaching weight loss.'
]

print('Testing speech overlap scores:')
print('=' * 80)

for text in texts[:11]:  # First 11 unique texts
    text_fixed = fix_ocr_errors(text)
    text_normalized = normalize_text(text_fixed)

    max_score = 0.0
    best_match = ''

    for speech in speeches:
        speech_fixed = fix_ocr_errors(speech)
        speech_normalized = normalize_text(speech_fixed)

        # Character similarity
        char_sim = SequenceMatcher(None, text_normalized, speech_normalized).ratio()

        # Word overlap
        text_words = set(text_normalized.split())
        speech_words = set(speech_normalized.split())
        word_overlap = len(text_words.intersection(speech_words)) / len(text_words) if text_words else 0.0

        score = max(char_sim, word_overlap)

        if score > max_score:
            max_score = score
            best_match = speech[:40]

    classification = 'CAPTION' if max_score > 0.7 else ('UNCERTAIN' if max_score > 0.3 else 'OVERLAY')
    print(f'{classification:10} {max_score:5.2f}  |  "{text}"')
    print(f'           Match: "{best_match}"')
    print()
