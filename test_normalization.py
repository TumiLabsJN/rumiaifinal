#!/usr/bin/env python3
import re

def normalize_text(text: str) -> str:
    """Normalize text for grouping similar OCR detections."""
    original = text.strip()
    # Convert to lowercase
    text = text.lower()

    # Remove remaining emojis and special characters, keep alphanumeric, spaces, and brackets
    text = re.sub(r'[^a-z0-9\s\[\]]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # If normalization resulted in empty string but original had content,
    # it was an unmapped emoji/symbol - create unique identifier
    if not text and original:
        import hashlib
        text = f"[emoji_{hashlib.md5(original.encode()).hexdigest()[:6]}]"
    return text

# Raw texts from segment_1 (3.0-4.33s)
raw_texts = [
    "3 THINGS YU NEED FOR",
    "BETTER GUT HEALTH",
    "by a nutritionist",
    "THINGS YOU NEED FOR",
    "BETTER GUt HEALTH",
    "3 THINGS YOU NEED FOR",
    "more sunshine + vitamin D"
]

normalized = set()
for text in raw_texts:
    norm = normalize_text(text)
    normalized.add(norm)
    print(f'"{text}" → "{norm}"')

print(f"\nUnique normalized texts: {len(normalized)}")
for i, text in enumerate(sorted(normalized), 1):
    print(f"{i}. {text}")