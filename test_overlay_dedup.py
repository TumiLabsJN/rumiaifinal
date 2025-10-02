#!/usr/bin/env python3
"""
Test overlay deduplication to understand why we get 10 overlays instead of 5.
"""
import difflib

def should_merge_texts_aggressive(text1, text2):
    """Aggressive text similarity for temporal clustering."""
    if not text1 or not text2:
        return False

    t1, t2 = text1.lower().strip(), text2.lower().strip()

    if t1 == t2:
        return True

    AGGRESSIVE_CHAR_THRESHOLD = 0.7
    AGGRESSIVE_TOKEN_THRESHOLD = 0.6

    char_sim = difflib.SequenceMatcher(None, t1, t2).ratio()
    tokens1 = set(t1.split())
    tokens2 = set(t2.split())
    if tokens1 and tokens2:
        token_sim = len(tokens1 & tokens2) / len(tokens1 | tokens2)
    else:
        token_sim = 0.0

    return char_sim > AGGRESSIVE_CHAR_THRESHOLD or token_sim > AGGRESSIVE_TOKEN_THRESHOLD

def aggressive_fuzzy_matching(texts):
    """More aggressive fuzzy matching for temporal clustering context."""
    if not texts:
        return []

    unique_texts = []

    for text in texts:
        if not text or not text.strip():
            continue

        found_match = False

        for i, unique_text in enumerate(unique_texts):
            if should_merge_texts_aggressive(text, unique_text):
                found_match = True
                if len(text) > len(unique_text):
                    unique_texts[i] = text
                break

        if not found_match:
            unique_texts.append(text)

    return unique_texts

# From our test: these are the texts classified as OVERLAYS (overlap < 0.3)
overlays = [
    "Iremember when | first",
    "wanted to lose weight",
    "gvm",
    "sayoh",
    "sayoh",
    "example for me I wanna",
    "sayoh",
]

# Also the UNCERTAIN texts (0.3-0.7) that might be classified as overlays
uncertain = [
    "and that's not howyoU",
    "be a modeland I I don't"
]

print("Original overlays (from speech overlap < 0.3):")
for i, text in enumerate(overlays, 1):
    print(f"  {i}. \"{text}\"")
print(f"Total: {len(overlays)}")
print()

print("Uncertain texts (might become overlays):")
for i, text in enumerate(uncertain, 1):
    print(f"  {i}. \"{text}\"")
print(f"Total: {len(uncertain)}")
print()

# Combine for worst case
all_possible_overlays = overlays + uncertain
print(f"If ALL uncertain become overlays: {len(all_possible_overlays)} texts")
print()

# Apply deduplication
deduplicated = aggressive_fuzzy_matching(all_possible_overlays)

print("After aggressive fuzzy matching deduplication:")
for i, text in enumerate(deduplicated, 1):
    print(f"  {i}. \"{text}\"")
print(f"Total unique: {len(deduplicated)}")
print()

print(f"Expected in output: {len(deduplicated)}")
print(f"Actual in output: 10")
print()

if len(deduplicated) == 10:
    print("✓ This matches! The 10 overlays are:")
    print("  - 5 original overlays (after dedup)")
    print("  - 2 uncertain texts classified as overlays")
    print("  - 3 duplicates that didn't merge")
else:
    print(f"✗ Mismatch: got {len(deduplicated)} but expected 10")
