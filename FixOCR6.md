# FixOCR6: Spatial Proximity Clustering for OCR Text Merging

## Problem Statement

Current temporal clustering successfully reduces OCR overcounting but misses spatially adjacent text that forms single visual overlays. Example: "daily movement" is counted as 2 overlays ("movement" + "daily") when it should be 1.

**Visual Reality**: "daily movement" appears as one cohesive text overlay from 8s-10s
**Current Count**: 2 separate overlays
**Target Count**: 1 merged overlay

## Current OCR Data Structure

```json
{
  "8s": [
    {
      "text": "movement",
      "position": "right",
      "size": "medium",
      "style": "normal"
    },
    {
      "text": "daily",
      "position": "right",
      "size": "medium",
      "style": "normal"
    }
  ]
}
```

## Implementation Strategy

### Phase 1: Spatial Grouping Within Time Buckets

**Enhancement to existing temporal clustering algorithm:**

```python
def spatial_cluster_within_bucket(bucket_texts: List[Dict]) -> List[str]:
    """
    Group texts by spatial metadata and attempt phrase formation
    """
    # Group by spatial signature
    spatial_groups = {}
    for entry in bucket_texts:
        spatial_key = (
            entry['position'],  # "right", "left", "center"
            entry['size'],      # "small", "medium", "large"
            entry['style']      # "normal", "bold", etc.
        )
        if spatial_key not in spatial_groups:
            spatial_groups[spatial_key] = []
        spatial_groups[spatial_key].append(entry['text'])

    # Process each spatial group
    merged_texts = []
    for group_texts in spatial_groups.values():
        merged_group = merge_spatial_group(group_texts)
        merged_texts.extend(merged_group)

    return merged_texts

def merge_spatial_group(texts: List[str]) -> List[str]:
    """
    Attempt to merge texts in same spatial location into coherent phrases
    """
    if len(texts) <= 1:
        return texts

    # Try all combinations to form valid phrases
    merged = []
    used_indices = set()

    for i, text1 in enumerate(texts):
        if i in used_indices:
            continue

        best_combination = [text1]
        best_indices = {i}

        # Look for complementary texts
        for j, text2 in enumerate(texts):
            if j in used_indices or j == i:
                continue

            if can_form_phrase(best_combination + [text2]):
                best_combination.append(text2)
                best_indices.add(j)

        if len(best_combination) > 1:
            # Merge into single phrase
            merged_phrase = combine_into_phrase(best_combination)
            merged.append(merged_phrase)
        else:
            merged.append(text1)

        used_indices.update(best_indices)

    return merged
```

### Phase 2: Phrase Detection Logic

```python
def can_form_phrase(texts: List[str]) -> bool:
    """
    Determine if texts can logically form a coherent phrase
    """
    if len(texts) < 2:
        return False

    # Normalize texts
    normalized = [normalize_text_global(text) for text in texts]

    # Check common phrase patterns
    combined_variations = generate_phrase_combinations(normalized)

    for combination in combined_variations:
        if is_valid_phrase(combination):
            return True

    return False

def is_valid_phrase(phrase: str) -> bool:
    """
    Check if combined text forms a valid phrase
    """
    # Known fitness/health phrases
    fitness_phrases = {
        "daily movement", "daily exercise", "daily workout",
        "more variety", "plant foods", "vitamin d",
        "gut health", "better health", "wellness hacks"
    }

    phrase_lower = phrase.lower().strip()

    # Direct match
    if phrase_lower in fitness_phrases:
        return True

    # Partial matches for compound phrases
    for known_phrase in fitness_phrases:
        if phrase_lower in known_phrase or known_phrase in phrase_lower:
            if len(phrase_lower) >= len(known_phrase) * 0.7:  # Threshold
                return True

    # Generic patterns (adjective + noun)
    words = phrase_lower.split()
    if len(words) == 2:
        # Check for common adjective-noun patterns
        if has_adjective_noun_pattern(words):
            return True

    return False

def combine_into_phrase(texts: List[str]) -> str:
    """
    Intelligently combine texts into single phrase
    """
    # Sort by length (shorter words often come first)
    sorted_texts = sorted(texts, key=len)

    # Try different orderings
    combinations = [
        " ".join(sorted_texts),
        " ".join(reversed(sorted_texts)),
        " ".join(sorted(sorted_texts, key=lambda x: x.lower()))
    ]

    # Return the most phrase-like combination
    for combo in combinations:
        if is_valid_phrase(combo):
            return combo.strip()

    # Fallback: alphabetical order
    return " ".join(sorted(texts)).strip()
```

### Phase 3: Integration with Existing Algorithm

**Modified temporal clustering flow:**

```python
def temporal_cluster_overlays_with_spatial(overlay_entries: List[Dict]) -> List[str]:
    """
    Enhanced temporal clustering with spatial proximity merging
    """
    TEMPORAL_BUCKET_SIZE = 0.5
    time_buckets = {}

    # Step 1: Temporal bucketing (unchanged)
    for entry in overlay_entries:
        time_bucket = round(entry['timestamp'] / TEMPORAL_BUCKET_SIZE) * TEMPORAL_BUCKET_SIZE
        if time_bucket not in time_buckets:
            time_buckets[time_bucket] = []
        time_buckets[time_bucket].append({
            'text': entry['text'],
            'position': entry.get('position', 'unknown'),
            'size': entry.get('size', 'medium'),
            'style': entry.get('style', 'normal')
        })

    bucket_results = []

    # Step 2: Within-bucket processing with spatial clustering
    for bucket_entries in time_buckets.values():
        # First apply spatial clustering
        spatially_clustered = spatial_cluster_within_bucket(bucket_entries)

        # Then apply existing fuzzy matching
        normalized_texts = [normalize_text_global(text) for text in spatially_clustered]
        unique_texts = aggressive_fuzzy_matching(normalized_texts)
        bucket_results.extend(unique_texts)

    # Step 3: Cross-bucket clustering (unchanged)
    return aggressive_fuzzy_matching(bucket_results)
```

## Expected Impact

**Test Case: Video 7099027230512139526 Closing Segment**

**Before (current):**
- "more variety of plant foods"
- "movement"
- "daily"
**Total: 3 overlays**

**After (with spatial clustering):**
- "more variety of plant foods"
- "daily movement" (merged from "movement" + "daily")
**Total: 2 overlays**

**Reduction: 33% improvement in accuracy**

## Implementation Location

**File**: `/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py`

**Method**: Replace `temporal_cluster_overlays()` with `temporal_cluster_overlays_with_spatial()`

**Integration**: Modify `process_text_overlays()` to pass position metadata to clustering algorithm

## Deployment Strategy

1. **Backup current implementation**
2. **Add spatial clustering functions** to temporal_compute.py
3. **Update process_text_overlays()** to use enhanced algorithm
4. **Test on known cases** (7099027230512139526, 7459548276413435178)
5. **Single-day deployment** with monitoring
6. **Rollback plan** if overcounting increases

## Risk Assessment

**Low Risk:**
- Builds on proven temporal clustering foundation
- Uses existing OCR metadata (no new data dependencies)
- Conservative phrase detection (only merges high-confidence cases)
- Maintains existing fuzzy matching as fallback

**Success Metrics:**
- Reduce false positive overlays by 20-30%
- Maintain or improve temporal clustering accuracy
- No increase in false negatives (missed overlays)