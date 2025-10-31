# Stage 2.7 Output Validation Guide

**Purpose**: Validate dual-flow classification outputs from ContentAnalysispt2.md implementation.

**Test Location**: `/home/jorge/rumiaifinal/data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive/`

---

## Overview

Stage 2.7 now uses **dual-flow classification** (ContentAnalysispt2.md Step 4):
- **Flow 1**: Videos with valid transcripts → Full classification (transcript + caption)
- **Flow 2**: Videos with invalid transcripts (music/noise) → Caption-only analysis

**Expected Results** (wellnesspt2_test5):
- Total videos: 300
- Valid transcripts: 204 (Flow 1)
- Invalid transcripts: 96 (Flow 2)
- Success rate: 295/300 (98.3%)
- Failed: 5 videos (JSON parsing errors)

---

## Test 1: Flow Distribution

**Verify**: Flow 1 and Flow 2 counts match validation cache.

```bash
cd /home/jorge/rumiaifinal/data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive

# Count Flow 1 (full classification)
find content_analysis -name "*_content.json" | xargs jq -r '.transcript_available' | grep -c "true"
# Expected: ~204

# Count Flow 2 (caption only)
find content_analysis -name "*_content.json" | xargs jq -r '.transcript_available' | grep -c "false"
# Expected: ~96

# Verification: Sum should equal 295-300 (total - failures)
```

**Pass Criteria**:
- Flow 1 count ≈ 200-204
- Flow 2 count ≈ 90-96
- Total = successful videos (295)

---

## Test 2: Flow 1 Schema Validation

**Verify**: Videos with valid transcripts have all 14 fields populated.

```bash
# Sample 5 Flow 1 outputs
find content_analysis -name "*_content.json" | xargs jq -r 'select(.transcript_available == true) | .video_id' | head -5 | while read vid; do
    echo "=== Video: $vid ==="
    find content_analysis -name "${vid}_content.json" | xargs jq '{
        video_id,
        transcript_available,
        taxonomy_version,
        content_category,
        hook_strategy,
        closing_strategy,
        pain_points: (.pain_points | length),
        keywords: (.keywords | length),
        engagement_drivers: (.engagement_drivers | length),
        content_tactics: (.content_tactics | length),
        caption_hook: .caption_analysis.hook_type,
        caption_cta: .caption_analysis.cta_type,
        hashtag_count: .caption_analysis.hashtag_count,
        confidence
    }'
    echo ""
done
```

**Pass Criteria** (Flow 1):
- `transcript_available: true`
- `taxonomy_version: "stage2.6_output"`
- `content_category`: NOT null (string from taxonomy)
- `hook_strategy`: NOT null (string from taxonomy)
- `closing_strategy`: NOT null (string from taxonomy)
- `pain_points`: array (may be empty [])
- `keywords`: array (may be empty [])
- `engagement_drivers`: array (may be empty [])
- `content_tactics`: array (may be empty [])
- `caption_analysis.hook_type`: one of ["statement", "question", "command", "teaser"]
- `caption_analysis.cta_type`: one of ["link_in_bio", "save_post", "comment", "follow", "share", "tag_friend", "none"]
- `caption_analysis.hashtag_count`: number ≥ 0
- `confidence`: one of ["high", "medium", "low"]

---

## Test 3: Flow 2 Schema Validation

**Verify**: Videos with invalid transcripts have defaults for transcript fields.

```bash
# Sample 5 Flow 2 outputs
find content_analysis -name "*_content.json" | xargs jq -r 'select(.transcript_available == false) | .video_id' | head -5 | while read vid; do
    echo "=== Video: $vid ==="
    find content_analysis -name "${vid}_content.json" | xargs jq '{
        video_id,
        transcript_available,
        taxonomy_version,
        content_category,
        hook_strategy,
        closing_strategy,
        pain_points,
        keywords,
        engagement_drivers,
        content_tactics,
        caption_hook: .caption_analysis.hook_type,
        caption_cta: .caption_analysis.cta_type,
        hashtag_count: .caption_analysis.hashtag_count,
        confidence,
        note
    }'
    echo ""
done
```

**Pass Criteria** (Flow 2):
- `transcript_available: false`
- `taxonomy_version: "none_no_transcript"`
- `content_category: null`
- `hook_strategy: null`
- `closing_strategy: null`
- `pain_points: []` (empty array)
- `keywords: []` (empty array)
- `engagement_drivers: []` (empty array)
- `content_tactics: []` (empty array)
- `caption_analysis.hook_type`: one of ["statement", "question", "command", "teaser"]
- `caption_analysis.cta_type`: one of ["link_in_bio", "save_post", "comment", "follow", "share", "tag_friend", "none"]
- `caption_analysis.hashtag_count`: number ≥ 0
- `confidence: "n/a"`
- `note: "No valid transcript - caption analysis only"`

---

## Test 4: Validation Cache Match

**Verify**: transcript_available matches validation cache.

```bash
# Load validation cache
CACHE_FILE="content_taxonomies/transcript_validation_cache.json"

# Check 10 random videos
find content_analysis -name "*_content.json" | shuf -n 10 | while read file; do
    vid=$(jq -r '.video_id' "$file")
    classified_valid=$(jq -r '.transcript_available' "$file")
    cached_valid=$(jq -r --arg vid "$vid" '.results[$vid].is_valid' "$CACHE_FILE")

    if [ "$classified_valid" = "true" ] && [ "$cached_valid" = "true" ]; then
        echo "✅ $vid: Flow 1 (valid in cache)"
    elif [ "$classified_valid" = "false" ] && [ "$cached_valid" = "false" ]; then
        echo "✅ $vid: Flow 2 (invalid in cache)"
    else
        echo "❌ $vid: MISMATCH! Classified=$classified_valid, Cache=$cached_valid"
    fi
done
```

**Pass Criteria**:
- All 10 samples should show ✅
- No ❌ mismatches

---

## Test 5: Caption Analysis Quality

**Verify**: caption_analysis fields are populated for ALL videos (both flows).

```bash
# Check caption analysis completeness
find content_analysis -name "*_content.json" | xargs jq -r '
    select(.caption_analysis.hook_type == null or .caption_analysis.cta_type == null or .caption_analysis.hashtag_count == null) |
    .video_id
'
```

**Pass Criteria**:
- **Output**: Empty (no results)
- If videos are listed: caption_analysis has null fields (ERROR)

---

## Test 6: Taxonomy Field Quality (Flow 1 Only)

**Verify**: Flow 1 videos have taxonomy fields from curated taxonomy.

```bash
# Load taxonomy
TAXONOMY="content_taxonomies/wellnesspt2_test5_taxonomy.json"

# Sample Flow 1 outputs and check taxonomy field values
find content_analysis -name "*_content.json" | xargs jq -r 'select(.transcript_available == true) | .video_id' | head -10 | while read vid; do
    file=$(find content_analysis -name "${vid}_content.json")
    content_cat=$(jq -r '.content_category' "$file")
    hook_strat=$(jq -r '.hook_strategy' "$file")

    # Check if content_category exists in taxonomy
    taxonomy_has_cat=$(jq --arg cat "$content_cat" '.content_categories[]? | select(.name == $cat) | .name' "$TAXONOMY")

    if [ -n "$taxonomy_has_cat" ]; then
        echo "✅ $vid: content_category '$content_cat' found in taxonomy"
    else
        echo "⚠️  $vid: content_category '$content_cat' NOT in taxonomy (hallucination?)"
    fi
done
```

**Pass Criteria**:
- Most should show ✅ (taxonomy match)
- Some ⚠️ acceptable if LLM made minor typos (check manually)

---

## Test 7: hashtag_count Accuracy (M10 FIX)

**Verify**: hashtag_count matches actual hashtags in caption.

```bash
# Check hashtag_count calculation
find content_analysis -name "*_content.json" | shuf -n 10 | while read file; do
    vid=$(jq -r '.video_id' "$file")
    reported_count=$(jq -r '.caption_analysis.hashtag_count' "$file")

    # Get actual caption
    caption=$(jq -r '.caption // ""' "$file")

    # Count hashtags (words starting with #)
    actual_count=$(echo "$caption" | grep -o '#[[:alnum:]_]*' | wc -l)

    if [ "$reported_count" -eq "$actual_count" ]; then
        echo "✅ $vid: hashtag_count=$reported_count (correct)"
    else
        echo "❌ $vid: reported=$reported_count, actual=$actual_count"
    fi
done
```

**Pass Criteria**:
- All 10 samples should show ✅
- hashtag_count matches actual count in caption

---

## Test 8: Confidence Distribution

**Verify**: Flow 1 has normal confidence distribution, Flow 2 has "n/a".

```bash
echo "=== Flow 1 Confidence Distribution ==="
find content_analysis -name "*_content.json" | xargs jq -r 'select(.transcript_available == true) | .confidence' | sort | uniq -c

echo -e "\n=== Flow 2 Confidence Distribution ==="
find content_analysis -name "*_content.json" | xargs jq -r 'select(.transcript_available == false) | .confidence' | sort | uniq -c
```

**Pass Criteria** (Flow 1):
- Mix of "high", "medium", "low"
- "high" should be most common (>50%)

**Pass Criteria** (Flow 2):
- ALL should be "n/a"

---

## Test 9: Raw vs Validated Output Check

**Verify**: Raw LLM outputs were saved (debugging feature).

```bash
# Check if raw_llm_output directory exists
ls -d content_analysis/raw_llm_output/bucket_* 2>/dev/null && echo "✅ Raw outputs saved" || echo "❌ No raw outputs found"

# Count raw outputs
find content_analysis/raw_llm_output -name "*_raw.json" 2>/dev/null | wc -l
# Expected: ~295-300

# Compare raw vs validated for one video
SAMPLE_VID=$(find content_analysis -name "*_content.json" | head -1 | xargs jq -r '.video_id')
echo "Sample video: $SAMPLE_VID"
echo "Raw output:"
find content_analysis/raw_llm_output -name "${SAMPLE_VID}_raw.json" | xargs jq . | head -20
echo -e "\nValidated output:"
find content_analysis -name "${SAMPLE_VID}_content.json" | xargs jq . | head -20
```

**Pass Criteria**:
- raw_llm_output/ directory exists
- ~295-300 raw outputs present
- Raw output is similar but may have formatting differences

---

## Test 10: Failure Analysis

**Verify**: Failed videos are documented.

```bash
# Check Stage 2.7 logs for failures
# Expected: 5 failures (JSON parsing errors)

echo "Expected failures: 5 videos"
echo "Failed video IDs:"
echo "- 7486676266876194090"
echo "- 7511902992111619333"
echo "- 7482222847159569706"
echo "- 7564547563353804087"
echo "- 7477959232617205038"

# Verify these videos don't have output files
for vid in 7486676266876194090 7511902992111619333 7482222847159569706 7564547563353804087 7477959232617205038; do
    if find content_analysis -name "${vid}_content.json" | grep -q .; then
        echo "❌ $vid: Output file exists (should have failed)"
    else
        echo "✅ $vid: No output file (correctly failed)"
    fi
done
```

**Pass Criteria**:
- All 5 failed videos should show ✅ (no output file)

---

## Summary Validation Command

**Run all tests in sequence:**

```bash
cd /home/jorge/rumiaifinal/data/clients/rollo_test5/hashtags/wellnesspt2_test5/top_contrastive

echo "=== Test 1: Flow Distribution ==="
flow1=$(find content_analysis -name "*_content.json" | xargs jq -r '.transcript_available' | grep -c "true")
flow2=$(find content_analysis -name "*_content.json" | xargs jq -r '.transcript_available' | grep -c "false")
total=$((flow1 + flow2))
echo "Flow 1 (full): $flow1"
echo "Flow 2 (caption): $flow2"
echo "Total: $total"
echo "Expected: Flow 1 ≈ 204, Flow 2 ≈ 96, Total = 295-300"
echo ""

echo "=== Test 2: Schema Check (Sample Flow 1) ==="
find content_analysis -name "*_content.json" | xargs jq -r 'select(.transcript_available == true) | .video_id' | head -1 | while read vid; do
    find content_analysis -name "${vid}_content.json" | xargs jq '{transcript_available, taxonomy_version, content_category, confidence}'
done
echo ""

echo "=== Test 3: Schema Check (Sample Flow 2) ==="
find content_analysis -name "*_content.json" | xargs jq -r 'select(.transcript_available == false) | .video_id' | head -1 | while read vid; do
    find content_analysis -name "${vid}_content.json" | xargs jq '{transcript_available, taxonomy_version, content_category, confidence, note}'
done
echo ""

echo "=== Test 5: Caption Analysis Completeness ==="
missing=$(find content_analysis -name "*_content.json" | xargs jq -r 'select(.caption_analysis.hook_type == null or .caption_analysis.cta_type == null) | .video_id' | wc -l)
if [ "$missing" -eq 0 ]; then
    echo "✅ All videos have complete caption analysis"
else
    echo "❌ $missing videos missing caption analysis fields"
fi
echo ""

echo "=== Test 8: Confidence Distribution ==="
echo "Flow 1:"
find content_analysis -name "*_content.json" | xargs jq -r 'select(.transcript_available == true) | .confidence' | sort | uniq -c
echo "Flow 2:"
find content_analysis -name "*_content.json" | xargs jq -r 'select(.transcript_available == false) | .confidence' | sort | uniq -c
echo ""

echo "=== Test 9: Raw Outputs ==="
raw_count=$(find content_analysis/raw_llm_output -name "*_raw.json" 2>/dev/null | wc -l)
echo "Raw LLM outputs saved: $raw_count"
echo "Expected: 295-300"
```

---

## Expected Results Summary

**Successful Implementation:**
- ✅ Flow 1 count: 190-204 videos
- ✅ Flow 2 count: 90-96 videos
- ✅ Total: 295-300 videos (98%+ success rate)
- ✅ Flow 1 has populated taxonomy fields
- ✅ Flow 2 has null taxonomy fields + note
- ✅ All videos have caption_analysis
- ✅ hashtag_count matches actual count
- ✅ Flow 1 confidence: high/medium/low distribution
- ✅ Flow 2 confidence: all "n/a"
- ✅ Raw outputs saved for debugging
- ✅ 5 failures documented (JSON parsing errors)

---

## Troubleshooting

### Issue: Flow counts don't match validation cache

**Check**:
```bash
jq '.stats' content_taxonomies/transcript_validation_cache.json
```

**Expected**: 204 valid, 96 invalid

**Solution**: If mismatch, validation cache may be stale. Re-run Stage 2.5.1.

---

### Issue: Flow 1 videos have null taxonomy fields

**Check**:
```bash
find content_analysis -name "*_content.json" | xargs jq -r 'select(.transcript_available == true and .content_category == null) | .video_id'
```

**Solution**: LLM classification failed. Check raw_llm_output for these videos.

---

### Issue: Flow 2 videos have populated taxonomy fields

**Check**:
```bash
find content_analysis -name "*_content.json" | xargs jq -r 'select(.transcript_available == false and .content_category != null) | .video_id'
```

**Solution**: Flow routing bug. Check validation cache for these video IDs.

---

## Contact

If all tests pass: **Implementation is successful!** 🎉

The dual-flow classification is working correctly:
- Videos with valid transcripts get full analysis
- Videos with music/noise get caption-only analysis
- No data loss - all videos classified!
