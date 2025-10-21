# Production Testing Cheatsheet

**Purpose**: Quick validation checklist for each pipeline stage during production runs

**Last Updated**: 2025-01-28

---

## Stages

### 3.4 Review CSV Generation

**Quick Validation Checklist**:

1. **Check output file exists**
   ```bash
   # Path pattern: {analysis_base}/buckets/bucket_{duration}/validation/video_review.csv
   ls -lh data/clients/{client}/hashtags/{target}/{mode}_{strategy}/buckets/bucket_*/validation/video_review.csv
   ```

2. **Verify file size is reasonable**
   ```bash
   # Expected: ~1KB per video (for N=100: ~100KB)
   # Too small (<1KB total): Likely only 1-2 videos, check logs
   # Too large (>500KB for N=100): Unexpected, investigate
   du -h data/clients/{client}/hashtags/{target}/{mode}_{strategy}/buckets/*/validation/video_review.csv
   ```

3. **Open in Excel/LibreOffice and verify structure**
   ```bash
   # Quick preview first 5 rows
   head -5 data/clients/{client}/hashtags/{target}/{mode}_{strategy}/buckets/bucket_18-33s/validation/video_review.csv

   # Check column count (should be aggregated_features.csv columns + 1)
   head -1 video_review.csv | tr ',' '\n' | wc -l
   ```

4. **Validate column order**
   - Column 1: `video_id` (numeric string like "7428596413707144481")
   - Column 2: `url` (should be `https://www.tiktok.com/@...`)
   - Column 3: `duration` (float, matches bucket range)
   - Remaining: Temporal features (hook_*, middle_*, closing_*)

5. **Spot-check clickable URLs**
   ```bash
   # Extract first URL and test in browser
   awk -F',' 'NR==2 {print $2}' video_review.csv

   # Copy URL and paste in browser - should load TikTok video
   # If URL broken/404: Stage 2 url field issue
   ```

6. **Compare row counts with aggregated_features.csv**
   ```bash
   # Review CSV should have ≤ rows than aggregated CSV
   wc -l buckets/bucket_18-33s/validation/video_review.csv
   wc -l buckets/bucket_18-33s/ml_analysis/aggregated_features.csv

   # If difference > 10%: Many videos missing URLs, check Stage 2 logs
   ```

7. **Check pipeline logs for Stage 3.4 messages**
   ```bash
   # Look for Stage 3.4 completion messages
   grep "Stage 3.4" logs/rumiai_ml_*.log

   # Expected: "✓ Review CSV: validation/video_review.csv" per bucket
   # Warning OK: "⚠️ Review CSV not generated (all videos missing url)"
   # Error NOT OK: Any other error message
   ```

8. **Verify all winning buckets have review CSVs**
   ```bash
   # List all validation directories
   find data/clients/{client}/hashtags/{target}/{mode}_{strategy}/buckets/ -name "video_review.csv"

   # Count should match winning_buckets count (typically 3)
   # If fewer: Check which bucket failed in logs
   ```

9. **Validate feature values match aggregated CSV (spot check)**
   ```bash
   # Compare first row of both CSVs (excluding url column)
   # Review CSV row 1, columns 1,3-N should match aggregated CSV row 1, columns 1-N

   # Quick Python check:
   python3 -c "
   import pandas as pd
   df_review = pd.read_csv('buckets/bucket_18-33s/validation/video_review.csv')
   df_agg = pd.read_csv('buckets/bucket_18-33s/ml_analysis/aggregated_features.csv')

   # Drop url column and compare
   df_review_no_url = df_review.drop(columns=['url'])
   print('Rows match:', df_review_no_url.equals(df_agg))
   print('Review shape:', df_review.shape)
   print('Aggregated shape:', df_agg.shape)
   "
   ```

10. **Excel Outlier Investigation (Manual QA)**
    - Open video_review.csv in Excel
    - Select all cells → Home → Conditional Formatting → Color Scales (Red-Yellow-Green)
    - Visually scan for red cells (outliers)
    - Click URL in Column B to watch flagged videos
    - Common outliers to investigate:
      - `hook_scene_count > 10` (rapid cuts, encoding issues?)
      - `middle_*_word_count = 0` (speech detection failure?)
      - `closing_eye_contact_rate = 0` (face detection issue?)

---

**Red Flags**:
- ❌ No video_review.csv generated for any bucket → Stage 2 `url` field missing (check temporal_compute.py line 2655)
- ❌ All URLs are null/empty → Stage 2 modification not deployed
- ❌ Row count = 0 → All videos filtered out, check logs
- ❌ Column count ≠ aggregated + 1 → Feature extraction mismatch, bug in review_csv_generator.py
- ❌ URLs don't start with `https://` → Data corruption or wrong field extracted

**Green Flags**:
- ✅ 3 video_review.csv files (one per winning bucket)
- ✅ Row counts within 90-100% of aggregated_features.csv
- ✅ All URLs clickable and load TikTok videos
- ✅ Feature values match aggregated CSV (spot-checked)
- ✅ File sizes reasonable (~1KB per video)
- ✅ Logs show "✓ Review CSV: validation/video_review.csv" for each bucket

---

## Quick Test Command (All Stages)

```bash
# Run this after pipeline completes to validate Stage 3.4 output
cd data/clients/{client}/hashtags/{target}/{mode}_{strategy}/

echo "=== STAGE 3.4 VALIDATION ==="
echo "1. Checking video_review.csv files exist..."
find buckets/ -name "video_review.csv" -exec ls -lh {} \;

echo ""
echo "2. Row count comparison (review vs aggregated)..."
for bucket in buckets/bucket_*/; do
    review_rows=$(wc -l < "$bucket/validation/video_review.csv" 2>/dev/null || echo "0")
    agg_rows=$(wc -l < "$bucket/ml_analysis/aggregated_features.csv" 2>/dev/null || echo "0")
    echo "  Bucket $(basename $bucket): review=$review_rows, aggregated=$agg_rows"
done

echo ""
echo "3. Checking URL format (first URL from each bucket)..."
for bucket in buckets/bucket_*/; do
    url=$(awk -F',' 'NR==2 {print $2}' "$bucket/validation/video_review.csv" 2>/dev/null)
    echo "  Bucket $(basename $bucket): $url"
done

echo ""
echo "4. Checking Stage 3.4 logs..."
grep "Stage 3.4\|Review CSV" logs/rumiai_ml_*.log | tail -10

echo ""
echo "=== VALIDATION COMPLETE ==="
```

---

## Common Issues & Solutions

### Issue 1: "⚠️ Review CSV not generated (all videos missing url)"

**Cause**: All videos in bucket have null `metadata.url`

**Solution**:
1. Check temporal_compute.py line 2655: `'url': metadata.get('url', '')`
2. Check unified_analysis.json has `metadata.url` field
3. Verify Apify scraper returning `webVideoUrl`
4. If intentional (testing without URLs): Ignore warning, pipeline continues

**Impact**: Cannot investigate outliers in Excel (no clickable links)

---

### Issue 2: Review CSV has fewer rows than expected

**Cause**: Some videos missing `url` field (filtered out)

**Diagnosis**:
```bash
# Check how many videos were excluded
grep "excluded from video_review.csv" logs/rumiai_ml_*.log
```

**Solution**:
- If < 10% excluded: Normal (some videos may legitimately lack URLs)
- If > 10% excluded: Investigate Stage 2 url extraction

**Impact**: Minor - Most videos still available for review

---

### Issue 3: URLs return 404 on TikTok

**Cause**: Video deleted/removed from TikTok after scraping

**Solution**:
- Normal for some videos (TikTok content ephemeral)
- If > 50% return 404: Scraping data may be stale, re-scrape

**Impact**: Cannot watch specific videos, but feature data still valid for ML

---

### Issue 4: Column count mismatch

**Cause**: Bug in feature extraction logic

**Diagnosis**:
```bash
# Compare column counts
head -1 buckets/bucket_18-33s/validation/video_review.csv | tr ',' '\n' | wc -l
head -1 buckets/bucket_18-33s/ml_analysis/aggregated_features.csv | tr ',' '\n' | wc -l

# Difference should be exactly 1 (the url column)
```

**Solution**:
- File bug report with column count details
- Check review_csv_generator.py for recent changes

**Impact**: High - Indicates feature extraction bug

---

## Integration with Main Pipeline

**Stage 3.4 runs automatically** after Stage 3 (Feature Aggregation) completes:

```python
# rumiai_ml_batch.py integration (lines 549-579)
try:
    generate_review_csv_for_bucket(bucket_path)
    logger.info(f"Bucket {bucket_name}: Generated video_review.csv")
    print(f"  ✓ Review CSV: validation/video_review.csv")
except ValueError as e:
    logger.warning(f"Bucket {bucket_name}: {e}")
    print(f"  ⚠️  Review CSV not generated (all videos missing url)")
    # Continue pipeline - review CSV is optional
```

**Key Points**:
- Non-fatal errors: Pipeline continues even if Stage 3.4 fails
- Per-bucket execution: Each bucket gets its own video_review.csv
- No checkpointing: Lightweight operation, re-runs with Stage 3 if needed

---

## Reference

- **Implementation**: `/ml_pipeline/stage3_aggregation/review_csv_generator.py`
- **Integration**: `/rumiai_ml_batch.py` lines 549-579
- **Tests**: `/rumiai_v2/ml_pipeline/stage3_aggregation/test_*.py` (14 tests, 100% pass)
- **HLD**: `/documentation_migration/FutureDevelopments/ChildDocs/ReviewCSVGenerationCHILD.md`
- **Test Documentation**: `/documentation_migration/FutureDevelopments/ChildDocs/Stage3.4TestFiles.md`
