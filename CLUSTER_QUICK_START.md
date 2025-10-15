# Hashtag Cluster System - Quick Start Guide

## What is Hashtag Clustering?

The **Narrow Semantic Clustering Strategy** scrapes multiple related hashtags to solve the 57% reduction in video volume caused by US geographic filtering. This provides **2-3x more unique videos** with rich analytics for optimization.

---

## Quick Start (3 Steps)

### Step 1: Create a Cluster Configuration

```bash
python generate_cluster.py
```

**Follow the prompts:**
1. Enter cluster ID (e.g., `nutrition`)
2. Enter description
3. Enter primary hashtag (e.g., `#nutrition`)
4. Enter variant hashtags (e.g., `#nutritionist`, `#nutritiontips`, `#nutritioncoach`)
5. Configure scraping parameters (or use defaults)

**Result:** Configuration saved to `/config/hashtag_clusters/nutrition.json`

---

### Step 2: Run the ML Pipeline

```bash
python rumiai_ml_batch.py \
  --client acme_corp \
  --target nutrition \
  --analysis-type hashtag
```

**Note:** Target is cluster ID (`nutrition`), **NOT** hashtag (`#nutrition`)

**What happens:**
1. System detects cluster mode (no # prefix)
2. Loads cluster config from `/config/hashtag_clusters/nutrition.json`
3. Scrapes all hashtags × runs (e.g., 4 hashtags × 2 runs = 8 scrapes)
4. Deduplicates with provenance tracking
5. Generates cluster analytics
6. Continues to winner analysis and bucket selection

---

### Step 3: Review Cluster Analytics

```bash
cat data/clients/acme_corp/hashtag/nutrition/cluster_analytics.json | jq
```

**Key Metrics:**
- `scrape_summary.total_unique_videos` - Total unique videos after deduplication
- `scrape_summary.overall_duplication_rate` - Percentage overlap between hashtags
- `per_hashtag_contribution` - Which hashtags contributed most videos
- `pairwise_overlaps` - Overlap percentage between hashtag pairs
- `run_effectiveness` - How many new videos did run 2 add?

---

## Example Cluster Configuration

```json
{
  "cluster_id": "nutrition",
  "description": "Nutrition niche - narrow semantic cluster",
  "primary_hashtag": "#nutrition",
  "variant_hashtags": [
    "#nutritionist",
    "#nutritiontips",
    "#nutritioncoach"
  ],
  "scrape_config": {
    "runs_per_hashtag": 2,
    "delay_between_runs_ms": 120000,
    "results_per_page": 800
  }
}
```

**This configuration scrapes:**
- 4 hashtags (#nutrition + 3 variants)
- 2 runs per hashtag
- 800 videos per scrape
- **Total:** 8 scrapes, ~1,900 videos before deduplication → ~1,400 unique videos

---

## CLI Parameters

### Required Parameters

```bash
--client CLIENT_ID          # Your client identifier
--target CLUSTER_ID         # Cluster name (NOT hashtag with #)
--analysis-type hashtag     # Must be "hashtag" for cluster mode
```

### Optional Parameters (with defaults)

```bash
--analysis-mode top         # "top" or "recent" (default: top)
--selection-strategy contrastive  # "contrastive" or "top" (default: contrastive)
--video-count 100           # Videos per bucket (default: 100)
--date-filter last_90_days  # Date range (default: last_90_days)
--country-code US           # Geographic filter (default: US)
--auto-confirm              # Skip confirmation prompts
```

---

## Example Workflows

### Workflow 1: Nutrition Niche Analysis

```bash
# 1. Create cluster
python generate_cluster.py
# Enter: nutrition, #nutrition, #nutritionist, #nutritiontips, #nutritioncoach

# 2. Run pipeline
python rumiai_ml_batch.py \
  --client acme_corp \
  --target nutrition \
  --analysis-type hashtag \
  --analysis-mode top \
  --video-count 100 \
  --date-filter last_90_days

# 3. Check analytics
cat data/clients/acme_corp/hashtag/nutrition/cluster_analytics.json | jq '.scrape_summary'
```

---

### Workflow 2: Fitness Niche (Quick Test)

```bash
# 1. Create cluster with minimal config
python generate_cluster.py
# Enter: fitness, #fitness, #fitnesstips
# Runs: 1 (instead of default 2)
# Results: 400 (instead of default 800)

# 2. Run pipeline
python rumiai_ml_batch.py \
  --client test_client \
  --target fitness \
  --analysis-type hashtag

# 3. Review results quickly
cat data/clients/test_client/hashtag/fitness/cluster_analytics.json | jq '.per_hashtag_contribution'
```

---

## Understanding Cluster Analytics

### Scrape Summary

```json
{
  "total_scrapes_attempted": 8,
  "total_scrapes_succeeded": 8,
  "total_scraped_videos": 1939,
  "total_unique_videos": 1400,
  "overall_duplication_rate": 27.8
}
```

**Interpretation:**
- ✅ All scrapes successful
- 1,939 videos scraped → 1,400 unique (27.8% overlap)
- **This is good!** 20-30% overlap means hashtags are semantically related but not redundant

---

### Per-Hashtag Contribution

```json
{
  "#nutrition": {
    "total_found": 380,
    "exclusive_videos": 260,
    "contribution_percentage": 27.1
  }
}
```

**Interpretation:**
- Primary hashtag contributed 380 videos (27.1% of total)
- 260 videos were exclusive to #nutrition (not found by other hashtags)
- **Action:** If contribution is <10%, consider removing this hashtag

---

### Pairwise Overlaps

```json
{
  "nutrition_nutritionist": 18.2,
  "nutrition_nutritiontips": 25.4
}
```

**Interpretation:**
- #nutrition and #nutritionist overlap by 18.2%
- #nutrition and #nutritiontips overlap by 25.4%
- **Ideal:** 15-30% overlap (semantically related, not redundant)
- **Too high (>50%):** Consider removing redundant hashtag
- **Too low (<5%):** Hashtags may be too different

---

### Run Effectiveness

```json
{
  "#nutrition": {
    "run_1_videos": 250,
    "run_2_new_videos": 130,
    "run_2_new_percentage": 52.0
  }
}
```

**Interpretation:**
- Run 2 added 130 NEW videos (52% of run 2 results)
- **Good:** >40% new videos → worth doing 2 runs
- **Poor:** <20% new videos → consider reducing to 1 run per hashtag

---

## Troubleshooting

### Error: "Cluster config not found"

**Cause:** Cluster configuration file doesn't exist

**Solution:**
```bash
# Create cluster config first
python generate_cluster.py

# Then run pipeline
python rumiai_ml_batch.py --client YOUR_CLIENT --target YOUR_CLUSTER --analysis-type hashtag
```

---

### Error: "Single hashtag scraping is deprecated"

**Cause:** Using hashtag with # prefix as target

**Incorrect:**
```bash
python rumiai_ml_batch.py --target "#nutrition"  # ❌ Wrong!
```

**Correct:**
```bash
python rumiai_ml_batch.py --target nutrition     # ✅ Correct!
```

---

### Error: "All scrapes failed"

**Cause:** Network issues or invalid Apify API key

**Solution:**
1. Check Apify API key: `echo $APIFY_API_KEY`
2. Verify network connectivity
3. Check cluster config hashtags are valid
4. Review logs: `cat data/logs/rumiai_ml_*.log`

---

## Advanced Configuration

### Custom Scrape Parameters

```json
{
  "scrape_config": {
    "runs_per_hashtag": 3,           // 1-5 (default: 2)
    "delay_between_runs_ms": 180000, // 60000-600000 (default: 120000 = 2 min)
    "results_per_page": 600          // 100-800 (default: 800)
  }
}
```

**When to adjust:**
- **More runs (3-5):** Need maximum data volume
- **Longer delay (5-10 min):** Avoid rate limiting, allow TikTok feed to refresh
- **Fewer results (100-400):** Quick testing, lower cost

---

### Cluster Optimization Workflow

1. **Initial scrape** with default config (2 runs, 800 results)
2. **Review analytics** - identify underperforming hashtags
3. **Optimize cluster:**
   - Remove hashtags with <10% contribution
   - Remove hashtags with >50% overlap
   - Add new variants if needed
4. **Re-scrape** with optimized cluster
5. **Compare analytics** - verify improvement

---

## Best Practices

### ✅ DO

- Use 3-5 semantically related hashtags
- Start with 2 runs per hashtag (default)
- Review cluster analytics after each scrape
- Create separate clusters for different niches
- Use descriptive cluster IDs (e.g., `nutrition_wellness`, `fitness_weightloss`)

### ❌ DON'T

- Mix unrelated hashtags (e.g., #nutrition + #cars)
- Use >10 variant hashtags (diminishing returns)
- Use identical hashtags (e.g., #nutrition + #Nutrition)
- Skip cluster analytics review
- Use single hashtags anymore (deprecated!)

---

## File Locations

```
/home/jorge/rumiaifinal/
├── config/
│   └── hashtag_clusters/
│       ├── nutrition.json              # Your cluster configs
│       └── nutrition_example.json      # Example config
│
├── data/
│   └── clients/
│       └── {client_id}/
│           └── hashtag/
│               └── {cluster_id}/
│                   ├── cluster_analytics.json  # Cluster health metrics
│                   └── top_contrastive/
│                       ├── winner_analysis.json
│                       └── buckets/
│                           └── bucket_*/
│                               └── selected_videos.json
│
└── generate_cluster.py                 # Interactive cluster generator
```

---

## Next Steps

1. **Create your first cluster:** `python generate_cluster.py`
2. **Run the pipeline:** `python rumiai_ml_batch.py --client YOUR_CLIENT --target YOUR_CLUSTER --analysis-type hashtag`
3. **Review analytics:** `cat data/clients/YOUR_CLIENT/hashtag/YOUR_CLUSTER/cluster_analytics.json | jq`
4. **Optimize and iterate:** Use analytics to improve cluster composition

---

## Support

- **Full Implementation Details:** See `HASHTAG_CLUSTER_IMPLEMENTATION.md`
- **Technical Specification:** See `HashtagVolumeV2_TI.md`
- **Business Context:** See `HashtagVolumeV2.md`

---

**Ready to start?** Run `python generate_cluster.py` to create your first cluster! 🚀
