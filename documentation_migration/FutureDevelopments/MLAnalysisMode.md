# Analysis Mode System

## Overview

### Business Problem
Different business questions require analyzing different video sets:
- **"What works?"** → Analyze top-performing content (highest engagement)
- **"What's happening now?"** → Analyze most recent content (current trends/strategy)

Without dual-mode support, users would need to run separate analyses or manually filter videos, reducing flexibility and insight quality.

### Solution
Implement `--analysis-mode` flag with two options:
- `top`: Sort by engagement, analyze highest-performing videos
- `recent`: Sort by publish date, analyze most recent videos

### Stakeholder Value
- **Tumi Labs**: Single system handles multiple analysis types (ML training, trend monitoring, strategy tracking)
- **Brands**: Understand both "what works historically" and "what's trending now"
- **Competitive Intelligence**: Track rival's best work AND detect strategy shifts

---

## The Two Modes

### Top Mode (`--analysis-mode top`)

**What it does**: Analyzes highest-engagement videos within date filter

**How it works**:
1. Apify scrapes videos with `sortBy: engagement`
2. Videos sorted by composite score: `views × share_boost_factor`
3. Takes top N videos by engagement
4. Analyzes patterns that correlate with success

**Use cases**:
- **Hashtag**: Train ML models on viral patterns ("what makes #nutrition content go viral?")
- **Competitor**: Benchmark rival's best-performing content ("what works for @rival_brand?")
- **Creator**: Understand creator's peak performance style (for coaching)

**Output insight**: "These creative patterns correlate with high engagement"

---

### Recent Mode (`--analysis-mode recent`)

**What it does**: Analyzes most recently published videos within date filter

**How it works**:
1. Apify scrapes videos with `sortBy: date`
2. Videos sorted by `createTime` (newest first)
3. Takes most recent N videos
4. Analyzes current content strategy

**Use cases**:
- **Hashtag**: Detect trend shifts ("are #nutrition creators posting more long-form now?")
- **Competitor**: Track strategy changes ("has @rival_brand changed their approach?")
- **Creator**: Understand natural production style (for vetting)

**Output insight**: "This is what's being produced right now"

---

## Default Modes per Analysis Type

| Analysis Type | Default Mode | Reasoning |
|---------------|--------------|-----------|
| **Hashtag** | `top` | ML training needs viral patterns, not just recent posts |
| **Competitor** | `top` | Benchmark rival's best work first, track trends optionally |
| **Creator** | `recent` | Vetting needs natural style, not cherry-picked best |

**CLI behavior**:
```bash
# If --analysis-mode not specified, uses defaults above
python rumiai_ml_batch.py --analysis-type hashtag --target "#nutrition"
# Automatically uses: --analysis-mode top

python rumiai_ml_batch.py --analysis-type creator --target "@affiliate"
# Automatically uses: --analysis-mode recent
```

---

## Apify Integration

### Top Mode - Engagement Sorting

**Apify Scraper Parameters**:
```json
{
  "hashtagsUrls": ["#nutrition"],
  "resultsPerPage": 300,
  "shouldDownloadVideos": true,
  "sortBy": "engagement",
  "sortOrder": "desc"
}
```

**Post-Processing**:
```python
def calculate_engagement_score(video):
    """
    Composite engagement score for ranking
    Higher weight on shares (viral indicator)
    """
    views = video.get('playCount', 0)
    likes = video.get('diggCount', 0)
    shares = video.get('shareCount', 0)
    comments = video.get('commentCount', 0)

    # Share boost: shares indicate viral potential
    share_boost = 1 + (shares / max(views, 1)) * 10

    return views * share_boost

# Sort by engagement score
videos = sorted(videos, key=calculate_engagement_score, reverse=True)
top_videos = videos[:video_count]
```

---

### Recent Mode - Date Sorting

**Apify Scraper Parameters**:
```json
{
  "hashtagsUrls": ["#nutrition"],
  "resultsPerPage": 300,
  "shouldDownloadVideos": true,
  "sortBy": "date",
  "sortOrder": "desc"
}
```

**Post-Processing**:
```python
# Sort by publish date (newest first)
videos = sorted(videos, key=lambda v: v['createTime'], reverse=True)
recent_videos = videos[:video_count]
```

---

## Engagement Score Calculation

### Overview

For `--analysis-mode top`, videos are ranked by **engagement score** - a composite metric that identifies viral potential beyond simple view counts.

### Formula

```python
def calculate_engagement_score(video):
    """
    Composite engagement score with share boost factor

    Returns higher scores for videos with strong viral indicators
    """
    views = video.get('playCount', 0)
    likes = video.get('diggCount', 0)
    comments = video.get('commentCount', 0)
    shares = video.get('shareCount', 0)

    # Calculate share rate (shares as % of views)
    share_rate = shares / max(views, 1)

    # Share boost factor: shares indicate viral potential
    # 10x multiplier means 1% share rate = 10% boost
    share_boost = 1 + (share_rate * 10)

    # Final score = views × share boost
    engagement_score = views * share_boost

    return engagement_score
```

### Rationale

**Why not just use views?**
- High views don't always mean replicable patterns
- Some videos get views from paid promotion, not organic virality
- Shares indicate content people actively want to spread

**Why weight shares heavily?**
- **Shares = Viral Indicator**: People only share content they believe others will engage with
- **Quality Signal**: Shares require more commitment than likes (social risk)
- **Amplification Factor**: Shared content reaches new audiences organically
- **TikTok Algorithm**: Share rate heavily influences "For You" page placement

**The 10x multiplier:**
- A video with 1% share rate (10,000 shares / 1M views) gets 10% boost
- A video with 2% share rate gets 20% boost
- Typical TikTok share rates: 0.5-2% (viral videos: 3-5%+)

### Examples

#### Example 1: High Views, Low Shares (Not Viral)
```python
video_a = {
    'playCount': 1000000,
    'shareCount': 2000
}

share_rate = 2000 / 1000000 = 0.002 (0.2%)
share_boost = 1 + (0.002 * 10) = 1.02
engagement_score = 1000000 * 1.02 = 1,020,000

# Minimal boost - views alone don't indicate viral pattern
```

#### Example 2: Medium Views, High Shares (Viral)
```python
video_b = {
    'playCount': 500000,
    'shareCount': 15000
}

share_rate = 15000 / 500000 = 0.03 (3%)
share_boost = 1 + (0.03 * 10) = 1.30
engagement_score = 500000 * 1.30 = 650,000

# 30% boost for high share rate
# Would rank LOWER than video_a despite better viral indicators
```

#### Example 3: Lower Views, Exceptional Shares (Highly Viral)
```python
video_c = {
    'playCount': 300000,
    'shareCount': 15000
}

share_rate = 15000 / 300000 = 0.05 (5%)
share_boost = 1 + (0.05 * 10) = 1.50
engagement_score = 300000 * 1.50 = 450,000

# 50% boost for exceptional share rate
# Still lower than video_a, but identifies strong viral pattern
```

### Sorting Logic

Videos are sorted by engagement score in descending order:

```python
def sort_by_engagement(videos):
    """Sort videos by composite engagement score"""
    for video in videos:
        video['engagement_score'] = calculate_engagement_score(video)

    return sorted(videos, key=lambda v: v['engagement_score'], reverse=True)
```

### Data Source

All metrics come from Apify TikTok scraper:

```json
{
  "playCount": 3200000,      // → views
  "diggCount": 346500,       // → likes
  "commentCount": 872,        // → comments
  "shareCount": 15500         // → shares
}
```

**Reliability**:
- ✅ Apify always provides these metrics (core TikTok data)
- ✅ If missing, video is skipped (not processed)
- ✅ Point-in-time snapshot (sufficient for pattern analysis)

### Alternative Considerations

**Simple Engagement Rate (Not Used)**:
```python
# Alternative approach (simpler but less effective)
engagement_rate = (likes + comments + shares) / views
```

**Why we don't use this**:
- Treats all interactions equally (shares should be weighted higher)
- Doesn't account for viral amplification effect
- Lower correlation with "replicable viral patterns"

**Likes + Comments Weighting (Future Enhancement)**:
```python
# Potential refinement
engagement_score = views * (
    1 +
    (shares / views * 10) +           # 10x weight
    (comments / views * 5) +          # 5x weight
    (likes / views * 1)               # 1x weight
)
```

Currently not implemented to keep formula simple and interpretable.

### Usage in ML Pipeline

```python
def select_top_videos_for_training(videos, per_bucket=40):
    """
    Select top-performing videos for ML training
    Used in hashtag and competitor analysis (top mode)
    """
    # Calculate engagement scores
    for video in videos:
        video['engagement_score'] = calculate_engagement_score(video)

    # Sort by engagement score
    sorted_videos = sorted(videos, key=lambda v: v['engagement_score'], reverse=True)

    # Bucket by duration
    bucketed = bucket_by_duration(sorted_videos)

    # Select top 40 + bottom 20 per bucket (contrastive analysis)
    selected = {}
    for bucket_name, bucket_videos in bucketed.items():
        selected[bucket_name] = {
            'top_40': bucket_videos[:40],
            'bottom_20': bucket_videos[-20:]
        }

    return selected
```

### Quality Filters

Before calculating engagement score, apply minimum thresholds:

```python
def filter_qualified_videos(videos):
    """Remove low-quality videos before engagement scoring"""
    qualified = []

    for video in videos:
        # Minimum sample size
        if video['playCount'] < 1000:
            continue

        # Minimum engagement threshold
        basic_engagement_rate = (
            video['diggCount'] +
            video['commentCount'] +
            video['shareCount']
        ) / video['playCount']

        if basic_engagement_rate < 0.02:  # 2% minimum
            continue

        qualified.append(video)

    return qualified
```

**Thresholds**:
- **Minimum 1,000 views**: Ensures statistical significance
- **Minimum 2% engagement**: Filters "dead" content (bots, low-quality)

### Validation & Testing

```python
def test_engagement_score_calculation():
    """Unit test for engagement score formula"""

    # Test 1: Zero shares = no boost
    video = {'playCount': 100000, 'shareCount': 0}
    score = calculate_engagement_score(video)
    assert score == 100000  # 100000 * 1.0

    # Test 2: 1% share rate = 10% boost
    video = {'playCount': 100000, 'shareCount': 1000}
    score = calculate_engagement_score(video)
    assert score == 110000  # 100000 * 1.10

    # Test 3: 5% share rate = 50% boost
    video = {'playCount': 100000, 'shareCount': 5000}
    score = calculate_engagement_score(video)
    assert score == 150000  # 100000 * 1.50

    # Test 4: Edge case - zero views
    video = {'playCount': 0, 'shareCount': 10}
    score = calculate_engagement_score(video)
    assert score == 0  # Handled gracefully
```

---

## How Mode Affects Each Analysis Type

### Hashtag Analysis

#### Top Mode (Default)
```bash
python rumiai_ml_batch.py \
  --analysis-type hashtag \
  --target "#nutrition" \
  --video-count 300 \
  --date-filter "last_90_days" \
  --analysis-mode top
```

**Process**:
1. Scrape 300 highest-engagement #nutrition videos from last 90 days
2. Bucket by duration (8 buckets)
3. Select top 40 + bottom 20 per bucket (contrastive analysis)
4. Train ML models on viral patterns
5. Generate reports: "What makes #nutrition content go viral?"

**Insight**: "15-18s videos with joy_ratio > 0.6 and fast cuts have 3x engagement"

---

#### Recent Mode (Optional)
```bash
--analysis-mode recent
```

**Process**: Same but analyzes most recent 300 posts (regardless of engagement)

**Insight**: "Trend shift detected: 60% of recent #nutrition posts are 60-90s storytelling (was 20% three months ago)"

**Use case**: Quarterly trend reports, detect market shifts

---

### Competitor Analysis

#### Top Mode (Default)
```bash
python rumiai_ml_batch.py \
  --analysis-type competitor \
  --target "@rival_brand" \
  --video-count 150 \
  --date-filter "last_90_days" \
  --analysis-mode top
```

**Process**:
1. Scrape 150 highest-engagement videos from @rival_brand
2. Bucket by duration
3. Process ALL videos (no top/bottom selection)
4. Analyze creative patterns in their best work
5. Generate report: "What works for @rival_brand?"

**Insight**: "@rival_brand's top videos average 0.75 energy_level and 42 words in hook"

---

#### Recent Mode
```bash
--analysis-mode recent
```

**Process**: Same but analyzes most recent 150 posts

**Insight**: "Strategy shift: @rival_brand recently increased 13-18s content from 30% to 65%"

**Use case**: Monthly competitor monitoring, detect strategy changes

---

### Creator Analysis

#### Recent Mode (Default)
```bash
python rumiai_ml_batch.py \
  --analysis-type creator \
  --target "@potential_affiliate" \
  --video-count 40 \
  --analysis-mode recent \
  --compare-to hashtag:nutrition
```

**Process**:
1. Scrape most recent 40 videos from @potential_affiliate
2. Bucket by duration
3. Calculate distribution (what they naturally produce)
4. Compare to client's hashtag patterns
5. Generate compatibility score + hiring recommendation

**Insight**: "Creator naturally produces 55% in 13-18s bucket (matches client's 45% viral distribution) - STRONG FIT"

---

#### Top Mode (Optional)
```bash
--analysis-mode top
```

**Process**: Same but analyzes top 40 videos by engagement

**Insight**: "Creator's best work: 60-90s storytelling (avg 250K views), but rarely produces this format (10% of content)"

**Use case**: Understanding creator's peak performance for coaching (less useful for vetting)

---

## Mode Comparison Examples

### Scenario: Competitor Strategy Shift Detection

**Client**: Acme Nutrition brand wants to track rival's strategy

**Approach**: Run both modes monthly

```bash
# Month 1: Baseline (Top Mode)
python rumiai_ml_batch.py \
  --analysis-type competitor \
  --target "@rival_brand" \
  --video-count 150 \
  --date-filter "last_90_days" \
  --analysis-mode top

# Output: Best work is 13-18s with high energy (0.75)
```

```bash
# Month 2: Track recent strategy (Recent Mode)
python rumiai_ml_batch.py \
  --analysis-type competitor \
  --target "@rival_brand" \
  --video-count 150 \
  --date-filter "last_90_days" \
  --analysis-mode recent

# Output: Recent posts shifted to 60-90s storytelling (strategy change detected)
```

**Action**: Acme adjusts their content strategy based on rival's pivot

---

### Scenario: Creator Vetting (Recent) vs Coaching (Top)

**Client**: Wants to vet @fitness_jane for affiliate program

**Step 1: Vetting (Recent Mode)**
```bash
python rumiai_ml_batch.py \
  --analysis-type creator \
  --target "@fitness_jane" \
  --video-count 40 \
  --analysis-mode recent \
  --compare-to hashtag:nutrition

# Output: Natural style = 55% in 13-18s bucket
# Compatibility score: 0.82 (STRONG FIT)
# Recommendation: Tier 1 - Immediate Hire
```

**Step 2: Coaching (Top Mode)** - After hiring
```bash
--analysis-mode top

# Output: Best performing videos = 0.68 joy_ratio, 15 text overlays
# Coach creator: "Your viral videos have higher joy_ratio than your average (0.35)"
```

---

## Implementation Design

### CLI Flag Handling

```python
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-type', required=True, choices=['hashtag', 'competitor', 'creator'])
    parser.add_argument('--analysis-mode', choices=['top', 'recent'], default=None)
    # ... other args

    args = parser.parse_args()

    # Apply defaults if not specified
    if args.analysis_mode is None:
        if args.analysis_type == 'creator':
            args.analysis_mode = 'recent'
        else:  # hashtag or competitor
            args.analysis_mode = 'top'

    return args
```

---

### Apify Scraper Integration

```python
class ApifyVideoScraper:
    def fetch_videos(self, target, video_count, date_filter, analysis_mode):
        """
        Fetch videos with sorting based on analysis mode
        """

        # Determine Apify sort parameter
        sort_by = 'engagement' if analysis_mode == 'top' else 'date'

        # Apify scraper configuration
        scraper_input = {
            'resultsPerPage': video_count,
            'shouldDownloadVideos': True,
            'sortBy': sort_by,
            'sortOrder': 'desc'
        }

        # Add target (hashtag or profile)
        if target.startswith('#'):
            scraper_input['hashtagsUrls'] = [f'https://www.tiktok.com/tag/{target[1:]}']
        else:  # @handle
            scraper_input['profilesUrls'] = [f'https://www.tiktok.com/{target}']

        # Run Apify scraper
        videos = self.run_scraper(scraper_input)

        # Post-process based on mode
        if analysis_mode == 'top':
            videos = self._sort_by_engagement(videos)
        else:  # recent
            videos = self._sort_by_date(videos)

        # Apply date filter
        videos = self._apply_date_filter(videos, date_filter)

        return videos[:video_count]

    def _sort_by_engagement(self, videos):
        """Calculate engagement score and sort"""
        for video in videos:
            video['engagement_score'] = self._calculate_engagement_score(video)
        return sorted(videos, key=lambda v: v['engagement_score'], reverse=True)

    def _sort_by_date(self, videos):
        """Sort by publish date (newest first)"""
        return sorted(videos, key=lambda v: v['createTime'], reverse=True)

    def _calculate_engagement_score(self, video):
        """Composite engagement score with share boost"""
        views = video.get('playCount', 0)
        shares = video.get('shareCount', 0)

        share_boost = 1 + (shares / max(views, 1)) * 10
        return views * share_boost
```

---

### Checkpoint Integration

**Checkpoint must store analysis mode** to validate resume:

```json
{
  "config": {
    "video_count": 300,
    "date_filter": "last_90_days",
    "analysis_mode": "top"  // ← Must match on resume
  }
}
```

**Resume validation**:
```python
def validate_checkpoint_config(checkpoint, new_config):
    """Ensure analysis mode matches when resuming"""
    if checkpoint['config']['analysis_mode'] != new_config['analysis_mode']:
        raise ValueError(
            f"Cannot resume: checkpoint used '{checkpoint['config']['analysis_mode']}' mode, "
            f"but you specified '{new_config['analysis_mode']}'. "
            "Use --force to restart with new mode."
        )
```

---

## Edge Cases & Handling

### Case 1: Not Enough High-Engagement Videos

**Scenario**: Requesting top 300 videos, but only 150 have >1K views

**Handling**: Take all available videos, log warning
```python
if len(high_engagement_videos) < video_count:
    logger.warning(f"Only {len(high_engagement_videos)} videos meet engagement threshold, using all available")
    return high_engagement_videos
```

---

### Case 2: Creator Has Deleted Recent Videos

**Scenario**: Recent mode requests 40 videos, but only 30 available

**Handling**: Process all available, adjust compatibility confidence
```json
{
  "analysis_type": "recent_40_videos",
  "total_videos_analyzed": 30,
  "confidence_penalty": "reduced_sample_size",
  "note": "Only 30 videos available (requested 40)"
}
```

---

### Case 3: Date Filter Eliminates All Videos

**Scenario**: `--date-filter last_7_days` but competitor hasn't posted in 10 days

**Handling**: Error with suggestion
```
✗ No videos found matching criteria:
  - Target: @rival_brand
  - Date filter: last_7_days
  - Analysis mode: top

Suggestions:
  1. Expand date filter (try --date-filter last_30_days)
  2. Check if account is still active
```

---

### Case 4: Engagement Ties

**Scenario**: Multiple videos with identical engagement scores

**Handling**: Use `createTime` as tiebreaker (newer first)
```python
videos = sorted(videos, key=lambda v: (v['engagement_score'], v['createTime']), reverse=True)
```

---

## Reporting Differences by Mode

### Top Mode Reports

**Focus**: "What works" (pattern → outcome)

**Example excerpt**:
```
## Hook Analysis (13-18s bucket)

Top-performing videos (avg 250K views) show:
- Joy ratio: 0.68 (vs bottom 20: 0.32) → +112% engagement
- Text overlays in first 3s: 85% (vs bottom: 45%) → +89% retention
- Energy level: 0.75 (vs bottom: 0.42) → +78% shares

Recommendation: Prioritize joyful energy and immediate text overlays in 13-18s content.
```

---

### Recent Mode Reports

**Focus**: "What's happening" (trend → strategy)

**Example excerpt**:
```
## Recent Content Strategy Shift (Last 30 days)

Duration distribution change:
- 13-18s: 65% (was 45% three months ago) → +44% increase
- 60-90s: 15% (was 35% three months ago) → -57% decrease

Creative pattern changes:
- Average energy level: 0.82 (was 0.65) → More dynamic editing
- Text overlay density: +40% increase → Heavier use of captions

Interpretation: Market shifting toward short-form, high-energy content with heavy text overlays.
```

---

## Testing Strategy

### Unit Tests

```python
def test_analysis_mode_defaults():
    """Test correct defaults per analysis type"""
    args_hashtag = parse_args(['--analysis-type', 'hashtag', '--target', '#test'])
    assert args_hashtag.analysis_mode == 'top'

    args_creator = parse_args(['--analysis-type', 'creator', '--target', '@test'])
    assert args_creator.analysis_mode == 'recent'

def test_engagement_score_calculation():
    """Test composite engagement scoring"""
    video = {'playCount': 100000, 'shareCount': 500}
    score = calculate_engagement_score(video)
    # 500 shares / 100k views = 0.005 share rate
    # boost = 1 + (0.005 * 10) = 1.05
    # score = 100000 * 1.05 = 105000
    assert score == 105000

def test_date_sorting():
    """Test recent mode sorts by date correctly"""
    videos = [
        {'createTime': '2025-01-20', 'id': '1'},
        {'createTime': '2025-01-25', 'id': '2'},
        {'createTime': '2025-01-15', 'id': '3'}
    ]
    sorted_videos = sort_by_date(videos)
    assert [v['id'] for v in sorted_videos] == ['2', '1', '3']
```

---

## Future Enhancements

### Mixed Mode Analysis
Compare top vs recent in single run:
```bash
--analysis-mode both
```
Output: Side-by-side comparison showing strategy shifts

---

### Custom Sorting
Allow custom engagement formulas:
```bash
--analysis-mode top --engagement-formula "views * 0.5 + shares * 2 + comments * 1.5"
```

---

### Time-Weighted Recent
Exponentially decay older videos in recent mode:
```bash
--analysis-mode recent --time-decay 0.95
```

---

## Summary

### Key Decisions
- ✅ **Two modes**: `top` (engagement) and `recent` (date)
- ✅ **Smart defaults**: hashtag/competitor use `top`, creator uses `recent`
- ✅ **Apify integration**: `sortBy` parameter controls video ordering
- ✅ **Flexible use**: Same infrastructure, different business questions

### Success Metrics
- **Flexibility**: One command switch changes analysis purpose
- **Intelligence**: Different insights based on mode (patterns vs trends)
- **Business value**: Answers "what works?" AND "what's happening now?"