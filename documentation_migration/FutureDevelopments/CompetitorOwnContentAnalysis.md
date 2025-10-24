# Competitor Own Content Analysis Tool

**Quick Summary**: Standalone Python script that filters competitor creative intelligence to show ONLY their original content (excluding reposts/affiliate content). Provides cleaner A/B testing insights by analyzing what competitors are actually creating vs what they're reposting from their network. **For internal analysis only - not client-facing.**

---

## Table of Contents

- [What This Is](#what-this-is)
- [Why This Exists](#why-this-exists)
- [Key Features](#key-features)
- [Usage](#usage)
- [Implementation](#implementation)
- [Output Format](#output-format)
- [Technical Details](#technical-details)

---

## What This Is

**`analyze_original_content.py`** - A standalone analysis script that:

1. **Reads existing pipeline outputs** (Stage 2.7 Content Analysis, Stage 7 Formulas, unified_analysis metadata)
2. **Identifies repost videos** using @mention extraction and repost indicators
3. **Filters creative patterns** to show only original content
4. **Outputs filtered analysis** for internal competitive intelligence

**NOT part of production code** - This is a "frankenstein bandaid" tool that sits on top of existing outputs without modifying any production code.

---

## Why This Exists

### Problem:
- Competitor reports currently analyze ALL content (original + reposts)
- Reposted/affiliate content doesn't reflect competitor's actual creative testing
- Hard to see what they're actively producing vs sourcing from their network

### Solution:
- Filter to show ONLY original content
- Reveals true creative strategy and A/B testing focus
- Shows resource allocation (what they create vs repost)

### Value Proposition:
```
Example Insight:

ALL Content Analysis:
- Recipe Tutorial: 38% of content
- Expert Interview: 18% of content

ORIGINAL Content Only:
- Recipe Tutorial: 28% of content (↓ 26%) → They REPOST most recipes
- Expert Interview: 32% of content (↑ 78%) → They CREATE expert content

Strategic Intelligence:
Competitor builds authority with original expert interviews,
leverages affiliate network for recipe volume. They invest creative
resources in credibility, not tutorials.
```

**This reveals**:
- What competitors are testing (real A/B tests)
- Where they invest creative resources
- Content sourcing strategy (create vs repost)

---

## Key Features

### ✅ **Non-Invasive**
- Reads existing pipeline outputs
- Zero modifications to production code
- Standalone script you can run ad-hoc

### ✅ **Fast & Flexible**
- ~2-3 hours to build
- Easy to modify/iterate
- Multiple output formats (terminal, JSON, CSV)

### ✅ **Comprehensive Filtering**
Shows original-only versions of:
- Creative formulas (Stage 7)
- Content categories
- Hook strategies
- Pain points
- Keywords
- Engagement drivers

---

## Usage

### Basic Command

```bash
# Analyze single competitor
python analyze_original_content.py --competitor @wellness_pro --hashtag nutrition

# Save results to JSON file
python analyze_original_content.py --competitor @rival_brand --output results.json

# Specify client ID
python analyze_original_content.py --competitor @fitness_guru --client-id client_xyz --hashtag fitness
```

### Parameters

| Parameter | Required | Description | Example |
|-----------|----------|-------------|---------|
| `--competitor` | Yes | Competitor handle | `@wellness_pro` |
| `--hashtag` | No | Hashtag analyzed (defaults to competitor handle) | `nutrition` |
| `--client-id` | No | Client ID (default: `acme_corp`) | `client_xyz` |
| `--output` | No | Save results to JSON file | `results.json` |

### Prerequisites

**Must have run normal pipeline for competitor first**:
1. Stage 2 (Whisper + Apify scraping) → Creates `unified_analysis/{video_id}.json`
2. Stage 2.5 (Video selection) → Creates `selection_manifest.json`
3. Stage 2.7 (Content Analysis) → Creates `bucket_*/content_analysis/{video_id}_content.json`
4. Stage 7 (Formula generation) → Creates `winning_formulas.json`

Then run this script to get filtered original-only analysis.

---

## Implementation

### Script Location

```
/home/jorge/rumiaifinal/scripts/analyze_original_content.py
```

### Complete Script

```python
#!/usr/bin/env python3
"""
analyze_original_content.py

Standalone script to analyze competitor's ORIGINAL content only (excluding reposts).
Reads existing pipeline outputs, filters out reposted content, shows creative patterns.

Usage:
    python analyze_original_content.py --competitor @wellness_pro --hashtag nutrition
    python analyze_original_content.py --competitor @rival_brand --output results.json
"""

import json
import re
from collections import Counter
from pathlib import Path
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_PATH = "/data/clients/{client_id}/hashtags/{hashtag}/top_contrastive"
UNIFIED_ANALYSIS_PATH = "/home/jorge/rumiaifinal/unified_analysis"

REPOST_INDICATORS = ['repost', 'via', 'credit', 'by', 'from']

# ============================================================================
# STEP 1: IDENTIFY REPOST VIDEOS
# ============================================================================

def identify_repost_videos(manifest_path, unified_path):
    """
    Identify which videos are reposts based on @mentions and repost indicators.

    Args:
        manifest_path: Path to selection_manifest.json
        unified_path: Path to unified_analysis directory

    Returns:
        dict: {
            'repost_ids': [...],
            'original_ids': [...],
            'repost_rate': float,
            'total_videos': int,
            'repost_details': [{video_id, reason}, ...]
        }
    """
    print("🔍 Analyzing captions to identify reposts...")

    # Load manifest to get video IDs
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_video_ids = []
    for bucket, videos in manifest['videos_by_bucket'].items():
        all_video_ids.extend(videos.get('top_performers', []))
        all_video_ids.extend(videos.get('bottom_performers', []))

    repost_ids = []
    original_ids = []
    repost_details = []

    for video_id in all_video_ids:
        unified_file = Path(unified_path) / f"{video_id}.json"

        if not unified_file.exists():
            print(f"⚠️  Warning: {video_id}.json not found in unified_analysis")
            continue

        with open(unified_file) as f:
            data = json.load(f)

        caption = data.get('metadata', {}).get('description', '')

        # Check for @mentions
        mentions = re.findall(r'@(\w+)', caption)

        # Check for repost indicators
        repost_words = [ind for ind in REPOST_INDICATORS if ind in caption.lower()]

        if mentions or repost_words:
            repost_ids.append(video_id)
            repost_details.append({
                'video_id': video_id,
                'mentions': mentions,
                'repost_indicators': repost_words,
                'caption_preview': caption[:100]
            })
        else:
            original_ids.append(video_id)

    repost_rate = (len(repost_ids) / len(all_video_ids) * 100) if all_video_ids else 0

    print(f"✅ Found {len(original_ids)} original videos, {len(repost_ids)} reposts")
    print(f"   Repost rate: {repost_rate:.1f}%\n")

    return {
        'repost_ids': repost_ids,
        'original_ids': original_ids,
        'repost_rate': repost_rate,
        'total_videos': len(all_video_ids),
        'repost_details': repost_details
    }

# ============================================================================
# STEP 2: FILTER CREATIVE FORMULAS
# ============================================================================

def filter_formulas(formulas_path, original_ids):
    """
    Filter Stage 7 winning formulas to only include original content.

    Args:
        formulas_path: Path to winning_formulas.json
        original_ids: List of original video IDs

    Returns:
        list: Filtered formulas (original content only)
    """
    print("🎨 Filtering creative formulas (original content only)...")

    # Load Stage 7 formulas
    with open(formulas_path) as f:
        formulas = json.load(f)

    # Filter to original content only
    # Note: This assumes formulas have 'video_id' or 'representative_video_id' field
    # Adjust based on actual Stage 7 output structure
    original_formulas = []

    for formula in formulas.get('winning_formulas', []):
        # Check if formula references original videos
        # (Implementation depends on Stage 7 output structure)
        formula_videos = formula.get('video_ids', [formula.get('video_id')])

        if any(vid in original_ids for vid in formula_videos):
            original_formulas.append(formula)

    print(f"✅ {len(original_formulas)} formulas from original content\n")

    return original_formulas

# ============================================================================
# STEP 3: AGGREGATE CONTENT ANALYSIS (ORIGINAL ONLY)
# ============================================================================

def aggregate_content_analysis(base_path, original_ids):
    """
    Aggregate Stage 2.7 content analysis for original videos only.

    Args:
        base_path: Base path to pipeline outputs
        original_ids: List of original video IDs

    Returns:
        dict: Aggregated content analysis (original only)
    """
    print("📊 Aggregating content analysis (original content only)...")

    content_categories = Counter()
    hook_strategies = Counter()
    pain_points = Counter()
    keywords = Counter()
    engagement_drivers = Counter()

    buckets_path = Path(base_path) / "buckets"

    # Search all bucket subdirectories for content_analysis files
    for bucket_dir in buckets_path.glob("bucket_*/content_analysis"):
        for video_id in original_ids:
            content_file = bucket_dir / f"{video_id}_content.json"

            if not content_file.exists():
                continue

            with open(content_file) as f:
                data = json.load(f)

            # Aggregate core fields
            content_categories[data.get('content_category')] += 1
            hook_strategies[data.get('hook_strategy')] += 1

            # Aggregate arrays
            for pp in data.get('pain_points', []):
                pain_points[pp] += 1

            for kw in data.get('keywords', []):
                keywords[kw] += 1

            for ed in data.get('engagement_drivers', []):
                engagement_drivers[ed] += 1

    total = len(original_ids)

    print(f"✅ Aggregated {total} original videos\n")

    return {
        'total_videos': total,
        'content_categories': [
            {'name': cat, 'count': count, 'percentage': round(count/total*100, 1)}
            for cat, count in content_categories.most_common(5)
        ],
        'hook_strategies': [
            {'name': hook, 'count': count, 'percentage': round(count/total*100, 1)}
            for hook, count in hook_strategies.most_common(4)
        ],
        'pain_points': [
            {'name': pp, 'count': count, 'percentage': round(count/total*100, 1)}
            for pp, count in pain_points.most_common(5)
        ],
        'keywords': [kw for kw, _ in keywords.most_common(10)],
        'engagement_drivers': [
            {'name': ed, 'count': count, 'percentage': round(count/total*100, 1)}
            for ed, count in engagement_drivers.most_common(5)
        ]
    }

# ============================================================================
# STEP 4: PRINT RESULTS
# ============================================================================

def print_results(competitor, repost_analysis, formulas, content_analysis):
    """
    Print nicely formatted terminal output for analysis.
    """
    print("\n" + "="*70)
    print(f"ORIGINAL CONTENT CREATIVE INTELLIGENCE - {competitor}")
    print("="*70 + "\n")

    # Repost stats
    print(f"📊 CONTENT SOURCING:")
    print(f"   Original: {100-repost_analysis['repost_rate']:.1f}% ({len(repost_analysis['original_ids'])} videos)")
    print(f"   Reposted: {repost_analysis['repost_rate']:.1f}% ({len(repost_analysis['repost_ids'])} videos)")
    print()

    # Top formulas
    if formulas:
        print("🎨 TOP CREATIVE FORMULAS (ORIGINAL ONLY):")
        for i, formula in enumerate(formulas[:5], 1):
            print(f"   {i}. {formula.get('pattern_name', 'Unnamed')}")
            print(f"      Bucket: {formula.get('bucket_range', 'Unknown')}")
            # print(f"      Video ID: {formula.get('video_id', 'N/A')}")
        print()

    # Content categories
    if content_analysis['content_categories']:
        print("📁 CONTENT CATEGORIES:")
        for cat in content_analysis['content_categories']:
            print(f"   {cat['name']:<30} {cat['percentage']:>5.1f}% ({cat['count']} videos)")
        print()

    # Hook strategies
    if content_analysis['hook_strategies']:
        print("🎣 HOOK STRATEGIES:")
        for hook in content_analysis['hook_strategies']:
            print(f"   {hook['name']:<30} {hook['percentage']:>5.1f}% ({hook['count']} videos)")
        print()

    # Pain points
    if content_analysis['pain_points']:
        print("💢 PAIN POINTS ADDRESSED:")
        for pp in content_analysis['pain_points']:
            print(f"   {pp['name']:<30} {pp['percentage']:>5.1f}% ({pp['count']} videos)")
        print()

    # Keywords
    if content_analysis['keywords']:
        print("🔑 TOP KEYWORDS:")
        print(f"   {', '.join(content_analysis['keywords'])}")
        print()

    # Engagement drivers
    if content_analysis['engagement_drivers']:
        print("⚡ ENGAGEMENT DRIVERS:")
        for ed in content_analysis['engagement_drivers']:
            print(f"   {ed['name']:<30} {ed['percentage']:>5.1f}% ({ed['count']} videos)")
        print()

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze competitor original content (excluding reposts/affiliates)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_original_content.py --competitor @wellness_pro --hashtag nutrition
  python analyze_original_content.py --competitor @rival_brand --output results.json
  python analyze_original_content.py --competitor @fitness_guru --client-id xyz
        """
    )

    parser.add_argument('--competitor', required=True, help='Competitor handle (e.g., @wellness_pro)')
    parser.add_argument('--hashtag', help='Hashtag analyzed (defaults to competitor handle)')
    parser.add_argument('--client-id', default='acme_corp', help='Client ID (default: acme_corp)')
    parser.add_argument('--output', help='Save results to JSON file')
    parser.add_argument('--verbose', action='store_true', help='Show detailed repost information')

    args = parser.parse_args()

    # Build paths
    hashtag = args.hashtag or args.competitor.replace('@', '')
    base = BASE_PATH.format(client_id=args.client_id, hashtag=hashtag)
    manifest_path = f"{base}/selection_manifest.json"
    formulas_path = f"{base}/winning_formulas.json"

    print(f"\n🔎 Analyzing original content for {args.competitor}\n")
    print(f"📁 Base path: {base}\n")

    # Step 1: Identify reposts
    try:
        repost_analysis = identify_repost_videos(manifest_path, UNIFIED_ANALYSIS_PATH)
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find {manifest_path}")
        print(f"   Make sure pipeline has been run for this competitor first.")
        return

    # Step 2: Filter formulas (optional - depends on Stage 7 output)
    formulas = []
    try:
        formulas = filter_formulas(formulas_path, repost_analysis['original_ids'])
    except FileNotFoundError:
        print(f"⚠️  Warning: {formulas_path} not found. Skipping formula analysis.")
    except Exception as e:
        print(f"⚠️  Warning: Could not filter formulas: {e}")

    # Step 3: Aggregate content analysis
    content_analysis = aggregate_content_analysis(base, repost_analysis['original_ids'])

    # Step 4: Print results
    print_results(args.competitor, repost_analysis, formulas, content_analysis)

    # Optional: Show detailed repost info
    if args.verbose and repost_analysis['repost_details']:
        print("\n" + "="*70)
        print("REPOST DETAILS")
        print("="*70 + "\n")
        for detail in repost_analysis['repost_details'][:10]:  # Show first 10
            print(f"Video: {detail['video_id']}")
            if detail['mentions']:
                print(f"  @Mentions: {', '.join(detail['mentions'])}")
            if detail['repost_indicators']:
                print(f"  Indicators: {', '.join(detail['repost_indicators'])}")
            print(f"  Caption: {detail['caption_preview']}...")
            print()

    # Optional: Save to file
    if args.output:
        results = {
            'competitor': args.competitor,
            'repost_analysis': {
                'repost_rate': repost_analysis['repost_rate'],
                'original_count': len(repost_analysis['original_ids']),
                'repost_count': len(repost_analysis['repost_ids']),
                'total_videos': repost_analysis['total_videos']
            },
            'formulas': formulas,
            'content_analysis': content_analysis
        }

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Saved results to {args.output}")

if __name__ == '__main__':
    main()
```

---

## Output Format

### Terminal Output

```
🔎 Analyzing original content for @wellness_pro

📁 Base path: /data/clients/acme_corp/hashtags/nutrition/top_contrastive

🔍 Analyzing captions to identify reposts...
✅ Found 174 original videos, 126 reposts
   Repost rate: 42.0%

🎨 Filtering creative formulas (original content only)...
✅ 12 formulas from original content

📊 Aggregating content analysis (original content only)...
✅ Aggregated 174 original videos

======================================================================
ORIGINAL CONTENT CREATIVE INTELLIGENCE - @wellness_pro
======================================================================

📊 CONTENT SOURCING:
   Original: 58.0% (174 videos)
   Reposted: 42.0% (126 videos)

🎨 TOP CREATIVE FORMULAS (ORIGINAL ONLY):
   1. The Expert Interview Format
      Bucket: 18-33s

   2. The Transformation Journey
      Bucket: 33-60s

   3. The Wellness Practice Tutorial
      Bucket: 18-33s

📁 CONTENT CATEGORIES:
   expert_interview                 32.0% (56 videos)
   recipe_tutorial                  28.0% (49 videos)
   wellness_practice                25.0% (44 videos)
   transformation_story             10.0% (17 videos)
   supplement_review                 5.0% (8 videos)

🎣 HOOK STRATEGIES:
   question_hook                    45.0% (78 videos)
   problem_solution                 35.0% (61 videos)
   direct_statement                 20.0% (35 videos)

💢 PAIN POINTS ADDRESSED:
   gut_health                       52.0% (90 videos)
   low_energy                       45.0% (78 videos)
   inflammation                     38.0% (66 videos)
   weight_management                32.0% (56 videos)
   bloating                         28.0% (49 videos)

🔑 TOP KEYWORDS:
   #guthealth, #wellness, #holistic, #naturalhealing, #inflammation,
   #nutrition, #healthylifestyle, #protein, #cleaneating, #metabolism

⚡ ENGAGEMENT DRIVERS:
   personal_testimony               45.0% (78 videos)
   expert_credentials               42.0% (73 videos)
   before_after_reveal              38.0% (66 videos)
   specific_metrics                 35.0% (61 videos)
   scientific_evidence              28.0% (49 videos)
```

### JSON Output (--output results.json)

```json
{
  "competitor": "@wellness_pro",
  "repost_analysis": {
    "repost_rate": 42.0,
    "original_count": 174,
    "repost_count": 126,
    "total_videos": 300
  },
  "formulas": [
    {
      "pattern_name": "The Expert Interview Format",
      "bucket_range": "18-33s",
      "video_id": "7526250443832331550"
    }
  ],
  "content_analysis": {
    "total_videos": 174,
    "content_categories": [
      {"name": "expert_interview", "count": 56, "percentage": 32.0},
      {"name": "recipe_tutorial", "count": 49, "percentage": 28.0}
    ],
    "hook_strategies": [
      {"name": "question_hook", "count": 78, "percentage": 45.0}
    ],
    "pain_points": [
      {"name": "gut_health", "count": 90, "percentage": 52.0}
    ],
    "keywords": ["#guthealth", "#wellness", "#holistic"],
    "engagement_drivers": [
      {"name": "personal_testimony", "count": 78, "percentage": 45.0}
    ]
  }
}
```

---

## Technical Details

### How It Works

1. **Identifies Reposts** via two methods:
   - **@mention detection**: `re.findall(r'@(\w+)', caption)`
   - **Repost indicators**: Keywords ["repost", "via", "credit", "by", "from"] in caption

2. **Filters Video IDs** into two lists:
   - `original_ids` - Videos with no mentions or repost indicators
   - `repost_ids` - Videos with mentions or repost indicators

3. **Aggregates Data** for original videos only:
   - Filters Stage 7 formulas
   - Aggregates Stage 2.7 content analysis
   - Calculates percentages based on original content count

### Data Sources

| Data Type | Source File | Usage |
|-----------|-------------|-------|
| Video IDs | `selection_manifest.json` | Get all analyzed videos |
| Captions | `unified_analysis/{video_id}.json` → `metadata.description` | Detect reposts |
| Content Analysis | `buckets/bucket_*/content_analysis/{video_id}_content.json` | Aggregate patterns |
| Creative Formulas | `winning_formulas.json` | Filter formulas |

### Repost Detection Logic

```python
# A video is classified as "repost" if:
is_repost = (
    bool(re.findall(r'@(\w+)', caption))  # Has @mentions
    OR
    any(word in caption.lower() for word in ['repost', 'via', 'credit', 'by', 'from'])
)
```

**Rationale**:
- @mentions typically indicate affiliate/UGC content
- Repost keywords explicitly indicate sourced content
- Conservative approach: when in doubt, mark as repost

---

## Limitations

1. **False Positives**: May incorrectly classify original content as repost if:
   - Creator @mentions themselves in caption
   - Words like "by" or "from" appear in non-repost context

2. **False Negatives**: May miss reposts if:
   - No @mention or keyword in caption
   - Repost from sources that don't require credit

3. **Depends on Pipeline Completeness**:
   - Requires full pipeline run (Stages 2, 2.5, 2.7, 7)
   - Missing files will cause partial results

4. **Not Real-Time**: Analyzes completed pipeline runs, not live data

---

## Future Enhancements

### Potential Improvements:
- [ ] Add engagement comparison (original vs repost performance)
- [ ] Show bucket-level repost distribution
- [ ] Compare original content % across multiple competitors
- [ ] Add CSV export option
- [ ] Visualize results (charts/graphs)
- [ ] Track repost rate trends over time

### Integration Options:
- Could be integrated into production reports later if valuable
- Could add to Template 3/4 as optional section
- Could create automated weekly analysis reports

---

## Related Documentation

- **Stage8MVP.md Section 0.5.4** - @Mention Extraction function specification
- **Stage8MVP_Reports.md lines 1848-1951** - Content Sourcing Strategy section (client-facing version)
- **Stage 2.7 Content Analysis** - Source of content classifications
- **Stage 7 Formula Generation** - Source of creative formulas

---

**Status**: Implementation ready. Script can be built in ~2-3 hours.

**Maintenance**: Standalone tool - no production code dependencies.
