#!/usr/bin/env python3
"""
extract_competitor_data.py - Report 3: Single Competitor → Client

Generates deep dive competitive intelligence on 1 competitor.

Usage:
    python extract_competitor_data.py --client acme --competitor drinkpoppi --mode top --strategy contrastive
"""

import argparse
import json
import os
import re
import pandas as pd
from collections import Counter
import qrcode


# =============================
# HELPER FUNCTIONS
# =============================

def calculate_engagement_metrics(video_metadata):
    """
    Calculate engagement rate from TikTok video metadata.

    Formula: (likes + comments + shares + saves) / views × 100

    Input fields (from selected_videos.json):
    - diggCount (likes)
    - commentCount
    - shareCount
    - collectCount (saves/bookmarks)
    - playCount (views)

    Returns: Float (percentage, e.g., 1.2 = 1.2%)
    """

    likes = video_metadata.get('diggCount', 0)
    comments = video_metadata.get('commentCount', 0)
    shares = video_metadata.get('shareCount', 0)
    saves = video_metadata.get('collectCount', 0)
    views = video_metadata.get('playCount', 1)  # Avoid division by zero

    total_interactions = likes + comments + shares + saves
    engagement_rate = (total_interactions / views) * 100

    return round(engagement_rate, 1)  # Round to 1 decimal place


def calculate_bucket_distribution(winner_analysis_path):
    """
    Calculate percentage distribution across all 8 duration buckets.

    Args:
        winner_analysis_path: Path to winner_analysis.json file

    Returns:
        dict: Bucket name → percentage (rounded to integer)
    """

    with open(winner_analysis_path) as f:
        data = json.load(f)

    # Try bucket_distribution first, fallback to top_100_distribution
    bucket_distribution = data.get("bucket_distribution", data.get("top_100_distribution", {}))
    total_videos = sum(bucket_distribution.values())

    # Calculate percentage for each bucket, rounded to integer
    bucket_percentages = {
        bucket: round((count / total_videos) * 100)
        for bucket, count in bucket_distribution.items()
    }

    return bucket_percentages


def rank_competitor_top_buckets(client_id, competitor_handle, mode='top', strategy='contrastive'):
    """
    Rank competitor's top 3 buckets by performance.

    Ranking Criteria:
    1. Primary: Average engagement rate (higher is better)
    2. Secondary: Average views (higher is better)
    """

    # Discover analysis directory
    base_path = f"/home/jorge/rumiaifinal/data/clients/{client_id}/competitors/{competitor_handle}"

    # Try to find analysis directory
    analysis_dir = f"{mode}_{strategy}"
    competitor_path = f"{base_path}/{analysis_dir}"

    if not os.path.exists(competitor_path):
        raise FileNotFoundError(f"Analysis directory not found: {competitor_path}")

    # Load top 3 buckets
    with open(f"{competitor_path}/winner_analysis.json") as f:
        winner_data = json.load(f)
    top_3_buckets = winner_data["top_3_buckets"]

    # Collect performance data for each bucket
    bucket_data = []
    for bucket in top_3_buckets:
        # Load selected videos for this bucket
        bucket_path = f"{competitor_path}/buckets/bucket_{bucket}/selected_videos.json"

        with open(bucket_path) as f:
            selected_data = json.load(f)

        top_count = selected_data["top_count"]
        top_videos = selected_data["videos"][:top_count]

        # Calculate avg views
        avg_views = sum(v["playCount"] for v in top_videos) / len(top_videos)

        # Calculate avg engagement
        engagement_rates = []
        for video in top_videos:
            engagement = calculate_engagement_metrics(video)
            engagement_rates.append(engagement)

        avg_engagement = sum(engagement_rates) / len(engagement_rates)

        bucket_data.append({
            "bucket": bucket,
            "avg_views": int(avg_views),
            "avg_engagement": round(avg_engagement, 1)
        })

    # Normalize views and calculate composite scores
    max_views = max(b["avg_views"] for b in bucket_data)

    for bucket in bucket_data:
        normalized_views = (bucket["avg_views"] / max_views) * 100
        composite_score = normalized_views + bucket["avg_engagement"]
        bucket["composite_score"] = composite_score

    # Sort by composite score (DESC)
    bucket_data.sort(key=lambda b: b["composite_score"], reverse=True)

    # Assign ranks and star ratings
    star_map = {1: "⭐⭐⭐⭐⭐", 2: "⭐⭐⭐⭐", 3: "⭐⭐⭐⭐"}

    for idx, bucket in enumerate(bucket_data, start=1):
        bucket["rank"] = idx
        bucket["stars"] = star_map[idx]
        bucket["is_sweet_spot"] = (idx == 1)

    return bucket_data


def aggregate_content_classifications(bucket_name, base_path, performer_type="top"):
    """
    Aggregate Stage 2.7 content classifications for a specific bucket.

    Returns dict with Counters for each classification field.
    """

    content_dir = os.path.join(base_path, 'content_analysis', 'validated', f'bucket_{bucket_name}')

    if not os.path.exists(content_dir):
        print(f"⚠️  Warning: Content directory not found: {content_dir}")
        return None

    # Initialize counters
    content_categories = Counter()
    hook_strategies = Counter()
    closing_strategies = Counter()
    pain_points = Counter()
    keywords = Counter()
    engagement_drivers = Counter()
    content_tactics = Counter()
    caption_cta_types = Counter()

    # Aggregate from all videos in this bucket with matching performer_type
    files_processed = 0
    for filename in os.listdir(content_dir):
        if not filename.endswith('_content.json'):
            continue

        filepath = os.path.join(content_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Filter by performer_type if field exists
        if data.get('performer_type') and data['performer_type'] != performer_type:
            continue

        files_processed += 1

        # Aggregate classification fields
        if data.get('content_category'):
            content_categories[data['content_category']] += 1

        if data.get('hook_strategy'):
            hook_strategies[data['hook_strategy']] += 1

        if data.get('closing_strategy'):
            closing_strategies[data['closing_strategy']] += 1

        for pain in data.get('pain_points', []):
            if pain:
                pain_points[pain] += 1

        for keyword in data.get('keywords', []):
            if keyword:
                keywords[keyword] += 1

        for driver in data.get('engagement_drivers', []):
            if driver:
                engagement_drivers[driver] += 1

        for tactic in data.get('content_tactics', []):
            if tactic:
                content_tactics[tactic] += 1

        # Caption analysis
        caption = data.get('caption_analysis', {})
        if caption.get('cta_type'):
            caption_cta_types[caption['cta_type']] += 1

    if files_processed == 0:
        print(f"⚠️  Warning: No {performer_type} performer files found in {bucket_name}")
        return None

    print(f"✓ Aggregated {files_processed} {performer_type} performer videos from {bucket_name}")

    return {
        'content_category': content_categories,
        'hook_strategy': hook_strategies,
        'closing_strategy': closing_strategies,
        'pain_points': pain_points,
        'keywords': keywords,
        'engagement_drivers': engagement_drivers,
        'content_tactics': content_tactics,
        'caption_cta_type': caption_cta_types
    }


def extract_hashtag_analysis(client_id, competitor_handle, mode='top', strategy='contrastive'):
    """
    Extract hashtag usage patterns from competitor's winning buckets.
    """

    # Discover analysis directory
    base_path = f"/home/jorge/rumiaifinal/data/clients/{client_id}/competitors/{competitor_handle}"

    analysis_dir = f"{mode}_{strategy}"
    competitor_path = f"{base_path}/{analysis_dir}"

    if not os.path.exists(competitor_path):
        raise FileNotFoundError(f"Analysis directory not found: {competitor_path}")

    # Load winning buckets
    with open(f"{competitor_path}/winner_analysis.json") as f:
        winner_data = json.load(f)
    winning_buckets = winner_data["top_3_buckets"]

    # Collect hashtags from all winning buckets
    all_hashtags = []
    total_videos = 0

    for bucket in winning_buckets:
        bucket_path = f"{competitor_path}/buckets/bucket_{bucket}/selected_videos.json"

        with open(bucket_path) as f:
            data = json.load(f)

        top_count = data["top_count"]
        top_videos = data["videos"][:top_count]
        total_videos += len(top_videos)

        for video in top_videos:
            hashtags = video.get("hashtags", [])
            for hashtag in hashtags:
                tag_name = hashtag.get("name", "")
                if tag_name:
                    all_hashtags.append(tag_name.lower())

    # Calculate statistics
    unique_hashtags = set(all_hashtags)
    hashtag_counter = Counter(all_hashtags)

    total_hashtags = len(all_hashtags)
    avg_hashtags_per_video = round(total_hashtags / total_videos, 1) if total_videos > 0 else 0

    # Top 10 hashtags
    top_10 = []
    for tag, count in hashtag_counter.most_common(10):
        usage_pct = round((count / total_videos) * 100)
        top_10.append({
            "tag": f"#{tag}",
            "usage_pct": usage_pct,
            "video_count": count
        })

    # Top 5 concentration
    top_5_count = sum(count for _, count in hashtag_counter.most_common(5))
    top_5_concentration = round((top_5_count / total_hashtags) * 100) if total_hashtags > 0 else 0

    return {
        "total_unique_hashtags": len(unique_hashtags),
        "avg_hashtags_per_video": avg_hashtags_per_video,
        "top_5_concentration": top_5_concentration,
        "top_10_hashtags": top_10
    }


def extract_mention_analysis(client_id, competitor_handle, mode='top', strategy='contrastive'):
    """
    Extract @mention patterns from video captions.

    Detects:
    - Videos with @mentions (potential UGC/affiliate content)
    - Repost indicators ("repost", "via", "credit", "by", "from")
    """

    # Discover analysis directory
    base_path = f"/home/jorge/rumiaifinal/data/clients/{client_id}/competitors/{competitor_handle}"

    analysis_dir = f"{mode}_{strategy}"
    competitor_path = f"{base_path}/{analysis_dir}"

    if not os.path.exists(competitor_path):
        raise FileNotFoundError(f"Analysis directory not found: {competitor_path}")

    # Load winning buckets
    with open(f"{competitor_path}/winner_analysis.json") as f:
        winner_data = json.load(f)
    winning_buckets = winner_data["top_3_buckets"]

    # Collect mentions from all winning buckets
    all_mentions = []
    mention_to_full_name = {}  # Map @handle to full brand name
    videos_with_mentions = 0
    total_videos = 0

    repost_indicators = ["repost", "via", "credit", "by", "from"]

    for bucket in winning_buckets:
        bucket_path = f"{competitor_path}/buckets/bucket_{bucket}/selected_videos.json"

        with open(bucket_path) as f:
            data = json.load(f)

        top_count = data["top_count"]
        top_videos = data["videos"][:top_count]
        total_videos += len(top_videos)

        for video in top_videos:
            caption_original = video.get("text", "")
            caption = caption_original.lower()

            # Extract @mentions with context to get full brand name
            # Pattern: @handle followed by optional text (capture up to 30 chars after @)
            mention_contexts = re.findall(r'@(\w+)([^\n#@]{0,30})', caption_original)

            for handle_lower, context in mention_contexts:
                handle = handle_lower.lower()
                all_mentions.append(handle)

                # Try to extract full brand name from context
                if handle not in mention_to_full_name:
                    # Clean up context: remove extra spaces, punctuation at start
                    full_context = context.strip().split()[0:3]  # Take first 3 words max
                    full_context_str = ' '.join(full_context).strip('.,!?;: ')

                    if full_context_str and len(full_context_str) > 1:
                        mention_to_full_name[handle] = f"@{handle_lower} ({full_context_str})"
                    else:
                        mention_to_full_name[handle] = f"@{handle_lower}"

            # Check repost indicators
            has_repost_indicator = any(indicator in caption for indicator in repost_indicators)

            if mention_contexts or has_repost_indicator:
                videos_with_mentions += 1

    # Calculate statistics
    unique_mentions = set(all_mentions)
    mention_counter = Counter(all_mentions)

    mention_rate = round((videos_with_mentions / total_videos) * 100) if total_videos > 0 else 0
    repost_rate = mention_rate  # Same metric for this analysis

    # Top 10 mentions
    top_10 = []
    for handle, count in mention_counter.most_common(10):
        percentage = round((count / total_videos) * 100, 1)
        full_name = mention_to_full_name.get(handle, f"@{handle}")
        top_10.append({
            "handle": full_name,
            "percentage": percentage,
            "mention_count": count
        })

    return {
        "total_videos": total_videos,
        "videos_with_mentions": videos_with_mentions,
        "mention_rate": mention_rate,
        "repost_rate": repost_rate,
        "total_unique_mentions": len(unique_mentions),
        "top_10_mentions": top_10
    }


def select_qr_code_videos(bucket_path, performance_group="top", num_videos=2):
    """
    Select top 2 performer videos for QR code generation.

    Args:
        bucket_path: Path to bucket directory
        performance_group: "top" or "bottom"
        num_videos: Number of top videos to select (default 2)

    Returns:
        List of video dicts with url, views, engagement, etc.
    """

    with open(f"{bucket_path}/selected_videos.json") as f:
        data = json.load(f)

    selected_videos = []

    if performance_group == "top":
        # First N videos in array = highest views
        top_count = data["top_count"]
        for i in range(min(num_videos, top_count)):
            video = data["videos"][i]
            engagement = calculate_engagement_metrics(video)
            selected_videos.append({
                "video_id": video["id"],
                "url": video["webVideoUrl"],
                "views": video["playCount"],
                "engagement": engagement,
                "duration": video["duration"],
                "rank": i + 1
            })
    elif performance_group == "bottom":
        # Last N videos among bottom performers
        top_count = data["top_count"]
        bottom_count = data["bottom_count"]
        for i in range(min(num_videos, bottom_count)):
            video = data["videos"][top_count + bottom_count - 1 - i]
            engagement = calculate_engagement_metrics(video)
            selected_videos.append({
                "video_id": video["id"],
                "url": video["webVideoUrl"],
                "views": video["playCount"],
                "engagement": engagement,
                "duration": video["duration"],
                "rank": i + 1
            })
    else:
        raise ValueError(f"Invalid performance_group: {performance_group}")

    return selected_videos


def generate_qr_codes(qr_data_list, output_dir):
    """
    Generate QR code PNG files from TikTok URLs.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for qr_data in qr_data_list:
        # Create QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_data["url"])
        qr.make(fit=True)

        # Generate image
        img = qr.make_image(fill_color="black", back_color="white")

        # Save to file
        output_path = os.path.join(output_dir, qr_data["filename"])
        img.save(output_path)

    print(f"✓ Generated {len(qr_data_list)} QR code(s)")


def format_views(view_count):
    """Format view count with K or M suffix."""
    if view_count >= 1000000:
        return f"{view_count / 1000000:.1f}M"
    elif view_count >= 1000:
        return f"{int(view_count / 1000)}K"
    else:
        return str(view_count)


def calculate_posting_frequency(client_id, competitor_handle, mode='top', strategy='contrastive'):
    """
    Calculate videos per week from winner_analysis.json.

    Uses actual date_filter from config.json to calculate accurate posting frequency.
    """

    base_path = f"/home/jorge/rumiaifinal/data/clients/{client_id}/competitors/{competitor_handle}"

    analysis_dir = f"{mode}_{strategy}"
    winner_analysis_path = f"{base_path}/{analysis_dir}/winner_analysis.json"

    with open(winner_analysis_path) as f:
        data = json.load(f)

    total_videos = sum(data["top_100_distribution"].values())

    # Extract actual date filter from config.json
    date_filter = extract_date_filter_from_config(client_id, competitor_handle, mode, strategy)

    # Extract number of days from date_filter (e.g., "last_90_days" → 90)
    try:
        if date_filter.startswith('last_') and date_filter.endswith('_days'):
            days = int(date_filter.replace('last_', '').replace('_days', ''))
        else:
            days = 90  # Fallback to 90 days
    except:
        days = 90  # Fallback to 90 days

    weeks = days / 7

    posting_freq = round(total_videos / weeks, 1)

    return posting_freq


def extract_transcript_quality(client_id, competitor_handle, mode='top', strategy='contrastive'):
    """
    Extract transcript validation statistics from validation cache.

    Returns:
        dict: {
            "with_speech": 48,
            "speech_pct": 36
        }
        or None if cache doesn't exist
    """
    base_path = f"/home/jorge/rumiaifinal/data/clients/{client_id}/competitors/{competitor_handle}"
    analysis_dir = f"{mode}_{strategy}"
    cache_path = f"{base_path}/{analysis_dir}/content_taxonomies/transcript_validation_cache.json"

    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)

        total_videos = data['stats']['total']
        valid_count = data['stats']['valid']

        return {
            'with_speech': valid_count,
            'speech_pct': round((valid_count / total_videos) * 100) if total_videos > 0 else 0
        }
    except Exception as e:
        print(f"⚠️  Warning: Could not read transcript quality for {competitor_handle}: {e}")
        return None


def determine_hashtag_strategy_type(total_unique_hashtags):
    """Determine if hashtag strategy is Diversified or Focused."""
    return "Diversified" if total_unique_hashtags > 20 else "Focused"


def calculate_original_content_percentage(repost_rate):
    """Calculate original content percentage (inverse of repost rate)."""
    return 100 - int(repost_rate)


def extract_date_filter_from_config(client_id, competitor_handle, mode='top', strategy='contrastive'):
    """
    Extract date_filter from competitor's config.json.

    Returns:
        str: date_filter value (e.g., "last_90_days") or "last_90_days" as fallback
    """
    config_path = f"/home/jorge/rumiaifinal/data/clients/{client_id}/competitors/{competitor_handle}/{mode}_{strategy}/config.json"

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('date_filter', 'last_90_days')
    except Exception as e:
        print(f"⚠️  Warning: Could not read date_filter from config for {competitor_handle}: {e}")
        return 'last_90_days'


def format_date_filter(date_filter):
    """
    Convert date_filter to human-readable format.

    Examples:
        "last_90_days" → "Last 90 days"
        "last_30_days" → "Last 30 days"
        "last_180_days" → "Last 180 days"

    Returns:
        str: Formatted date filter for display
    """
    if not date_filter or date_filter == 'none':
        return "All time"

    # Handle "last_N_days" format
    if date_filter.startswith('last_') and date_filter.endswith('_days'):
        # Extract number from "last_90_days"
        try:
            days = date_filter.replace('last_', '').replace('_days', '')
            return f"Last {days} days"
        except:
            return "Last 90 days"

    # Fallback
    return "Last 90 days"


def load_taxonomy_descriptions(client_id, competitor_handle, mode='top', strategy='top'):
    """
    Load taxonomy descriptions from competitor's taxonomy file.

    Returns dict with mappings:
    {
        'content_categories': {'comedic_relatable_moment': 'Short comedic skits...', ...},
        'hook_strategies': {'direct_statement': 'Opens with a clear...', ...},
        'engagement_drivers': {'treat_yourself_messaging': 'description', ...},
        'cta_strategies': {...}  # May not have definitions, handle gracefully
    }
    """
    base_path = f"/home/jorge/rumiaifinal/data/clients/{client_id}/competitors/{competitor_handle}"
    analysis_dir = f"{mode}_{strategy}"
    taxonomy_path = f"{base_path}/{analysis_dir}/content_taxonomies/{competitor_handle}_taxonomy.json"

    # Return empty mappings if taxonomy file doesn't exist
    if not os.path.exists(taxonomy_path):
        print(f"⚠️  Warning: Taxonomy file not found: {taxonomy_path}")
        return {
            'content_categories': {},
            'hook_strategies': {},
            'engagement_drivers': {},
            'cta_strategies': {}
        }

    with open(taxonomy_path, 'r') as f:
        taxonomy = json.load(f)

    # Build mappings from name -> definition
    mappings = {
        'content_categories': {},
        'hook_strategies': {},
        'engagement_drivers': {},
        'cta_strategies': {}
    }

    # Content categories (array of objects with name/definition)
    for item in taxonomy.get('content_categories', []):
        name = item.get('name', '')
        definition = item.get('definition', '')
        if name:
            mappings['content_categories'][name] = definition

    # Hook strategies (array of objects with name/definition)
    for item in taxonomy.get('hook_strategies', []):
        name = item.get('name', '')
        definition = item.get('definition', '')
        if name:
            mappings['hook_strategies'][name] = definition

    # Engagement drivers (may be simple strings, not objects)
    engagement_drivers = taxonomy.get('engagement_drivers', [])
    if engagement_drivers and isinstance(engagement_drivers[0], dict):
        # If they have definitions
        for item in engagement_drivers:
            name = item.get('name', '')
            definition = item.get('definition', '')
            if name:
                mappings['engagement_drivers'][name] = definition
    else:
        # If they're just strings, use the string as both key and description
        for driver in engagement_drivers:
            if driver:
                mappings['engagement_drivers'][driver] = driver.replace('_', ' ').title()

    # CTA strategies (may not exist in taxonomy, handle gracefully)
    cta_strategies = taxonomy.get('cta_strategies', [])
    if cta_strategies:
        if isinstance(cta_strategies[0], dict):
            for item in cta_strategies:
                name = item.get('name', '')
                definition = item.get('definition', '')
                if name:
                    mappings['cta_strategies'][name] = definition

    return mappings


# =============================
# MAIN EXTRACTION WORKFLOW
# =============================

def main():
    """Main extraction workflow"""

    # =============================
    # STEP 1: Parse CLI Arguments
    # =============================
    parser = argparse.ArgumentParser(description='Extract Report 3: Single Competitor')
    parser.add_argument('--client', required=True, help='Client ID (e.g., acme)')
    parser.add_argument('--competitor', required=True, help='Competitor handle without @ (e.g., drinkpoppi)')
    parser.add_argument('--mode', default='top', help='Mode (default: top)')
    parser.add_argument('--strategy', default='contrastive', help='Strategy (default: contrastive)')
    args = parser.parse_args()

    print(f"\n🔍 Extracting Competitor Report for @{args.competitor}")

    # =============================
    # STEP 2: Build File Paths
    # =============================
    base_path = f"/home/jorge/rumiaifinal/data/clients/{args.client}/competitors/{args.competitor}"

    analysis_dir = f"{args.mode}_{args.strategy}"
    analysis_base_path = f"{base_path}/{analysis_dir}"

    if not os.path.exists(analysis_base_path):
        print(f"❌ Error: Analysis directory not found: {analysis_base_path}")
        return

    # Create reports/competitor/ directory structure
    reports_base_path = os.path.join(analysis_base_path, 'reports', 'competitor')
    os.makedirs(reports_base_path, exist_ok=True)

    # =============================
    # STEP 3: Load Core Data
    # =============================
    winner_analysis_path = os.path.join(analysis_base_path, 'winner_analysis.json')
    with open(winner_analysis_path) as f:
        winner_data = json.load(f)

    winning_buckets = winner_data['top_3_buckets']

    # Calculate total videos analyzed
    total_videos = 0
    for bucket in winning_buckets:
        bucket_path = os.path.join(analysis_base_path, 'buckets', f'bucket_{bucket}')
        with open(f"{bucket_path}/selected_videos.json") as f:
            data = json.load(f)
        total_videos += data['selected_count']

    print(f"✓ Analyzing {total_videos} videos across {len(winning_buckets)} winning buckets...")

    # =============================
    # STEP 4: Calculate Performance Metrics
    # =============================
    print("📊 Calculating performance metrics...")

    # Bucket distribution
    bucket_percentages = calculate_bucket_distribution(winner_analysis_path)
    primary_focus = max(bucket_percentages, key=bucket_percentages.get)

    # Rank top 3 buckets
    ranked_buckets = rank_competitor_top_buckets(args.client, args.competitor, args.mode, args.strategy)
    sweet_spot = ranked_buckets[0]['bucket']

    # Coverage percentage
    coverage_pct = sum(bucket_percentages[b] for b in winning_buckets)

    # Posting frequency
    posting_freq = calculate_posting_frequency(args.client, args.competitor, args.mode, args.strategy)

    # =============================
    # STEP 4.5: Load Taxonomy Descriptions
    # =============================
    taxonomy_descriptions = load_taxonomy_descriptions(args.client, args.competitor, args.mode, args.strategy)

    # =============================
    # STEP 5: Aggregate Content Intelligence
    # =============================
    print("🧠 Aggregating content intelligence from Stage 2.7 classifications...")

    # Aggregate across all winning buckets
    all_content_categories = Counter()
    all_hook_strategies = Counter()
    all_closing_strategies = Counter()  # CTA Strategy from closing_strategy field
    all_caption_cta_types = Counter()  # Caption CTA from caption_analysis.cta_type
    all_pain_points = Counter()
    all_keywords = Counter()
    all_engagement_drivers = Counter()
    all_content_tactics = Counter()

    for bucket in winning_buckets:
        aggregated = aggregate_content_classifications(
            bucket_name=bucket,
            base_path=analysis_base_path,
            performer_type="top"
        )

        if aggregated:
            all_content_categories.update(aggregated['content_category'])
            all_hook_strategies.update(aggregated['hook_strategy'])
            all_closing_strategies.update(aggregated['closing_strategy'])  # CTA Strategy
            all_caption_cta_types.update(aggregated['caption_cta_type'])  # Caption CTA
            all_pain_points.update(aggregated['pain_points'])
            all_keywords.update(aggregated['keywords'])
            all_engagement_drivers.update(aggregated['engagement_drivers'])
            all_content_tactics.update(aggregated['content_tactics'])

    # Get top N
    top_5_categories = all_content_categories.most_common(5)
    top_4_hooks = all_hook_strategies.most_common(4)
    top_4_cta_strategies = all_closing_strategies.most_common(4)  # CTA Strategy from closing_strategy
    top_4_caption_ctas = all_caption_cta_types.most_common(4)  # Caption CTA from caption
    top_5_pain_points = all_pain_points.most_common(5)
    top_5_keywords = all_keywords.most_common(5)
    top_4_drivers = all_engagement_drivers.most_common(4)
    top_4_tactics = all_content_tactics.most_common(4)

    total_classified = sum(all_content_categories.values())

    # =============================
    # STEP 6: Extract Hashtag & Mention Analysis
    # =============================
    print("🏷️  Extracting hashtag and mention analysis...")

    hashtag_analysis = extract_hashtag_analysis(args.client, args.competitor, args.mode, args.strategy)
    mention_analysis = extract_mention_analysis(args.client, args.competitor, args.mode, args.strategy)

    strategy_type = determine_hashtag_strategy_type(hashtag_analysis['total_unique_hashtags'])
    original_content_pct = calculate_original_content_percentage(mention_analysis['repost_rate'])

    # =============================
    # STEP 7: Select QR Code Videos (Top 2 per Winning Bucket)
    # =============================
    num_qr_codes = len(winning_buckets) * 2
    print(f"📱 Generating {num_qr_codes} QR codes (2 per bucket)...")

    # Collect top 2 videos from each winning bucket
    qr_videos_by_bucket = []
    for bucket_rank_data in ranked_buckets:
        bucket_name = bucket_rank_data['bucket']
        bucket_path = os.path.join(analysis_base_path, 'buckets', f'bucket_{bucket_name}')

        # Get top 2 videos from this bucket
        videos = select_qr_code_videos(bucket_path, "top", num_videos=2)

        for video in videos:
            qr_videos_by_bucket.append({
                'bucket_name': bucket_name,
                'bucket_rank': bucket_rank_data['rank'],
                'is_sweet_spot': bucket_rank_data['is_sweet_spot'],
                'video_rank': video['rank'],
                'video_data': video
            })

    # Generate QR codes
    qr_output_dir = os.path.join(reports_base_path, 'qr_codes')
    os.makedirs(qr_output_dir, exist_ok=True)

    qr_data = []
    for qr_info in qr_videos_by_bucket:
        filename = f"{args.competitor}_{qr_info['bucket_name']}_rank{qr_info['video_rank']}.png"
        qr_data.append({
            "filename": filename,
            "url": qr_info['video_data']['url']
        })

    generate_qr_codes(qr_data, qr_output_dir)

    # =============================
    # STEP 8: Build Excel Data Structure
    # =============================
    tab_data = []

    # PAGE 1: EXECUTIVE OVERVIEW
    tab_data.append(['PAGE_1_EXECUTIVE_OVERVIEW', ''])
    tab_data.append(['', ''])

    tab_data.append(['COMPETITOR_HANDLE', f'@{args.competitor}'])

    # Extract date_filter from config.json
    date_filter_raw = extract_date_filter_from_config(args.client, args.competitor, args.mode, args.strategy)
    analysis_period = format_date_filter(date_filter_raw)
    tab_data.append(['ANALYSIS_PERIOD', analysis_period])

    tab_data.append(['VIDEOS_ANALYZED', str(total_videos)])

    # PAGE 2: CONTENT STRATEGY ANALYSIS
    tab_data.append(['', ''])
    tab_data.append(['PAGE_2_CONTENT_STRATEGY', ''])
    tab_data.append(['', ''])

    # Duration Distribution
    all_buckets = ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]
    for bucket in all_buckets:
        field_name = f'BUCKET_{bucket.replace("-", "_").upper()}_PCT'
        pct = bucket_percentages.get(bucket, 0)
        tab_data.append([field_name, str(pct)])

    tab_data.append(['', ''])
    tab_data.append(['PRIMARY_FOCUS_BUCKET', primary_focus])

    # Performance by Duration
    tab_data.append(['', ''])
    for i, bucket_data in enumerate(ranked_buckets, 1):
        tab_data.append([f'PERF_BUCKET_{i}_NAME', bucket_data['bucket']])
        tab_data.append([f'PERF_BUCKET_{i}_AVG_VIEWS', format_views(bucket_data['avg_views'])])
        tab_data.append([f'PERF_BUCKET_{i}_AVG_ENG', str(bucket_data['avg_engagement'])])
        tab_data.append([f'PERF_BUCKET_{i}_STARS', bucket_data['stars']])
        tab_data.append([f'PERF_BUCKET_{i}_IS_SWEET_SPOT', str(bucket_data['is_sweet_spot'])])
        tab_data.append(['', ''])

    tab_data.append(['SWEET_SPOT_BUCKET', sweet_spot])
    tab_data.append(['COVERAGE_PERCENTAGE', str(coverage_pct)])

    # Posting Frequency
    tab_data.append(['', ''])
    tab_data.append(['POSTING_FREQUENCY', str(posting_freq)])
    tab_data.append(['POSTING_ACTIVITY_LABEL', f'@{args.competitor} posts {posting_freq} videos per week on average'])

    # Transcript Quality (Speech Data)
    tab_data.append(['', ''])
    transcript_quality = extract_transcript_quality(args.client, args.competitor, args.mode, args.strategy)

    if transcript_quality:
        tab_data.append(['WITH_SPEECH', str(transcript_quality['with_speech'])])
        tab_data.append(['SPEECH_PCT', str(transcript_quality['speech_pct'])])
        # Calculate WITHOUT_SPEECH metrics
        without_speech = total_videos - transcript_quality['with_speech']
        without_speech_pct = 100 - transcript_quality['speech_pct']
        tab_data.append(['WITHOUT_SPEECH', str(without_speech)])
        tab_data.append(['WITHOUT_SPEECH_PCT', str(without_speech_pct)])
    else:
        tab_data.append(['WITH_SPEECH', 'N/A'])
        tab_data.append(['SPEECH_PCT', 'N/A'])
        tab_data.append(['WITHOUT_SPEECH', 'N/A'])
        tab_data.append(['WITHOUT_SPEECH_PCT', 'N/A'])

    # PAGE 3: CREATIVE INTELLIGENCE
    tab_data.append(['', ''])
    tab_data.append(['PAGE_3_CREATIVE_INTELLIGENCE', ''])
    tab_data.append(['', ''])

    # Section 1: Content DNA
    tab_data.append(['SECTION_WHAT_CREATES', f'WHAT @{args.competitor.upper()} CREATES:'])
    tab_data.append(['', ''])

    # Content Categories
    for i, (category, count) in enumerate(top_5_categories, 1):
        pct = round((count / total_classified) * 100) if total_classified > 0 else 0
        tab_data.append([f'CONTENT_CATEGORY_{i}', category.replace('_', ' ').title()])
        tab_data.append([f'CONTENT_CATEGORY_{i}_PCT', str(pct)])
        # Get description from taxonomy
        description = taxonomy_descriptions['content_categories'].get(category, 'No description available')
        tab_data.append([f'CONTENT_CATEGORY_{i}_DESC', description])

    tab_data.append(['', ''])

    # Engagement Drivers (no descriptions in taxonomy)
    for i, (driver, count) in enumerate(top_4_drivers, 1):
        pct = round((count / total_classified) * 100) if total_classified > 0 else 0
        tab_data.append([f'ENGAGEMENT_DRIVER_{i}', driver.replace('_', ' ').title()])
        tab_data.append([f'ENGAGEMENT_DRIVER_{i}_PCT', str(pct)])

    # Section 2: Execution Playbook
    tab_data.append(['', ''])
    tab_data.append(['SECTION_HOW_EXECUTES', f'HOW @{args.competitor.upper()} EXECUTES:'])
    tab_data.append(['', ''])

    # 1. Hook Strategies (with DESC)
    for i, (hook, count) in enumerate(top_4_hooks, 1):
        pct = round((count / total_classified) * 100) if total_classified > 0 else 0
        tab_data.append([f'HOOK_STRATEGY_{i}', hook.replace('_', ' ').title()])
        tab_data.append([f'HOOK_STRATEGY_{i}_PCT', str(pct)])
        # Get description from taxonomy
        description = taxonomy_descriptions['hook_strategies'].get(hook, 'No description available')
        tab_data.append([f'HOOK_STRATEGY_{i}_DESC', description])

    tab_data.append(['', ''])

    # 2. CTA Strategies (from closing_strategy, no DESC)
    for i, (cta, count) in enumerate(top_4_cta_strategies, 1):
        pct = round((count / total_classified) * 100) if total_classified > 0 else 0
        tab_data.append([f'CTA_STRATEGY_{i}', cta.replace('_', ' ').title()])
        tab_data.append([f'CTA_STRATEGY_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # 3. Pain Points (calculate based on total videos, not total_classified)
    for i, (pain, count) in enumerate(top_5_pain_points, 1):
        pct = round((count / total_videos) * 100) if total_videos > 0 else 0
        tab_data.append([f'PAIN_POINT_{i}', pain.replace('_', ' ').title()])
        tab_data.append([f'PAIN_POINT_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # 4. Content Tactics
    for i, (tactic, count) in enumerate(top_4_tactics, 1):
        pct = round((count / total_classified) * 100) if total_classified > 0 else 0
        tab_data.append([f'CONTENT_TACTIC_{i}', tactic.replace('_', ' ').title()])
        tab_data.append([f'CONTENT_TACTIC_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # 5. Caption CTA Strategies (from caption_analysis.cta_type, no DESC)
    for i, (caption_cta, count) in enumerate(top_4_caption_ctas, 1):
        pct = round((count / total_videos) * 100) if total_videos > 0 else 0
        tab_data.append([f'CAPTION_CTA_STRATEGY_{i}', caption_cta.replace('_', ' ').title()])
        tab_data.append([f'CAPTION_CTA_STRATEGY_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # 6. Keywords
    for i, (keyword, count) in enumerate(top_5_keywords, 1):
        tab_data.append([f'KEYWORD_{i}', keyword])

    # Hashtag Strategy
    tab_data.append(['', ''])
    for i, hashtag_data in enumerate(hashtag_analysis['top_10_hashtags'], 1):
        tab_data.append([f'HASHTAG_{i}', hashtag_data['tag']])
        tab_data.append([f'HASHTAG_{i}_PCT', str(hashtag_data['usage_pct'])])

    tab_data.append(['', ''])
    tab_data.append(['TOTAL_UNIQUE_HASHTAGS', str(hashtag_analysis['total_unique_hashtags'])])
    tab_data.append(['AVG_HASHTAGS_PER_VIDEO', str(round(hashtag_analysis['avg_hashtags_per_video']))])
    tab_data.append(['STRATEGY_TYPE', strategy_type])

    # Caption Strategy
    tab_data.append(['', ''])
    tab_data.append(['AVG_HASHTAG_COUNT', str(round(hashtag_analysis['avg_hashtags_per_video']))])

    if all_caption_cta_types:
        top_caption_cta, top_caption_cta_count = all_caption_cta_types.most_common(1)[0]
        top_caption_cta_pct = round((top_caption_cta_count / total_videos) * 100) if total_videos > 0 else 0
        tab_data.append(['TOP_CAPTION_CTA_TYPE', top_caption_cta.replace('_', ' ').title()])
        tab_data.append(['TOP_CAPTION_CTA_TYPE_PCT', str(top_caption_cta_pct)])

    # Content Sourcing
    tab_data.append(['', ''])
    tab_data.append(['ORIGINAL_CONTENT_PCT', str(original_content_pct)])
    tab_data.append(['REPOSTED_AFFILIATE_PCT', str(mention_analysis['repost_rate'])])
    tab_data.append(['', ''])

    # Top 5 affiliates
    for i, affiliate in enumerate(mention_analysis['top_10_mentions'][:5], 1):
        tab_data.append([f'AFFILIATE_{i}_HANDLE', affiliate['handle']])
        tab_data.append([f'AFFILIATE_{i}_PCT', str(affiliate['percentage'])])
        tab_data.append([f'AFFILIATE_{i}_COUNT', str(affiliate['mention_count'])])

    tab_data.append(['', ''])
    tab_data.append(['TOTAL_UNIQUE_MENTIONS', str(mention_analysis['total_unique_mentions'])])

    # Section 6: Creative Formulas
    tab_data.append(['', ''])
    for i, bucket_name in enumerate(winner_data['top_3_buckets'], 1):
        bucket_path_formulas = os.path.join(analysis_base_path, 'buckets', f'bucket_{bucket_name}')
        winning_formulas_path = os.path.join(bucket_path_formulas, 'ml_analysis', 'llm', 'winning_formulas.json')

        tab_data.append([f'BUCKET_{i}_NAME', bucket_name])

        if os.path.exists(winning_formulas_path):
            with open(winning_formulas_path, 'r') as f:
                winning_formulas = json.load(f)
                creative_reports = winning_formulas.get('creative_reports', [])

                for j in range(min(3, len(creative_reports))):
                    formula_name = creative_reports[j].get('formula_name', '')
                    tab_data.append([f'BUCKET_{i}_FORMULA_{j+1}_NAME', formula_name if formula_name else 'Insufficient data'])
        else:
            # Add placeholder formulas if file doesn't exist
            for j in range(3):
                tab_data.append([f'BUCKET_{i}_FORMULA_{j+1}_NAME', 'Insufficient data'])

        if i < len(winner_data['top_3_buckets']):  # Add empty row between buckets (not after last one)
            tab_data.append(['', ''])

    # QR Code Metadata (2 per bucket, dynamically generated)
    tab_data.append(['', ''])
    for qr_info in qr_videos_by_bucket:
        bucket_name = qr_info['bucket_name']
        bucket_rank = qr_info['bucket_rank']
        video_rank = qr_info['video_rank']
        video = qr_info['video_data']
        is_sweet_spot = qr_info['is_sweet_spot']

        # Label: SWEET_SPOT_BUCKET_QR_CODE_1 or BUCKET_2_QR_CODE_1, etc.
        if is_sweet_spot:
            label_prefix = 'SWEET_SPOT_BUCKET'
        else:
            label_prefix = f'BUCKET_{bucket_rank}'

        tab_data.append([f'{label_prefix}_QR_CODE_{video_rank}_FILE', f"{args.competitor}_{bucket_name}_rank{video_rank}.png"])
        tab_data.append([f'{label_prefix}_QR_CODE_{video_rank}_URL', video['url']])
        tab_data.append([f'{label_prefix}_QR_CODE_{video_rank}_VIEWS', format_views(video['views'])])
        tab_data.append([f'{label_prefix}_QR_CODE_{video_rank}_ENGAGEMENT', str(video['engagement'])])
        tab_data.append([f'{label_prefix}_QR_CODE_{video_rank}_DURATION', f"{video['duration']}s"])
        tab_data.append([f'{label_prefix}_QR_CODE_{video_rank}_BUCKET', bucket_name])
        tab_data.append(['', ''])

    # =============================
    # STEP 9: Write Excel File
    # =============================
    print("💾 Writing Excel file...")

    excel_filename = f"{args.competitor}_analysis_data.xlsx"
    excel_path = os.path.join(reports_base_path, excel_filename)

    df = pd.DataFrame(tab_data, columns=['Field Name', 'Value'])

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Report_Data', index=False)

    # =============================
    # STEP 10: Print Success Message
    # =============================
    print(f"\n✅ Extraction complete!")
    print(f"  📁 Excel: {excel_path}")
    print(f"  📁 QR code: {os.path.join(qr_output_dir, qr_data[0]['filename'])}")
    print(f"  📊 Total fields: {len(tab_data)}")


if __name__ == '__main__':
    main()
