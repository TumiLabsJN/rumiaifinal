#!/usr/bin/env python3
"""
extract_client_data.py - Report 1: Hashtag → Client Executive Report

Generates executive dashboard for clients with market intelligence across all buckets.

Usage:
    python extract_client_data.py --client rollo_test5 --hashtag wellnesspt2_test5 --mode top --strategy contrastive

Output:
    - Single Excel file with ~122 fields (all buckets aggregated)
"""

import argparse
import json
import os
import pandas as pd
from collections import Counter


# =============================
# HELPER FUNCTIONS
# =============================

def calculate_engagement_metrics(video):
    """
    Calculate real engagement rate from TikTok video metadata.

    Args:
        video: Dict with TikTok video metadata

    Returns:
        float: Engagement rate percentage
    """
    views = video.get('playCount', 0)
    if views == 0:
        return 0.0

    likes = video.get('diggCount', 0)
    comments = video.get('commentCount', 0)
    shares = video.get('shareCount', 0)

    total_engagement = likes + comments + shares
    engagement_rate = (total_engagement / views) * 100

    return round(engagement_rate, 1)


def aggregate_content_classifications(bucket_name, base_path, performer_type="top"):
    """
    Aggregate Stage 2.7 content classifications for a specific bucket and performer type.

    Uses the NEW per-bucket validated structure with performer_type filtering.

    Args:
        bucket_name: Bucket identifier (e.g., "18-33s")
        base_path: Base path to analysis directory
        performer_type: "top" or "bottom"

    Returns:
        dict: Aggregated Counter objects for each classification field
    """
    content_dir = os.path.join(base_path, 'content_analysis', 'validated', f'bucket_{bucket_name}')

    if not os.path.exists(content_dir):
        print(f"⚠️  Warning: Content directory not found: {content_dir}")
        return None

    # Initialize Counters
    content_categories = Counter()
    hook_strategies = Counter()
    closing_strategies = Counter()
    pain_points = Counter()
    keywords = Counter()
    engagement_drivers = Counter()
    content_tactics = Counter()

    # Aggregate from all videos in this bucket with matching performer_type
    files_processed = 0
    for filename in os.listdir(content_dir):
        if not filename.endswith('_content.json'):
            continue

        filepath = os.path.join(content_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Filter by performer_type
        if data.get('performer_type') != performer_type:
            continue

        files_processed += 1

        # Aggregate fields (handle None values)
        if data.get('content_category'):
            content_categories[data['content_category']] += 1

        if data.get('hook_strategy'):
            hook_strategies[data['hook_strategy']] += 1

        if data.get('closing_strategy'):
            closing_strategies[data['closing_strategy']] += 1

        for pain_point in data.get('pain_points', []):
            if pain_point:
                pain_points[pain_point] += 1

        for keyword in data.get('keywords', []):
            if keyword:
                keywords[keyword] += 1

        for driver in data.get('engagement_drivers', []):
            if driver:
                engagement_drivers[driver] += 1

        for tactic in data.get('content_tactics', []):
            if tactic:
                content_tactics[tactic] += 1

    if files_processed == 0:
        return None

    return {
        'content_category': content_categories,
        'hook_strategy': hook_strategies,
        'closing_strategy': closing_strategies,
        'pain_points': pain_points,
        'keywords': keywords,
        'engagement_drivers': engagement_drivers,
        'content_tactics': content_tactics
    }


def calculate_bucket_distribution(winner_analysis_path):
    """
    Calculate percentage distribution across 8 buckets.

    Args:
        winner_analysis_path: Path to winner_analysis.json

    Returns:
        dict: Bucket percentages
    """
    with open(winner_analysis_path, 'r') as f:
        data = json.load(f)

    bucket_distribution = data['top_100_distribution']
    total_videos = sum(bucket_distribution.values())

    bucket_percentages = {
        bucket: round((count / total_videos) * 100)
        for bucket, count in bucket_distribution.items()
    }

    return bucket_percentages


def rank_top_buckets(winner_analysis_path, base_path):
    """
    Rank top 3 buckets by performance metrics.

    Args:
        winner_analysis_path: Path to winner_analysis.json
        base_path: Base path to analysis directory

    Returns:
        list: Ranked bucket data with metrics
    """
    with open(winner_analysis_path, 'r') as f:
        winner_data = json.load(f)

    winning_buckets = winner_data['top_3_buckets']

    bucket_metrics = []

    for bucket in winning_buckets:
        bucket_path = os.path.join(base_path, 'buckets', f'bucket_{bucket}')
        selected_videos_path = os.path.join(bucket_path, 'selected_videos.json')

        with open(selected_videos_path, 'r') as f:
            data = json.load(f)

        top_count = data['top_count']
        top_videos = data['videos'][:top_count]

        # Calculate metrics
        total_views = sum(v['playCount'] for v in top_videos)
        total_engagement = sum(calculate_engagement_metrics(v) for v in top_videos)
        avg_views = int(total_views / len(top_videos)) if top_videos else 0
        avg_engagement = round(total_engagement / len(top_videos), 1) if top_videos else 0.0

        bucket_metrics.append({
            'bucket': bucket,
            'avg_views': avg_views,
            'avg_engagement': avg_engagement,
            'video_count': len(top_videos)
        })

    # Calculate composite scores and rank
    if bucket_metrics:
        max_views = max(b['avg_views'] for b in bucket_metrics)

        for bucket_data in bucket_metrics:
            normalized_views = (bucket_data['avg_views'] / max_views) * 100 if max_views > 0 else 0
            composite_score = normalized_views + bucket_data['avg_engagement']
            bucket_data['composite_score'] = composite_score

        # Sort by composite score DESC
        bucket_metrics.sort(key=lambda b: b['composite_score'], reverse=True)

        # Assign ranks and stars
        for idx, bucket_data in enumerate(bucket_metrics, start=1):
            bucket_data['rank'] = idx
            bucket_data['stars'] = '⭐' * (4 - idx)  # Rank 1 = 3 stars, Rank 2 = 2 stars, Rank 3 = 1 star
            bucket_data['is_sweet_spot'] = (idx == 1)

    return bucket_metrics


def format_views(view_count):
    """Format view count with K or M suffix."""
    if view_count >= 1000000:
        return f"{view_count / 1000000:.1f}M"
    elif view_count >= 1000:
        return f"{int(view_count / 1000)}K"
    else:
        return str(view_count)


# =============================
# MAIN EXTRACTION WORKFLOW
# =============================

def main():
    """Main extraction workflow"""

    # =============================
    # STEP 1: Parse CLI Arguments
    # =============================
    parser = argparse.ArgumentParser(description='Extract Report 1: Client Executive Report')
    parser.add_argument('--client', required=True, help='Client ID (e.g., rollo_test5)')
    parser.add_argument('--hashtag', required=True, help='Hashtag name (e.g., wellnesspt2_test5)')
    parser.add_argument('--mode', default='top', help='Mode (default: top)')
    parser.add_argument('--strategy', default='contrastive', help='Strategy (default: contrastive)')
    args = parser.parse_args()

    print(f"\n📊 Extracting Client Executive Report for #{args.hashtag}")

    # =============================
    # STEP 2: Build File Paths
    # =============================
    base_path = f"/home/jorge/rumiaifinal/data/clients/{args.client}/hashtags/{args.hashtag}/{args.mode}_{args.strategy}"

    if not os.path.exists(base_path):
        print(f"❌ Error: Analysis directory not found: {base_path}")
        return

    winner_analysis_path = os.path.join(base_path, 'winner_analysis.json')

    # =============================
    # STEP 3: Load Winning Buckets
    # =============================
    with open(winner_analysis_path, 'r') as f:
        winner_data = json.load(f)

    winning_buckets = winner_data['top_3_buckets']
    coverage_pct = winner_data['winner_coverage']

    print(f"✓ Winning buckets: {', '.join(winning_buckets)}")
    print(f"✓ Coverage: {coverage_pct:.1f}%")

    # =============================
    # STEP 4: Load Total Scraped Videos from Cluster Analytics
    # =============================
    # cluster_analytics.json is at the hashtag level (not in top_contrastive)
    cluster_analytics_path = f"/home/jorge/rumiaifinal/data/clients/{args.client}/hashtag/{args.hashtag}/cluster_analytics.json"

    with open(cluster_analytics_path, 'r') as f:
        cluster_data = json.load(f)

    total_scraped_videos = cluster_data['scrape_summary']['total_scraped_videos']

    print(f"✓ Total videos scraped: {total_scraped_videos}")

    # =============================
    # STEP 5: Calculate Bucket Distribution
    # =============================
    print(f"\n📐 Calculating bucket distribution...")
    bucket_percentages = calculate_bucket_distribution(winner_analysis_path)

    # Identify primary focus bucket
    primary_focus = max(bucket_percentages, key=bucket_percentages.get)

    # =============================
    # STEP 6: Rank Top Buckets by Performance
    # =============================
    print(f"🏆 Ranking top buckets by performance...")
    ranked_buckets = rank_top_buckets(winner_analysis_path, base_path)
    sweet_spot = ranked_buckets[0]['bucket'] if ranked_buckets else None

    # =============================
    # STEP 7: Aggregate Content Intelligence (ALL BUCKETS)
    # =============================
    print(f"\n🎯 Aggregating content intelligence across all winning buckets...")

    # Combine Counters from all buckets
    all_content_categories = Counter()
    all_hook_strategies = Counter()
    all_closing_strategies = Counter()
    all_pain_points = Counter()
    all_keywords = Counter()
    all_engagement_drivers = Counter()
    all_content_tactics = Counter()

    for bucket in winning_buckets:
        aggregated = aggregate_content_classifications(bucket, base_path, performer_type="top")

        if aggregated:
            all_content_categories.update(aggregated['content_category'])
            all_hook_strategies.update(aggregated['hook_strategy'])
            all_closing_strategies.update(aggregated['closing_strategy'])
            all_pain_points.update(aggregated['pain_points'])
            all_keywords.update(aggregated['keywords'])
            all_engagement_drivers.update(aggregated['engagement_drivers'])
            all_content_tactics.update(aggregated['content_tactics'])

    total_classified = sum(all_content_categories.values())

    print(f"✓ Aggregated {total_classified} content classifications")

    # =============================
    # STEP 8: Extract Creative Formula Names
    # =============================
    print(f"\n🎨 Extracting creative formulas...")

    formula_names = []
    for bucket in winning_buckets:
        bucket_path = os.path.join(base_path, 'buckets', f'bucket_{bucket}')
        winning_formulas_path = os.path.join(bucket_path, 'ml_analysis', 'llm', 'winning_formulas.json')

        if os.path.exists(winning_formulas_path):
            with open(winning_formulas_path, 'r') as f:
                formulas = json.load(f)

            creative_reports = formulas.get('creative_reports', [])

            for report in creative_reports[:3]:  # Top 3 per bucket
                formula_names.append({
                    'bucket': bucket,
                    'name': report['formula_name']
                })

    print(f"✓ Found {len(formula_names)} creative formulas")

    # =============================
    # STEP 9: Build Excel Data Structure
    # =============================
    print(f"\n📝 Building Excel data structure...")

    tab_data = []

    # PAGE 1: SCALE OF ANALYSIS
    tab_data.append(['PAGE_1_SCALE_OF_ANALYSIS', ''])
    tab_data.append(['', ''])

    tab_data.append(['HASHTAG', f'#{args.hashtag.replace("_test5", "").replace("pt2", "")}'])
    tab_data.append(['ANALYSIS_PERIOD', 'Last 90 days'])
    tab_data.append(['TOTAL_VIDEOS_ANALYZED', str(total_scraped_videos)])
    tab_data.append(['WINNING_BUCKETS_COUNT', str(len(winning_buckets))])
    tab_data.append(['COVERAGE_PERCENTAGE', str(round(coverage_pct, 1))])
    tab_data.append(['', ''])

    # Winning buckets list
    for i, bucket in enumerate(winning_buckets, 1):
        tab_data.append([f'WINNING_BUCKET_{i}', bucket])

    # PAGE 2: HASHTAG INTELLIGENCE DASHBOARD
    tab_data.append(['', ''])
    tab_data.append(['PAGE_2_HASHTAG_INTELLIGENCE', ''])
    tab_data.append(['', ''])

    # Duration Distribution (all 8 buckets)
    all_buckets = ["0-3s", "3-9s", "9-13s", "13-18s", "18-33s", "33-60s", "60-90s", "90-120s"]
    for bucket in all_buckets:
        field_name = f'BUCKET_{bucket.replace("-", "_").upper()}_PCT'
        pct = bucket_percentages.get(bucket, 0)
        tab_data.append([field_name, str(pct)])

    tab_data.append(['', ''])
    tab_data.append(['PRIMARY_FOCUS_BUCKET', primary_focus])

    # Performance by Duration (ranked top 3)
    tab_data.append(['', ''])
    for bucket_data in ranked_buckets:
        rank = bucket_data['rank']
        tab_data.append([f'PERF_BUCKET_{rank}_NAME', bucket_data['bucket']])
        tab_data.append([f'PERF_BUCKET_{rank}_AVG_VIEWS', format_views(bucket_data['avg_views'])])
        tab_data.append([f'PERF_BUCKET_{rank}_AVG_ENG', str(bucket_data['avg_engagement'])])
        tab_data.append([f'PERF_BUCKET_{rank}_STARS', bucket_data['stars']])
        tab_data.append([f'PERF_BUCKET_{rank}_IS_SWEET_SPOT', str(bucket_data['is_sweet_spot'])])
        tab_data.append(['', ''])

    tab_data.append(['SWEET_SPOT_BUCKET', sweet_spot])

    # Content Intelligence (aggregated across all buckets)
    tab_data.append(['', ''])

    # Top 5 content categories
    for i, (category, count) in enumerate(all_content_categories.most_common(5), 1):
        pct = round((count / total_classified) * 100) if total_classified > 0 else 0
        tab_data.append([f'CONTENT_CATEGORY_{i}', category.replace('_', ' ').title()])
        tab_data.append([f'CONTENT_CATEGORY_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # Top 4 engagement drivers
    for i, (driver, count) in enumerate(all_engagement_drivers.most_common(4), 1):
        pct = round((count / total_classified) * 100) if total_classified > 0 else 0
        tab_data.append([f'ENGAGEMENT_DRIVER_{i}', driver.replace('_', ' ').title()])
        tab_data.append([f'ENGAGEMENT_DRIVER_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # Top 4 hook strategies
    for i, (hook, count) in enumerate(all_hook_strategies.most_common(4), 1):
        pct = round((count / total_classified) * 100) if total_classified > 0 else 0
        tab_data.append([f'HOOK_STRATEGY_{i}', hook.replace('_', ' ').title()])
        tab_data.append([f'HOOK_STRATEGY_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # Top 5 pain points
    for i, (pain, count) in enumerate(all_pain_points.most_common(5), 1):
        tab_data.append([f'PAIN_POINT_{i}', pain.replace('_', ' ').title()])

    tab_data.append(['', ''])

    # Top 5 keywords
    for i, (keyword, count) in enumerate(all_keywords.most_common(5), 1):
        tab_data.append([f'KEYWORD_{i}', f'#{keyword}'])

    # PAGE 3: YOUR CREATIVE REPORTS
    tab_data.append(['', ''])
    tab_data.append(['PAGE_3_YOUR_CREATIVE_REPORTS', ''])
    tab_data.append(['', ''])

    # Formula names (9 total: 3 per bucket × 3 buckets)
    formula_idx = 1
    for formula in formula_names:
        tab_data.append([f'FORMULA_{formula_idx}_BUCKET', formula['bucket']])
        tab_data.append([f'FORMULA_{formula_idx}_NAME', formula['name']])
        formula_idx += 1

    # =============================
    # STEP 10: Write Excel File
    # =============================
    excel_filename = f"{args.hashtag}_client_data.xlsx"
    excel_path = os.path.join(base_path, excel_filename)

    print(f"\n💾 Writing Excel file...")

    df = pd.DataFrame(tab_data, columns=['Field Name', 'Value'])
    df.to_excel(excel_path, sheet_name='Report_Data', index=False, engine='openpyxl')

    # =============================
    # STEP 11: Print Success Message
    # =============================
    print(f"\n✅ Extraction complete!")
    print(f"  📁 Excel: {excel_path}")
    print(f"  📊 Total fields: {len(tab_data)}")
    print(f"  🎨 Creative formulas: {len(formula_names)}")


if __name__ == '__main__':
    main()
