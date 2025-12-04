#!/usr/bin/env python3
"""
extract_creator_data.py - Report 2: Hashtag → Content Creator

Generates per-bucket creative reports for content creators with actionable formulas.

Usage:
    python extract_creator_data.py --client rollo_test5 --hashtag wellnesspt2_test5 --mode top --strategy contrastive

Output:
    - Excel file with 3 tabs (one per winning bucket)
    - 6 QR codes (2 per bucket: 1 top + 1 bottom)
"""

import argparse
import json
import os
import pandas as pd
from collections import Counter
from pathlib import Path


def normalize_classification_key(value: str) -> str:
    """
    Normalize classification values to snake_case for consistent aggregation.

    Handles LLM output inconsistency where the same category appears as both:
    - "explaining scientific mechanisms" (space-separated)
    - "explaining_scientific_mechanisms" (snake_case)
    """
    if not value:
        return value
    return value.strip().lower().replace(' ', '_')


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
                pain_points[normalize_classification_key(pain_point)] += 1

        for keyword in data.get('keywords', []):
            if keyword:
                keywords[normalize_classification_key(keyword)] += 1

        for driver in data.get('engagement_drivers', []):
            if driver:
                engagement_drivers[normalize_classification_key(driver)] += 1

        for tactic in data.get('content_tactics', []):
            if tactic:
                content_tactics[normalize_classification_key(tactic)] += 1

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
        'content_tactics': content_tactics
    }


def select_qr_code_videos(bucket_path, performer_type="top", count=1):
    """
    Select top/bottom performing videos for QR code generation.

    Args:
        bucket_path: Path to bucket directory
        performer_type: "top" or "bottom"
        count: Number of videos to select (default 1)

    Returns:
        list: Video metadata dicts with url, views, engagement, duration, bucket
    """
    selected_videos_path = os.path.join(bucket_path, 'selected_videos.json')

    with open(selected_videos_path, 'r') as f:
        data = json.load(f)

    bucket_name = data['bucket']

    # Select from top or bottom performers
    if performer_type == "top":
        top_count = data['top_count']
        videos = data['videos'][:top_count]
    else:
        top_count = data['top_count']
        videos = data['videos'][top_count:]

    # Sort by engagement and select top N
    video_list = []
    for video in videos[:count]:
        engagement = calculate_engagement_metrics(video)
        video_list.append({
            'video_id': video['id'],
            'url': video['webVideoUrl'],
            'views': video['playCount'],
            'engagement': engagement,
            'duration': video['videoMeta']['duration'],
            'bucket': bucket_name,
            'hashtags': video.get('hashtags', [])
        })

    return video_list


def generate_qr_codes(qr_data_list, output_dir):
    """
    Generate QR code PNG files from TikTok URLs.

    Args:
        qr_data_list: List of dicts with 'filename' and 'url' keys
        output_dir: Directory to save QR codes
    """
    import qrcode

    os.makedirs(output_dir, exist_ok=True)

    for qr_data in qr_data_list:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_data['url'])
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        output_path = os.path.join(output_dir, qr_data['filename'])
        img.save(output_path)

        print(f"✓ Generated QR code: {qr_data['filename']}")


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
    parser = argparse.ArgumentParser(description='Extract Report 2: Creator Report')
    parser.add_argument('--client', required=True, help='Client ID (e.g., rollo_test5)')
    parser.add_argument('--hashtag', required=True, help='Hashtag name (e.g., wellnesspt2_test5)')
    parser.add_argument('--mode', default='top', help='Mode (default: top)')
    parser.add_argument('--strategy', default='contrastive', help='Strategy (default: contrastive)')
    args = parser.parse_args()

    print(f"\n🎬 Extracting Creator Report for #{args.hashtag}")

    # =============================
    # STEP 2: Build File Paths
    # =============================
    analysis_base_path = f"/home/jorge/rumiaifinal/data/clients/{args.client}/hashtags/{args.hashtag}/{args.mode}_{args.strategy}"

    if not os.path.exists(analysis_base_path):
        print(f"❌ Error: Analysis directory not found: {analysis_base_path}")
        return

    # Create reports/creator/ directory structure
    reports_base_path = os.path.join(analysis_base_path, 'reports', 'creator')
    os.makedirs(reports_base_path, exist_ok=True)

    winner_analysis_path = os.path.join(analysis_base_path, 'winner_analysis.json')

    # =============================
    # STEP 3: Load Winning Buckets & Total Scraped Videos
    # =============================
    with open(winner_analysis_path, 'r') as f:
        winner_data = json.load(f)

    winning_buckets = winner_data['top_3_buckets']
    bucket_distribution = winner_data['top_100_distribution']  # Percentages per bucket

    # Load total scraped videos from cluster_analytics
    cluster_analytics_path = f"/home/jorge/rumiaifinal/data/clients/{args.client}/hashtag/{args.hashtag}/cluster_analytics.json"
    with open(cluster_analytics_path, 'r') as f:
        cluster_data = json.load(f)
    total_scraped_videos = cluster_data['scrape_summary']['total_scraped_videos']

    print(f"✓ Winning buckets: {', '.join(winning_buckets)}")

    # =============================
    # STEP 4: Generate QR Codes (12 total: 4 per bucket - 2 top + 2 bottom)
    # =============================
    print(f"\n📱 Generating QR codes...")

    qr_output_dir = os.path.join(reports_base_path, 'qr_codes')
    qr_data_list = []
    qr_metadata = {}  # Store metadata for Excel

    for bucket in winning_buckets:
        bucket_path = os.path.join(analysis_base_path, 'buckets', f'bucket_{bucket}')

        # Top performers (2)
        top_videos = select_qr_code_videos(bucket_path, "top", count=2)
        for idx, top_video in enumerate(top_videos, 1):
            qr_data_list.append({
                'filename': f"{args.hashtag}_{bucket}_top{idx}.png",
                'url': top_video['url']
            })
            qr_metadata[f"{bucket}_top{idx}"] = top_video

        # Bottom performers (2)
        bottom_videos = select_qr_code_videos(bucket_path, "bottom", count=2)
        for idx, bottom_video in enumerate(bottom_videos, 1):
            qr_data_list.append({
                'filename': f"{args.hashtag}_{bucket}_bottom{idx}.png",
                'url': bottom_video['url']
            })
            qr_metadata[f"{bucket}_bottom{idx}"] = bottom_video

    generate_qr_codes(qr_data_list, qr_output_dir)

    # =============================
    # STEP 4.5: Calculate Performance Metrics for ALL Winning Buckets
    # =============================
    print(f"\n📊 Calculating performance metrics for all winning buckets...")

    bucket_metrics = {}  # Store metrics for all 3 buckets
    for bucket in winning_buckets:
        bucket_path = os.path.join(analysis_base_path, 'buckets', f'bucket_{bucket}')
        selected_videos_path = os.path.join(bucket_path, 'selected_videos.json')

        with open(selected_videos_path, 'r') as f:
            data = json.load(f)

        top_count = data['top_count']
        top_videos = data['videos'][:top_count]

        # Calculate average views and engagement for top performers
        avg_views = int(sum(v['playCount'] for v in top_videos) / len(top_videos)) if top_videos else 0
        avg_engagement = round(sum(calculate_engagement_metrics(v) for v in top_videos) / len(top_videos), 1) if top_videos else 0.0

        bucket_metrics[bucket] = {
            'avg_views': avg_views,
            'avg_engagement': avg_engagement
        }

    # Rank buckets by engagement (primary), then views (secondary)
    ranked_buckets = sorted(
        bucket_metrics.items(),
        key=lambda x: (x[1]['avg_engagement'], x[1]['avg_views']),
        reverse=True
    )

    # Assign star ratings and labels
    star_ratings = {
        ranked_buckets[0][0]: '⭐⭐⭐⭐⭐',
        ranked_buckets[1][0]: '⭐⭐⭐⭐',
        ranked_buckets[2][0]: '⭐⭐⭐'
    }
    best_bucket = ranked_buckets[0][0]

    # Calculate coverage percentage
    coverage_pct = sum(bucket_distribution.get(b, 0) for b in winning_buckets)

    # =============================
    # STEP 5: Extract Data Per Bucket (3 Tabs)
    # =============================
    print(f"\n📊 Extracting data for {len(winning_buckets)} buckets...")

    excel_data = {}  # Dict of {bucket_name: DataFrame}

    for bucket_idx, bucket in enumerate(winning_buckets, 1):
        print(f"\n  Processing bucket {bucket_idx}/3: {bucket}")

        bucket_path = os.path.join(analysis_base_path, 'buckets', f'bucket_{bucket}')

        # Load selected videos
        selected_videos_path = os.path.join(bucket_path, 'selected_videos.json')
        with open(selected_videos_path, 'r') as f:
            selected_data = json.load(f)

        top_count = selected_data['top_count']
        bottom_count = selected_data['bottom_count']
        top_videos = selected_data['videos'][:top_count]
        bottom_videos = selected_data['videos'][top_count:top_count + bottom_count]

        # Calculate performance metrics
        total_views = sum(v['playCount'] for v in top_videos)
        total_engagement = sum(calculate_engagement_metrics(v) for v in top_videos)
        avg_views = int(total_views / len(top_videos)) if top_videos else 0
        avg_engagement = round(total_engagement / len(top_videos), 1) if top_videos else 0.0

        # Calculate "THE PROOF" metrics for this bucket
        # Top Cluster: videos ranked #5-#25 (indices 4-24, = 21 videos)
        top_cluster_videos = top_videos[4:25] if len(top_videos) >= 25 else top_videos[4:]
        top_cluster_avg_views = int(sum(v['playCount'] for v in top_cluster_videos) / len(top_cluster_videos)) if top_cluster_videos else 0
        top_cluster_avg_engagement = round(sum(calculate_engagement_metrics(v) for v in top_cluster_videos) / len(top_cluster_videos), 1) if top_cluster_videos else 0.0

        # Bottom Cluster: bottom 20 videos
        bottom_cluster_videos = bottom_videos[:20] if len(bottom_videos) >= 20 else bottom_videos
        bottom_cluster_avg_views = int(sum(v['playCount'] for v in bottom_cluster_videos) / len(bottom_cluster_videos)) if bottom_cluster_videos else 0
        bottom_cluster_avg_engagement = round(sum(calculate_engagement_metrics(v) for v in bottom_cluster_videos) / len(bottom_cluster_videos), 1) if bottom_cluster_videos else 0.0

        # Calculate multipliers and percentage increases
        view_multiplier = round(top_cluster_avg_views / bottom_cluster_avg_views, 1) if bottom_cluster_avg_views > 0 else 0
        engagement_multiplier = round(top_cluster_avg_engagement / bottom_cluster_avg_engagement, 1) if bottom_cluster_avg_engagement > 0 else 0
        view_pct_increase = int(((top_cluster_avg_views - bottom_cluster_avg_views) / bottom_cluster_avg_views) * 100) if bottom_cluster_avg_views > 0 else 0
        engagement_pct_increase = int(((top_cluster_avg_engagement - bottom_cluster_avg_engagement) / bottom_cluster_avg_engagement) * 100) if bottom_cluster_avg_engagement > 0 else 0

        # Aggregate content classifications (top performers only)
        aggregated = aggregate_content_classifications(bucket, analysis_base_path, performer_type="top")

        if not aggregated:
            print(f"  ⚠️  Skipping {bucket} - no content classifications found")
            continue

        # Load winning formulas from Stage 7
        winning_formulas_path = os.path.join(bucket_path, 'ml_analysis', 'llm', 'winning_formulas.json')

        if not os.path.exists(winning_formulas_path):
            print(f"  ⚠️  Warning: winning_formulas.json not found for {bucket}")
            creative_reports = []
            supplementary_insights = []
        else:
            with open(winning_formulas_path, 'r') as f:
                formulas = json.load(f)
            creative_reports = formulas.get('creative_reports', [])
            supplementary_insights = formulas.get('supplementary_insights', {}).get('universal_principles', [])

        # Get QR code metadata (2 top + 2 bottom)
        qr_top1 = qr_metadata.get(f"{bucket}_top1", {})
        qr_top2 = qr_metadata.get(f"{bucket}_top2", {})
        qr_bottom1 = qr_metadata.get(f"{bucket}_bottom1", {})
        qr_bottom2 = qr_metadata.get(f"{bucket}_bottom2", {})

        # =============================
        # BUILD TAB DATA (2-column format)
        # =============================
        tab_data = []

        # PAGE 1: WHY THIS WORKS
        tab_data.append(['PAGE_1_WHY_THIS_WORKS', ''])
        tab_data.append(['', ''])

        tab_data.append(['BUCKET', bucket])
        tab_data.append(['VIDEOS_ANALYZED', str(total_scraped_videos)])
        tab_data.append(['AVG_VIEWS', format_views(avg_views)])
        tab_data.append(['AVG_ENGAGEMENT', str(avg_engagement)])
        tab_data.append(['', ''])

        # Hashtag top 3 Performing Durations comparison table
        for i, winning_bucket in enumerate(winning_buckets, 1):
            metrics = bucket_metrics[winning_bucket]
            tab_data.append([f'BUCKET_{i}_DURATION', winning_bucket])
            tab_data.append([f'BUCKET_{i}_AVG_VIEWS', format_views(metrics['avg_views'])])
            tab_data.append([f'BUCKET_{i}_AVG_ENGAGEMENT', str(metrics['avg_engagement'])])
            tab_data.append([f'BUCKET_{i}_RATING', star_ratings[winning_bucket]])
            tab_data.append([f'BUCKET_{i}_LABEL', '← BEST' if winning_bucket == best_bucket else ''])
            tab_data.append(['', ''])

        tab_data.append(['COVERAGE_PERCENTAGE', f"{coverage_pct}%"])
        tab_data.append(['', ''])

        # THE PROOF - Performance Comparison (bucket-specific)
        tab_data.append(['TOP_CLUSTER_AVG_VIEWS', format_views(top_cluster_avg_views)])
        tab_data.append(['TOP_CLUSTER_AVG_ENGAGEMENT', str(top_cluster_avg_engagement)])
        tab_data.append(['BOTTOM_CLUSTER_AVG_VIEWS', format_views(bottom_cluster_avg_views)])
        tab_data.append(['BOTTOM_CLUSTER_AVG_ENGAGEMENT', str(bottom_cluster_avg_engagement)])
        tab_data.append(['', ''])
        tab_data.append(['VIEW_MULTIPLIER', f"{view_multiplier}x"])
        tab_data.append(['ENGAGEMENT_MULTIPLIER', f"{engagement_multiplier}x"])
        tab_data.append(['VIEW_PERCENTAGE_INCREASE', f"{view_pct_increase}%"])
        tab_data.append(['ENGAGEMENT_PERCENTAGE_INCREASE', f"{engagement_pct_increase}%"])
        tab_data.append(['', ''])

        # Top content categories
        for i, (category, count) in enumerate(aggregated['content_category'].most_common(5), 1):
            pct = round((count / len(top_videos)) * 100)
            tab_data.append([f'CONTENT_CATEGORY_{i}', category.replace('_', ' ').title()])
            tab_data.append([f'CONTENT_CATEGORY_{i}_PCT', str(pct)])

        tab_data.append(['', ''])

        # Top engagement drivers
        for i, (driver, count) in enumerate(aggregated['engagement_drivers'].most_common(4), 1):
            pct = round((count / len(top_videos)) * 100)
            tab_data.append([f'ENGAGEMENT_DRIVER_{i}', driver.replace('_', ' ').title()])
            tab_data.append([f'ENGAGEMENT_DRIVER_{i}_PCT', str(pct)])

        tab_data.append(['', ''])

        # Top hook strategies
        for i, (hook, count) in enumerate(aggregated['hook_strategy'].most_common(4), 1):
            pct = round((count / len(top_videos)) * 100)
            tab_data.append([f'HOOK_STRATEGY_{i}', hook.replace('_', ' ').title()])
            tab_data.append([f'HOOK_STRATEGY_{i}_PCT', str(pct)])

        tab_data.append(['', ''])

        # Top pain points
        for i, (pain, count) in enumerate(aggregated['pain_points'].most_common(5), 1):
            tab_data.append([f'PAIN_POINT_{i}', pain.replace('_', ' ').title()])

        tab_data.append(['', ''])

        # Top keywords
        for i, (keyword, count) in enumerate(aggregated['keywords'].most_common(5), 1):
            tab_data.append([f'KEYWORD_{i}', keyword])

        tab_data.append(['', ''])

        # Top content tactics
        for i, (tactic, count) in enumerate(aggregated['content_tactics'].most_common(4), 1):
            pct = round((count / len(top_videos)) * 100)
            tab_data.append([f'CONTENT_TACTIC_{i}', tactic.replace('_', ' ').title()])
            tab_data.append([f'CONTENT_TACTIC_{i}_PCT', str(pct)])

        tab_data.append(['', ''])

        # Supplementary insights (top 5) - RIGHT AFTER content tactics
        for i, insight in enumerate(supplementary_insights[:5], 1):
            tab_data.append([f'SUPPLEMENTARY_INSIGHT_{i}', insight])

        # PAGE 2: HOW TO EXECUTE
        tab_data.append(['', ''])
        tab_data.append(['PAGE_2_HOW_TO_EXECUTE', ''])
        tab_data.append(['', ''])

        # Creative formulas (up to 3)
        for i, report in enumerate(creative_reports[:3], 1):
            tab_data.append([f'FORMULA_{i}_NAME', report['formula_name']])
            tab_data.append([f'FORMULA_{i}_STRATEGY', report['strategy_description']])
            tab_data.append([f'FORMULA_{i}_WHEN_TO_USE', report['when_to_use']])

            # Step-by-step template
            for j, step in enumerate(report['step_by_step_template'], 1):
                tab_data.append([f'FORMULA_{i}_STEP_{j}', step])

            tab_data.append(['', ''])

        # QR code metadata (2 top + 2 bottom)
        tab_data.append(['', ''])

        # Top Performer #1
        if qr_top1:
            tab_data.append(['QR_TOP1_FILE', f"{args.hashtag}_{bucket}_top1.png"])
            tab_data.append(['QR_TOP1_URL', qr_top1.get('url', '')])
            tab_data.append(['QR_TOP1_VIEWS', format_views(qr_top1.get('views', 0))])
            tab_data.append(['QR_TOP1_ENGAGEMENT', str(qr_top1.get('engagement', 0))])
            tab_data.append(['', ''])

        # Top Performer #2
        if qr_top2:
            tab_data.append(['QR_TOP2_FILE', f"{args.hashtag}_{bucket}_top2.png"])
            tab_data.append(['QR_TOP2_URL', qr_top2.get('url', '')])
            tab_data.append(['QR_TOP2_VIEWS', format_views(qr_top2.get('views', 0))])
            tab_data.append(['QR_TOP2_ENGAGEMENT', str(qr_top2.get('engagement', 0))])
            tab_data.append(['', ''])

        # Bottom Performer #1
        if qr_bottom1:
            tab_data.append(['QR_BOTTOM1_FILE', f"{args.hashtag}_{bucket}_bottom1.png"])
            tab_data.append(['QR_BOTTOM1_URL', qr_bottom1.get('url', '')])
            tab_data.append(['QR_BOTTOM1_VIEWS', format_views(qr_bottom1.get('views', 0))])
            tab_data.append(['QR_BOTTOM1_ENGAGEMENT', str(qr_bottom1.get('engagement', 0))])
            tab_data.append(['', ''])

        # Bottom Performer #2
        if qr_bottom2:
            tab_data.append(['QR_BOTTOM2_FILE', f"{args.hashtag}_{bucket}_bottom2.png"])
            tab_data.append(['QR_BOTTOM2_URL', qr_bottom2.get('url', '')])
            tab_data.append(['QR_BOTTOM2_VIEWS', format_views(qr_bottom2.get('views', 0))])
            tab_data.append(['QR_BOTTOM2_ENGAGEMENT', str(qr_bottom2.get('engagement', 0))])

        # Create DataFrame for this bucket
        df = pd.DataFrame(tab_data, columns=['Field Name', 'Value'])
        excel_data[bucket] = df

        print(f"  ✓ Extracted {len(tab_data)} fields for {bucket}")

    # =============================
    # STEP 6: Write Excel File (3 Tabs)
    # =============================
    excel_filename = f"{args.hashtag}_creator_data.xlsx"
    excel_path = os.path.join(reports_base_path, excel_filename)

    print(f"\n💾 Writing Excel file...")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for bucket, df in excel_data.items():
            # Clean bucket name for sheet name (Excel limit: 31 chars, no special chars)
            sheet_name = f"Bucket_{bucket}".replace('-', '_')[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  ✓ Created tab: {sheet_name}")

    # =============================
    # STEP 7: Print Success Message
    # =============================
    print(f"\n✅ Extraction complete!")
    print(f"  📁 Excel: {excel_path}")
    print(f"  📁 QR codes: {qr_output_dir} ({len(qr_data_list)} files)")
    print(f"  📊 Tabs created: {len(excel_data)}")


if __name__ == '__main__':
    main()
