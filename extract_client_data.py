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
import qrcode
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

    # Caption analysis fields
    caption_hook_types = Counter()
    caption_cta_types = Counter()
    hashtag_counts = []

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

        # Aggregate caption_analysis fields
        caption_analysis = data.get('caption_analysis', {})
        if caption_analysis.get('hook_type'):
            caption_hook_types[caption_analysis['hook_type']] += 1
        if caption_analysis.get('cta_type'):
            caption_cta_types[caption_analysis['cta_type']] += 1
        if 'hashtag_count' in caption_analysis:
            hashtag_counts.append(caption_analysis['hashtag_count'])

    if files_processed == 0:
        return None

    return {
        'content_category': content_categories,
        'hook_strategy': hook_strategies,
        'closing_strategy': closing_strategies,
        'pain_points': pain_points,
        'keywords': keywords,
        'engagement_drivers': engagement_drivers,
        'content_tactics': content_tactics,
        'caption_hook_type': caption_hook_types,
        'caption_cta_type': caption_cta_types,
        'hashtag_counts': hashtag_counts
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
    analysis_base_path = f"/home/jorge/rumiaifinal/data/clients/{args.client}/hashtags/{args.hashtag}/{args.mode}_{args.strategy}"

    if not os.path.exists(analysis_base_path):
        print(f"❌ Error: Analysis directory not found: {analysis_base_path}")
        return

    # Create reports/client/ directory structure
    reports_base_path = os.path.join(analysis_base_path, 'reports', 'client')
    os.makedirs(reports_base_path, exist_ok=True)

    winner_analysis_path = os.path.join(analysis_base_path, 'winner_analysis.json')

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
    ranked_buckets = rank_top_buckets(winner_analysis_path, analysis_base_path)
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
    all_caption_hook_types = Counter()
    all_caption_cta_types = Counter()
    all_hashtag_counts = []

    for bucket in winning_buckets:
        aggregated = aggregate_content_classifications(bucket, analysis_base_path, performer_type="top")

        if aggregated:
            all_content_categories.update(aggregated['content_category'])
            all_hook_strategies.update(aggregated['hook_strategy'])
            all_closing_strategies.update(aggregated['closing_strategy'])
            all_pain_points.update(aggregated['pain_points'])
            all_keywords.update(aggregated['keywords'])
            all_engagement_drivers.update(aggregated['engagement_drivers'])
            all_content_tactics.update(aggregated['content_tactics'])
            all_caption_hook_types.update(aggregated['caption_hook_type'])
            all_caption_cta_types.update(aggregated['caption_cta_type'])
            all_hashtag_counts.extend(aggregated['hashtag_counts'])

    total_classified = sum(all_content_categories.values())

    print(f"✓ Aggregated {total_classified} content classifications")

    # =============================
    # STEP 8: Extract Creative Formula Names
    # =============================
    print(f"\n🎨 Extracting creative formulas...")

    formula_names = []
    for bucket in winning_buckets:
        bucket_path = os.path.join(analysis_base_path, 'buckets', f'bucket_{bucket}')
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
    # STEP 8.5: Generate QR Codes (6 total: 2 per winning bucket - top + bottom)
    # =============================
    print(f"\n📱 Generating QR codes...")

    qr_output_dir = os.path.join(reports_base_path, 'qr_codes')
    os.makedirs(qr_output_dir, exist_ok=True)

    qr_data_list = []
    qr_metadata = {}  # Store metadata for Excel

    for bucket in winning_buckets:
        bucket_path = os.path.join(analysis_base_path, 'buckets', f'bucket_{bucket}')

        # Select top and bottom performers from this bucket
        selected_videos_path = os.path.join(bucket_path, 'selected_videos.json')
        with open(selected_videos_path, 'r') as f:
            data = json.load(f)

        top_count = data['top_count']
        bottom_count = data['bottom_count']
        top_videos = data['videos'][:top_count]
        bottom_videos = data['videos'][top_count:top_count + bottom_count]

        # Get #1 top performer (highest views)
        best_top_video = None
        best_top_views = 0

        for video in top_videos:
            views = video['playCount']
            if views > best_top_views:
                best_top_views = views
                best_top_video = video

        # Get #1 bottom performer (lowest views)
        worst_bottom_video = None
        worst_bottom_views = float('inf')

        for video in bottom_videos:
            views = video['playCount']
            if views < worst_bottom_views:
                worst_bottom_views = views
                worst_bottom_video = video

        # Store metadata for Excel (top and bottom)
        qr_metadata[bucket] = {}

        if best_top_video:
            # Add top QR to generation list
            qr_data_list.append({
                'filename': f"{args.hashtag}_{bucket}_top.png",
                'url': best_top_video['webVideoUrl']
            })

            # Store top metadata
            top_engagement = calculate_engagement_metrics(best_top_video)
            qr_metadata[bucket]['top'] = {
                'video_id': best_top_video['id'],
                'url': best_top_video['webVideoUrl'],
                'views': best_top_video['playCount'],
                'engagement': top_engagement,
                'duration': best_top_video['videoMeta']['duration']
            }

        if worst_bottom_video:
            # Add bottom QR to generation list
            qr_data_list.append({
                'filename': f"{args.hashtag}_{bucket}_bottom.png",
                'url': worst_bottom_video['webVideoUrl']
            })

            # Store bottom metadata
            bottom_engagement = calculate_engagement_metrics(worst_bottom_video)
            qr_metadata[bucket]['bottom'] = {
                'video_id': worst_bottom_video['id'],
                'url': worst_bottom_video['webVideoUrl'],
                'views': worst_bottom_video['playCount'],
                'engagement': bottom_engagement,
                'duration': worst_bottom_video['videoMeta']['duration']
            }

    # Generate QR codes
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
        output_path = os.path.join(qr_output_dir, qr_data['filename'])
        img.save(output_path)

        print(f"✓ Generated QR code: {qr_data['filename']}")

    print(f"✓ Generated {len(qr_data_list)} QR codes")

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

    # Top 4 hook strategies
    for i, (hook, count) in enumerate(all_hook_strategies.most_common(4), 1):
        pct = round((count / total_classified) * 100) if total_classified > 0 else 0
        tab_data.append([f'HOOK_STRATEGY_{i}', hook.replace('_', ' ').title()])
        tab_data.append([f'HOOK_STRATEGY_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # Top 4 closing strategies
    for i, (closing, count) in enumerate(all_closing_strategies.most_common(4), 1):
        pct = round((count / total_classified) * 100) if total_classified > 0 else 0
        tab_data.append([f'CLOSING_STRATEGY_{i}', closing.replace('_', ' ').title()])
        tab_data.append([f'CLOSING_STRATEGY_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # Top 4 engagement drivers
    for i, (driver, count) in enumerate(all_engagement_drivers.most_common(4), 1):
        pct = round((count / total_classified) * 100) if total_classified > 0 else 0
        tab_data.append([f'ENGAGEMENT_DRIVER_{i}', driver.replace('_', ' ').title()])
        tab_data.append([f'ENGAGEMENT_DRIVER_{i}_PCT', str(pct)])

    tab_data.append(['', ''])

    # Top 5 pain points
    for i, (pain, count) in enumerate(all_pain_points.most_common(5), 1):
        tab_data.append([f'PAIN_POINT_{i}', pain.replace('_', ' ').title()])

    tab_data.append(['', ''])

    # Top 5 keywords (without # prefix)
    for i, (keyword, count) in enumerate(all_keywords.most_common(5), 1):
        tab_data.append([f'KEYWORD_{i}', keyword])

    # Caption Analysis Section
    tab_data.append(['', ''])

    total_videos_with_caption = sum(all_caption_cta_types.values())

    if total_videos_with_caption > 0:
        # Caption Hook Type (most common)
        if all_caption_hook_types:
            top_caption_hook, hook_count = all_caption_hook_types.most_common(1)[0]
            caption_hook_pct = round((hook_count / total_videos_with_caption) * 100)
            tab_data.append(['CAPTION_HOOK_TYPE', top_caption_hook.replace('_', ' ').title()])
            tab_data.append(['CAPTION_HOOK_TYPE_PCT', str(caption_hook_pct)])
        else:
            tab_data.append(['CAPTION_HOOK_TYPE', ''])
            tab_data.append(['CAPTION_HOOK_TYPE_PCT', '0'])

        tab_data.append(['', ''])

        # Calculate NO_CTA percentage
        videos_with_no_cta = all_caption_cta_types.get('none', 0)
        no_cta_pct = round((videos_with_no_cta / total_videos_with_caption) * 100)

        # Get top 3 CTAs (excluding "none")
        cta_without_none = Counter({k: v for k, v in all_caption_cta_types.items() if k != 'none'})
        top_3_ctas = list(cta_without_none.most_common(3))

        # Pad to ensure we always have 3 entries
        while len(top_3_ctas) < 3:
            top_3_ctas.append(('', 0))

        # Output top 3 CTAs
        for i, (cta, count) in enumerate(top_3_ctas, 1):
            cta_pct = round((count / total_videos_with_caption) * 100) if count > 0 else 0
            cta_display = cta.replace('_', ' ').title() if cta else ''
            tab_data.append([f'TOP_CTA_{i}', cta_display])
            tab_data.append([f'TOP_CTA_{i}_PCT', str(cta_pct)])

        tab_data.append(['NO_CTA_PCT', str(no_cta_pct)])

    else:
        # No caption analysis data - output zeros
        tab_data.append(['CAPTION_HOOK_TYPE', ''])
        tab_data.append(['CAPTION_HOOK_TYPE_PCT', '0'])
        tab_data.append(['', ''])
        for i in range(1, 4):
            tab_data.append([f'TOP_CTA_{i}', ''])
            tab_data.append([f'TOP_CTA_{i}_PCT', '0'])
        tab_data.append(['NO_CTA_PCT', '0'])

    # Hashtag Statistics
    tab_data.append(['', ''])

    # Calculate NO_HASHTAGS percentage
    hashtag_zeros = all_hashtag_counts.count(0)
    no_hashtags_pct = round((hashtag_zeros / len(all_hashtag_counts)) * 100) if all_hashtag_counts else 0

    # Calculate optimal hashtag count (excluding zeros)
    non_zero_hashtags = [h for h in all_hashtag_counts if h > 0]
    optimal_hashtag_count = round(sum(non_zero_hashtags) / len(non_zero_hashtags)) if non_zero_hashtags else 0

    tab_data.append(['NO_HASHTAGS_PCT', str(no_hashtags_pct)])
    tab_data.append(['OPTIMAL_HASHTAG_COUNT', str(optimal_hashtag_count)])

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

    # PAGE 4: VISUAL EXAMPLES (QR CODES)
    tab_data.append(['', ''])
    tab_data.append(['PAGE_4_VISUAL_EXAMPLES', ''])
    tab_data.append(['', ''])

    # Output QR code metadata for each winning bucket (top + bottom)
    for i, bucket in enumerate(winning_buckets, 1):
        if bucket in qr_metadata:
            tab_data.append([f'QR_BUCKET_{i}_NAME', bucket])
            tab_data.append(['', ''])

            # Top performer QR
            if 'top' in qr_metadata[bucket]:
                qr_top = qr_metadata[bucket]['top']
                tab_data.append([f'QR_BUCKET_{i}_TOP_FILE', f"{args.hashtag}_{bucket}_top.png"])
                tab_data.append([f'QR_BUCKET_{i}_TOP_URL', qr_top['url']])
                tab_data.append([f'QR_BUCKET_{i}_TOP_VIEWS', format_views(qr_top['views'])])
                tab_data.append([f'QR_BUCKET_{i}_TOP_ENGAGEMENT', str(qr_top['engagement'])])
                tab_data.append([f'QR_BUCKET_{i}_TOP_DURATION', f"{qr_top['duration']}s"])
                tab_data.append(['', ''])

            # Bottom performer QR
            if 'bottom' in qr_metadata[bucket]:
                qr_bottom = qr_metadata[bucket]['bottom']
                tab_data.append([f'QR_BUCKET_{i}_BOTTOM_FILE', f"{args.hashtag}_{bucket}_bottom.png"])
                tab_data.append([f'QR_BUCKET_{i}_BOTTOM_URL', qr_bottom['url']])
                tab_data.append([f'QR_BUCKET_{i}_BOTTOM_VIEWS', format_views(qr_bottom['views'])])
                tab_data.append([f'QR_BUCKET_{i}_BOTTOM_ENGAGEMENT', str(qr_bottom['engagement'])])
                tab_data.append([f'QR_BUCKET_{i}_BOTTOM_DURATION', f"{qr_bottom['duration']}s"])
                tab_data.append(['', ''])

    # =============================
    # STEP 10: Write Excel File
    # =============================
    excel_filename = f"{args.hashtag}_client_data.xlsx"
    excel_path = os.path.join(reports_base_path, excel_filename)

    print(f"\n💾 Writing Excel file...")

    df = pd.DataFrame(tab_data, columns=['Field Name', 'Value'])
    df.to_excel(excel_path, sheet_name='Report_Data', index=False, engine='openpyxl')

    # =============================
    # STEP 11: Print Success Message
    # =============================
    print(f"\n✅ Extraction complete!")
    print(f"  📁 Excel: {excel_path}")
    print(f"  📁 QR codes: {qr_output_dir} ({len(qr_data_list)} files)")
    print(f"  📊 Total fields: {len(tab_data)}")
    print(f"  🎨 Creative formulas: {len(formula_names)}")


if __name__ == '__main__':
    main()
