#!/bin/bash
################################################################################
# DRY RUN: Copy Missing Videos from Test 4 to Test 3
#
# Purpose: Test copy operation with 2 sample videos (1 per bucket)
#
# This script will actually copy 2 videos to verify the operation works
# before running the full 60-video copy.
#
# Created: 2025-10-29
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base paths
TEST4_BASE="/home/jorge/rumiaifinal/data/clients/rollo_test4/hashtags/wellness_test4/top_contrastive"
TEST3_BASE="/home/jorge/rumiaifinal/data/clients/rollo_test3/hashtags/wellness_test3/top_contrastive"

# Counters
total_videos=0
successful_videos=0
failed_videos=0
total_files=0
successful_files=0
failed_files=0

# Log file
LOG_FILE="/home/jorge/rumiaifinal/copy_dryrun_$(date +%Y%m%d_%H%M%S).log"

echo -e "${YELLOW}=====================================================================${NC}"
echo -e "${YELLOW}DRY RUN MODE - Testing with 2 sample videos${NC}"
echo -e "${YELLOW}=====================================================================${NC}"
echo ""
echo "Log file: $LOG_FILE"
echo "" | tee "$LOG_FILE"

################################################################################
# Function: Copy video files
################################################################################
copy_video_files() {
    local video_id=$1
    local bucket=$2
    local video_success=true

    echo -e "${BLUE}Processing video: $video_id (bucket: $bucket)${NC}" | tee -a "$LOG_FILE"

    # Source paths (Test 4)
    local src_video="$TEST4_BASE/buckets/bucket_${bucket}/videos/${video_id}.mp4"
    local src_insights="$TEST4_BASE/buckets/bucket_${bucket}/analysis/insights/${video_id}_temporal_windows_updated.json"
    local src_unified="$TEST4_BASE/buckets/bucket_${bucket}/analysis/unified/${video_id}.json"

    # Target paths (Test 3)
    local tgt_video="$TEST3_BASE/buckets/bucket_${bucket}/videos/${video_id}.mp4"
    local tgt_insights="$TEST3_BASE/buckets/bucket_${bucket}/analysis/insights/${video_id}_temporal_windows_updated.json"
    local tgt_unified="$TEST3_BASE/buckets/bucket_${bucket}/analysis/unified/${video_id}.json"

    echo "  Source paths:" | tee -a "$LOG_FILE"
    echo "    Video:    $src_video" | tee -a "$LOG_FILE"
    echo "    Insights: $src_insights" | tee -a "$LOG_FILE"
    echo "    Unified:  $src_unified" | tee -a "$LOG_FILE"
    echo "  Target paths:" | tee -a "$LOG_FILE"
    echo "    Video:    $tgt_video" | tee -a "$LOG_FILE"
    echo "    Insights: $tgt_insights" | tee -a "$LOG_FILE"
    echo "    Unified:  $tgt_unified" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Copy video file
    if [ -f "$src_video" ]; then
        echo "  Checking source video..." | tee -a "$LOG_FILE"
        ls -lh "$src_video" | tee -a "$LOG_FILE"
        if cp "$src_video" "$tgt_video"; then
            echo -e "  ${GREEN}✓ Copied video file${NC}" | tee -a "$LOG_FILE"
            ls -lh "$tgt_video" | tee -a "$LOG_FILE"
            successful_files=$((successful_files + 1))
        else
            echo -e "  ${RED}✗ Failed to copy video file${NC}" | tee -a "$LOG_FILE"
            failed_files=$((failed_files + 1))
            video_success=false
        fi
    else
        echo -e "  ${RED}✗ Source video file not found: $src_video${NC}" | tee -a "$LOG_FILE"
        failed_files=$((failed_files + 1))
        video_success=false
    fi
    total_files=$((total_files + 1))

    # Copy insights file
    if [ -f "$src_insights" ]; then
        echo "  Checking source insights..." | tee -a "$LOG_FILE"
        ls -lh "$src_insights" | tee -a "$LOG_FILE"
        if cp "$src_insights" "$tgt_insights"; then
            echo -e "  ${GREEN}✓ Copied insights file${NC}" | tee -a "$LOG_FILE"
            ls -lh "$tgt_insights" | tee -a "$LOG_FILE"
            successful_files=$((successful_files + 1))
        else
            echo -e "  ${RED}✗ Failed to copy insights file${NC}" | tee -a "$LOG_FILE"
            failed_files=$((failed_files + 1))
            video_success=false
        fi
    else
        echo -e "  ${RED}✗ Source insights file not found: $src_insights${NC}" | tee -a "$LOG_FILE"
        failed_files=$((failed_files + 1))
        video_success=false
    fi
    total_files=$((total_files + 1))

    # Copy unified file
    if [ -f "$src_unified" ]; then
        echo "  Checking source unified..." | tee -a "$LOG_FILE"
        ls -lh "$src_unified" | tee -a "$LOG_FILE"
        if cp "$src_unified" "$tgt_unified"; then
            echo -e "  ${GREEN}✓ Copied unified file${NC}" | tee -a "$LOG_FILE"
            ls -lh "$tgt_unified" | tee -a "$LOG_FILE"
            successful_files=$((successful_files + 1))
        else
            echo -e "  ${RED}✗ Failed to copy unified file${NC}" | tee -a "$LOG_FILE"
            failed_files=$((failed_files + 1))
            video_success=false
        fi
    else
        echo -e "  ${RED}✗ Source unified file not found: $src_unified${NC}" | tee -a "$LOG_FILE"
        failed_files=$((failed_files + 1))
        video_success=false
    fi
    total_files=$((total_files + 1))

    # Update video counters
    total_videos=$((total_videos + 1))
    if [ "$video_success" = true ]; then
        echo -e "  ${GREEN}✓✓✓ Video $video_id COMPLETE (all 3 files copied)${NC}" | tee -a "$LOG_FILE"
        successful_videos=$((successful_videos + 1))
    else
        echo -e "  ${RED}✗✗✗ Video $video_id INCOMPLETE (some files failed)${NC}" | tee -a "$LOG_FILE"
        failed_videos=$((failed_videos + 1))
    fi
    echo "" | tee -a "$LOG_FILE"
}

################################################################################
# Main execution - DRY RUN with 2 sample videos
################################################################################

echo "=====================================================================" | tee -a "$LOG_FILE"
echo "SAMPLE 1: Bucket 3-9s (testing 1 video)" | tee -a "$LOG_FILE"
echo "=====================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Get first video from bucket 3-9s
sample_video_1=$(head -1 /tmp/test3_missing_3_9s.txt)
echo "Selected video: $sample_video_1" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
copy_video_files "$sample_video_1" "3-9s"

echo "=====================================================================" | tee -a "$LOG_FILE"
echo "SAMPLE 2: Bucket 60-90s (testing 1 video)" | tee -a "$LOG_FILE"
echo "=====================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Get first video from bucket 60-90s
sample_video_2=$(head -1 /tmp/test3_missing_60_90s.txt)
echo "Selected video: $sample_video_2" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
copy_video_files "$sample_video_2" "60-90s"

################################################################################
# Summary
################################################################################

echo "" | tee -a "$LOG_FILE"
echo -e "${YELLOW}=====================================================================${NC}" | tee -a "$LOG_FILE"
echo -e "${YELLOW}DRY RUN COMPLETE${NC}" | tee -a "$LOG_FILE"
echo -e "${YELLOW}=====================================================================${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Videos tested: 2 (1 per bucket)" | tee -a "$LOG_FILE"
echo -e "  ${GREEN}Successful: $successful_videos${NC}" | tee -a "$LOG_FILE"
echo -e "  ${RED}Failed: $failed_videos${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Files copied (3 per video):" | tee -a "$LOG_FILE"
echo -e "  ${GREEN}Successful: $successful_files/6${NC}" | tee -a "$LOG_FILE"
echo -e "  ${RED}Failed: $failed_files/6${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $failed_videos -eq 0 ]; then
    echo -e "${GREEN}=====================================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}✓✓✓ DRY RUN SUCCESSFUL!${NC}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}=====================================================================${NC}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "The copy operation works correctly. You can now run the full script:" | tee -a "$LOG_FILE"
    echo "  ./copy_test4_to_test3.sh" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "This will copy the remaining 58 videos (36 + 22)." | tee -a "$LOG_FILE"
    exit 0
else
    echo -e "${RED}=====================================================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${RED}⚠ DRY RUN FAILED${NC}" | tee -a "$LOG_FILE"
    echo -e "${RED}=====================================================================${NC}" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Some files failed to copy. Review the log above to identify issues." | tee -a "$LOG_FILE"
    echo "DO NOT run the full script until this is resolved." | tee -a "$LOG_FILE"
    exit 1
fi
