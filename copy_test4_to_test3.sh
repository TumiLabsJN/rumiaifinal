#!/bin/bash
################################################################################
# Copy Missing Videos from Test 4 to Test 3
#
# Purpose: Copy 60 missing videos (37 from bucket_3-9s, 23 from bucket_60-90s)
#          from Test 4 to Test 3 to recover from checkpoint corruption
#
# Source: rollo_test4/hashtags/wellness_test4/
# Target: rollo_test3/hashtags/wellness_test3/
#
# Files copied per video:
#   1. videos/{video_id}.mp4
#   2. analysis/insights/{video_id}_temporal_windows_updated.json
#   3. analysis/unified/{video_id}.json
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
LOG_FILE="/home/jorge/rumiaifinal/copy_test4_to_test3_$(date +%Y%m%d_%H%M%S).log"
echo "Copy operation started at $(date)" | tee "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

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

    # Copy video file
    if [ -f "$src_video" ]; then
        if cp "$src_video" "$tgt_video"; then
            echo "  ✓ Copied video file" | tee -a "$LOG_FILE"
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
        if cp "$src_insights" "$tgt_insights"; then
            echo "  ✓ Copied insights file" | tee -a "$LOG_FILE"
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
        if cp "$src_unified" "$tgt_unified"; then
            echo "  ✓ Copied unified file" | tee -a "$LOG_FILE"
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
        echo -e "  ${GREEN}✓ Video $video_id complete${NC}" | tee -a "$LOG_FILE"
        successful_videos=$((successful_videos + 1))
    else
        echo -e "  ${RED}✗ Video $video_id incomplete${NC}" | tee -a "$LOG_FILE"
        failed_videos=$((failed_videos + 1))
    fi
    echo "" | tee -a "$LOG_FILE"
}

################################################################################
# Main execution
################################################################################

echo "=====================================================================" | tee -a "$LOG_FILE"
echo "COPYING BUCKET 3-9s (37 videos)" | tee -a "$LOG_FILE"
echo "=====================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Read video IDs from temp file
if [ ! -f /tmp/test3_missing_3_9s.txt ]; then
    echo -e "${RED}ERROR: Missing video list not found: /tmp/test3_missing_3_9s.txt${NC}" | tee -a "$LOG_FILE"
    echo "Please run the verification commands first to generate this file." | tee -a "$LOG_FILE"
    exit 1
fi

while IFS= read -r video_id; do
    copy_video_files "$video_id" "3-9s"
done < /tmp/test3_missing_3_9s.txt

echo "=====================================================================" | tee -a "$LOG_FILE"
echo "COPYING BUCKET 60-90s (23 videos)" | tee -a "$LOG_FILE"
echo "=====================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Read video IDs from temp file
if [ ! -f /tmp/test3_missing_60_90s.txt ]; then
    echo -e "${RED}ERROR: Missing video list not found: /tmp/test3_missing_60_90s.txt${NC}" | tee -a "$LOG_FILE"
    echo "Please run the verification commands first to generate this file." | tee -a "$LOG_FILE"
    exit 1
fi

while IFS= read -r video_id; do
    copy_video_files "$video_id" "60-90s"
done < /tmp/test3_missing_60_90s.txt

################################################################################
# Summary
################################################################################

echo "" | tee -a "$LOG_FILE"
echo "=====================================================================" | tee -a "$LOG_FILE"
echo "COPY OPERATION COMPLETE" | tee -a "$LOG_FILE"
echo "=====================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Videos:" | tee -a "$LOG_FILE"
echo "  Total: $total_videos" | tee -a "$LOG_FILE"
echo -e "  ${GREEN}Successful: $successful_videos${NC}" | tee -a "$LOG_FILE"
echo -e "  ${RED}Failed: $failed_videos${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Files (3 per video):" | tee -a "$LOG_FILE"
echo "  Total: $total_files" | tee -a "$LOG_FILE"
echo -e "  ${GREEN}Successful: $successful_files${NC}" | tee -a "$LOG_FILE"
echo -e "  ${RED}Failed: $failed_files${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Completed at $(date)" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ $failed_videos -eq 0 ]; then
    echo -e "${GREEN}✓ All videos copied successfully!${NC}" | tee -a "$LOG_FILE"
    exit 0
else
    echo -e "${YELLOW}⚠ Some videos failed to copy. Check log for details.${NC}" | tee -a "$LOG_FILE"
    exit 1
fi
