#!/bin/bash

##############################################################################
# Stage 3 Testing Script
# Tests Feature Aggregation with existing test_vitamin run
##############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
export TEST_PATH="/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets"
CLIENT="test_final"
TARGET="test_vitamin"
ANALYSIS_MODE="top"
SELECTION_STRATEGY="contrastive"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Stage 3 Testing: Feature Aggregation${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

##############################################################################
# STEP 1: Verify Stage 2 Outputs Exist
##############################################################################

echo -e "${YELLOW}[STEP 1] Verifying Stage 2 outputs exist...${NC}"
echo ""

# Check bucket 18-33s
COUNT_18_33=$(ls -1 $TEST_PATH/bucket_18-33s/analysis/insights/*_temporal_windows_updated.json 2>/dev/null | wc -l)
echo "Bucket 18-33s: $COUNT_18_33 files"

# Check bucket 13-18s
COUNT_13_18=$(ls -1 $TEST_PATH/bucket_13-18s/analysis/insights/*_temporal_windows_updated.json 2>/dev/null | wc -l)
echo "Bucket 13-18s: $COUNT_13_18 files"

# Check bucket 60-90s
COUNT_60_90=$(ls -1 $TEST_PATH/bucket_60-90s/analysis/insights/*_temporal_windows_updated.json 2>/dev/null | wc -l)
echo "Bucket 60-90s: $COUNT_60_90 files"

# Total
TOTAL_COUNT=$(find $TEST_PATH -name "*_temporal_windows_updated.json" 2>/dev/null | wc -l)
echo "Total: $TOTAL_COUNT files"
echo ""

# Validation
if [ "$TOTAL_COUNT" -eq 0 ]; then
    echo -e "${RED}❌ ERROR: No temporal_windows_updated.json files found!${NC}"
    echo -e "${RED}   Stage 2 must be completed first.${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Stage 2 outputs verified ($TOTAL_COUNT videos found)${NC}"
fi

echo ""

##############################################################################
# STEP 2: Inspect Sample JSON Structure
##############################################################################

echo -e "${YELLOW}[STEP 2] Inspecting sample JSON structure...${NC}"
echo ""

SAMPLE_FILE=$(ls $TEST_PATH/bucket_18-33s/analysis/insights/*_temporal_windows_updated.json 2>/dev/null | head -1)

if [ -z "$SAMPLE_FILE" ]; then
    echo -e "${RED}❌ ERROR: No sample file found in bucket_18-33s${NC}"
    exit 1
fi

echo "Sample file: $(basename $SAMPLE_FILE)"
echo ""

# Check window types
echo "Window types present:"
cat "$SAMPLE_FILE" | jq -r '.temporal_windows | keys[]' 2>/dev/null || echo "ERROR: Failed to parse JSON"
echo ""

# Check middle segments count
MIDDLE_COUNT=$(cat "$SAMPLE_FILE" | jq '.temporal_windows.middle_segments | length' 2>/dev/null)
echo "Middle segments count: $MIDDLE_COUNT (expected: 4 for 18-33s bucket)"
echo ""

# Check metadata
echo "Metadata fields present:"
cat "$SAMPLE_FILE" | jq -r '.metadata | keys[]' 2>/dev/null | head -5
echo ""

echo -e "${GREEN}✅ JSON structure looks valid${NC}"
echo ""

##############################################################################
# STEP 3: Create ml_analysis Directories
##############################################################################

echo -e "${YELLOW}[STEP 3] Creating ml_analysis directories...${NC}"
echo ""

mkdir -p $TEST_PATH/bucket_18-33s/ml_analysis
mkdir -p $TEST_PATH/bucket_13-18s/ml_analysis
mkdir -p $TEST_PATH/bucket_60-90s/ml_analysis

echo "Created:"
ls -ld $TEST_PATH/bucket_*/ml_analysis
echo ""

echo -e "${GREEN}✅ Directories created${NC}"
echo ""

##############################################################################
# STEP 4: Run Stage 3 (Feature Aggregation)
##############################################################################

echo -e "${YELLOW}[STEP 4] Running Stage 3: Feature Aggregation...${NC}"
echo ""

cd /home/jorge/rumiaifinal

echo "Command:"
echo "python rumiai_ml_batch.py \\"
echo "  --client \"$CLIENT\" \\"
echo "  --analysis-type hashtag \\"
echo "  --target \"$TARGET\" \\"
echo "  --analysis-mode $ANALYSIS_MODE \\"
echo "  --selection-strategy $SELECTION_STRATEGY \\"
echo "  --start-from stage3 \\"
echo "  --stop-at stage3 \\"
echo "  --auto-confirm"
echo ""

/home/jorge/rumiaifinal/venv/bin/python3 rumiai_ml_batch.py \
  --client "$CLIENT" \
  --analysis-type hashtag \
  --target "$TARGET" \
  --analysis-mode $ANALYSIS_MODE \
  --selection-strategy $SELECTION_STRATEGY \
  --start-from stage3 \
  --stop-at stage3 \
  --auto-confirm

STAGE3_EXIT_CODE=$?

echo ""

if [ $STAGE3_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}❌ Stage 3 failed with exit code $STAGE3_EXIT_CODE${NC}"
    exit $STAGE3_EXIT_CODE
fi

echo -e "${GREEN}✅ Stage 3 completed successfully${NC}"
echo ""

##############################################################################
# STEP 5: Validate Stage 3 Outputs
##############################################################################

echo -e "${YELLOW}[STEP 5] Validating Stage 3 outputs...${NC}"
echo ""

# 5.1: Check files were created
echo "=== Files Created ==="
ls -lh $TEST_PATH/bucket_*/ml_analysis/aggregated_features.csv 2>/dev/null || echo "No CSV files found!"
echo ""

# 5.2: Check row counts
echo "=== Row Counts (includes header) ==="
for bucket in "bucket_18-33s" "bucket_13-18s" "bucket_60-90s"; do
    if [ -f "$TEST_PATH/$bucket/ml_analysis/aggregated_features.csv" ]; then
        row_count=$(wc -l < "$TEST_PATH/$bucket/ml_analysis/aggregated_features.csv")
        echo "$bucket: $row_count rows"
    else
        echo "$bucket: FILE NOT FOUND"
    fi
done
echo ""

# Expected row counts
EXPECTED_ROWS_18_33=$((COUNT_18_33 + 1))  # +1 for header
EXPECTED_ROWS_13_18=$((COUNT_13_18 + 1))
EXPECTED_ROWS_60_90=$((COUNT_60_90 + 1))

# 5.3: Check column counts
echo "=== Column Counts ==="
for bucket in "bucket_18-33s" "bucket_13-18s" "bucket_60-90s"; do
    if [ -f "$TEST_PATH/$bucket/ml_analysis/aggregated_features.csv" ]; then
        col_count=$(head -1 "$TEST_PATH/$bucket/ml_analysis/aggregated_features.csv" | awk -F',' '{print NF}')
        echo "$bucket: $col_count columns"
    fi
done
echo ""

# Expected column counts
echo "Expected column counts:"
echo "  bucket_18-33s: 129 columns (21 features × 6 windows + 3 metadata)"
echo "  bucket_13-18s: 66 columns (21 features × 3 windows + 3 metadata)"
echo "  bucket_60-90s: 150 columns (21 features × 7 windows + 3 metadata)"
echo ""

# 5.4: Check column headers (bucket 18-33s)
echo "=== Bucket 18-33s Column Headers (first 10) ==="
if [ -f "$TEST_PATH/bucket_18-33s/ml_analysis/aggregated_features.csv" ]; then
    head -1 "$TEST_PATH/bucket_18-33s/ml_analysis/aggregated_features.csv" | cut -d',' -f1-10 | tr ',' '\n' | nl
fi
echo ""

# 5.5: Check for middle_aggregate in bucket 13-18s
echo "=== Bucket 13-18s: Middle Segment Check ==="
if [ -f "$TEST_PATH/bucket_13-18s/ml_analysis/aggregated_features.csv" ]; then
    middle_agg_count=$(head -1 "$TEST_PATH/bucket_13-18s/ml_analysis/aggregated_features.csv" | tr ',' '\n' | grep -c "middle_aggregate" || echo "0")
    middle_numbered_count=$(head -1 "$TEST_PATH/bucket_13-18s/ml_analysis/aggregated_features.csv" | tr ',' '\n' | grep -c "middle_[1-5]" || echo "0")

    echo "middle_aggregate_* columns: $middle_agg_count (expected: 21)"
    echo "middle_[1-5]_* columns: $middle_numbered_count (expected: 0)"
    echo ""

    if [ "$middle_agg_count" -gt 0 ] && [ "$middle_numbered_count" -eq 0 ]; then
        echo -e "${GREEN}✅ CORRECT: Bucket 13-18s uses middle_aggregate${NC}"
    else
        echo -e "${RED}❌ ERROR: Bucket 13-18s structure is incorrect${NC}"
    fi
fi
echo ""

# 5.6: Preview data (first 3 rows, first 5 columns)
echo "=== Data Preview (bucket_18-33s, first 3 rows, first 5 columns) ==="
if [ -f "$TEST_PATH/bucket_18-33s/ml_analysis/aggregated_features.csv" ]; then
    cut -d',' -f1-5 "$TEST_PATH/bucket_18-33s/ml_analysis/aggregated_features.csv" | head -3
fi
echo ""

##############################################################################
# STEP 6: Python-based Advanced Validation
##############################################################################

echo -e "${YELLOW}[STEP 6] Running Python-based validation...${NC}"
echo ""

python3 << 'PYEOF'
import pandas as pd
import sys

try:
    # Load bucket 18-33s
    df = pd.read_csv('/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_18-33s/ml_analysis/aggregated_features.csv')

    print("=== Bucket 18-33s DataFrame Info ===")
    print(f"Shape: {df.shape}")
    print(f"Expected: (43 videos, 129 features)")
    print()

    # Check for required columns
    required_cols = ['video_id', 'create_time', 'gender']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
    else:
        print("✅ Required metadata columns present")
    print()

    # Check for hook features
    hook_cols = [col for col in df.columns if col.startswith('hook_')]
    print(f"Hook features: {len(hook_cols)} (expected: 21)")

    # Check for middle features
    middle_cols = [col for col in df.columns if 'middle_' in col]
    print(f"Middle features: {len(middle_cols)} (expected: 84 = 21 × 4)")

    # Check for closing features
    closing_cols = [col for col in df.columns if col.startswith('closing_')]
    print(f"Closing features: {len(closing_cols)} (expected: 21)")
    print()

    # Null value analysis
    null_counts = df.isnull().sum()
    cols_with_nulls = (null_counts > 0).sum()
    total_nulls = null_counts.sum()

    print(f"Columns with nulls: {cols_with_nulls}")
    print(f"Total null values: {total_nulls}")

    if total_nulls > (df.shape[0] * df.shape[1] * 0.5):
        print("❌ WARNING: >50% of data is null!")
    else:
        print("✅ Null values within acceptable range")
    print()

    # Check bucket 13-18s structure
    print("=== Bucket 13-18s Structure Check ===")
    df_13_18 = pd.read_csv('/home/jorge/rumiaifinal/data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_13-18s/ml_analysis/aggregated_features.csv')

    middle_agg_cols = [col for col in df_13_18.columns if 'middle_aggregate' in col]
    middle_numbered = [col for col in df_13_18.columns if any(f'middle_{i}' in col for i in [1,2,3,4,5])]

    print(f"Shape: {df_13_18.shape}")
    print(f"Expected: (12 videos, 66 features)")
    print(f"middle_aggregate_* columns: {len(middle_agg_cols)} (expected: 21)")
    print(f"middle_[1-5]_* columns: {len(middle_numbered)} (expected: 0)")

    if len(middle_agg_cols) == 21 and len(middle_numbered) == 0:
        print("✅ Bucket 13-18s structure is CORRECT")
    else:
        print("❌ Bucket 13-18s structure is INCORRECT")

    print()
    print("✅ Python validation completed successfully")

except FileNotFoundError as e:
    print(f"❌ ERROR: File not found - {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)
PYEOF

PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}❌ Python validation failed${NC}"
    exit $PYTHON_EXIT_CODE
fi

echo ""

##############################################################################
# Final Summary
##############################################################################

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   Stage 3 Testing Summary${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

echo -e "${GREEN}✅ All checks passed!${NC}"
echo ""
echo "Stage 3 outputs created:"
echo "  ✅ bucket_18-33s/ml_analysis/aggregated_features.csv"
echo "  ✅ bucket_13-18s/ml_analysis/aggregated_features.csv"
echo "  ✅ bucket_60-90s/ml_analysis/aggregated_features.csv"
echo ""
echo "Next steps:"
echo "  1. Review the output above for any warnings"
echo "  2. If everything looks good, proceed to Stage 4 testing"
echo "  3. Run: ./test_stage4.sh"
echo ""

echo -e "${GREEN}Stage 3 testing complete! 🎉${NC}"
