#!/bin/bash
set -e

echo "🚀 TESTING WORKFLOW AFTER FIX IMPLEMENTATION"
echo "=============================================="

# Variables
CLIENT="test_final"
TARGET="test_vitamin"
STRATEGY="contrastive"
VIDEO_COUNT=50

echo ""
echo "📋 Test Prerequisites:"
echo "  ✓ Stage 1 fix implemented in video_selector.py"
echo "  ✓ Stage 4 fix implemented in feature_transformation.py"
echo ""

read -p "Press Enter to start testing..."

# ============================================================================
# PHASE 1: Test Stage 1 (Video Selection with Tags)
# ============================================================================
echo ""
echo "================================================"
echo "PHASE 1: Testing Stage 1 (Video Selection)"
echo "================================================"

echo ""
echo "Step 1.1: Re-run Stage 1 to regenerate selected_videos.json"
python3 rumiai_ml_batch.py \
    --client "$CLIENT" \
    --target "$TARGET" \
    --analysis-mode top \
    --selection-strategy "$STRATEGY" \
    --video-count "$VIDEO_COUNT" \
    --stage 1

echo ""
echo "Step 1.2: Verify Stage 1 tags"
python3 test_stage1_tagging.py

# ============================================================================
# PHASE 2: Test Stage 4 (Feature Transformation reads tags)
# ============================================================================
echo ""
echo "================================================"
echo "PHASE 2: Testing Stage 4 (Feature Transformation)"
echo "================================================"

echo ""
echo "Step 2.1: Re-run Stage 4 transformation"
python3 rumiai_ml_batch.py \
    --client "$CLIENT" \
    --target "$TARGET" \
    --analysis-mode top \
    --selection-strategy "$STRATEGY" \
    --video-count "$VIDEO_COUNT" \
    --stage 4

echo ""
echo "Step 2.2: Verify Stage 4 read tags correctly"
python3 test_stage4_labels.py

# ============================================================================
# PHASE 3: Test Stage 5 (ML Training with both classes)
# ============================================================================
echo ""
echo "================================================"
echo "PHASE 3: Testing Stage 5 (ML Training)"
echo "================================================"

echo ""
echo "Step 3.1: Run Stage 5 training"
python3 rumiai_ml_batch.py \
    --client "$CLIENT" \
    --target "$TARGET" \
    --analysis-mode top \
    --selection-strategy "$STRATEGY" \
    --video-count "$VIDEO_COUNT" \
    --stage 5

echo ""
echo "✅ ALL TESTS COMPLETE!"
echo "================================================"
