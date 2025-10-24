#!/bin/bash
# Pre-flight validation for Option B implementation

echo "🔍 Pre-flight Validation for Bug #1 Option B Fix"
echo "=================================================="

FAILED=0

# 1. Check Stage 3 outputs exist
echo -n "✓ Checking Stage 3 outputs exist... "
if ls data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_*/ml_analysis/aggregated_features.csv >/dev/null 2>&1; then
    COUNT=$(ls data/clients/test_final/hashtags/test_vitamin/top_contrastive/buckets/bucket_*/ml_analysis/aggregated_features.csv 2>/dev/null | wc -l)
    echo "✓ Found $COUNT CSV files"
else
    echo "❌ Stage 3 outputs missing!"
    FAILED=1
fi

# 2. Check Stage 4 script exists
echo -n "✓ Checking Stage 4 script exists... "
if [ -f scripts/stage4_transformation.py ]; then
    echo "✓ Found"
else
    echo "❌ scripts/stage4_transformation.py not found!"
    FAILED=1
fi

# 3. Check model_training.py exists
echo -n "✓ Checking Stage 5 module exists... "
if [ -f rumiai_v2/processors/model_training.py ]; then
    echo "✓ Found"
else
    echo "❌ rumiai_v2/processors/model_training.py not found!"
    FAILED=1
fi

# 4. Check virtual environment (optional)
echo -n "✓ Checking virtual environment... "
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Active: $VIRTUAL_ENV"
else
    echo "⚠️  No venv active (optional)"
fi

# 5. Check Python version
echo -n "✓ Checking Python version... "
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "✓ Python $PYTHON_VERSION"

# 6. Check pandas version
echo -n "✓ Checking pandas version... "
PANDAS_VERSION=$(python3 -c "import pandas; print(pandas.__version__)" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✓ Pandas $PANDAS_VERSION"
else
    echo "❌ Pandas not installed!"
    FAILED=1
fi

# 7. Check sklearn version
echo -n "✓ Checking sklearn version... "
SKLEARN_VERSION=$(python3 -c "import sklearn; print(sklearn.__version__)" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✓ Sklearn $SKLEARN_VERSION"
else
    echo "❌ Sklearn not installed!"
    FAILED=1
fi

# 8. Check disk space
echo -n "✓ Checking disk space... "
AVAILABLE=$(df -BG /home/jorge/rumiaifinal | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$AVAILABLE" -gt 5 ]; then
    echo "✓ ${AVAILABLE}GB available"
else
    echo "⚠️  Low disk space: ${AVAILABLE}GB"
fi

echo ""
echo "ℹ️  NOTE: This project uses venv with orchestrator (rumiai_ml_batch.py)"
echo "   Ensure you're in the correct venv before running Stage 4-5"

echo "=================================================="
if [ $FAILED -eq 0 ]; then
    echo "✅ All pre-flight checks passed! Ready to implement."
    exit 0
else
    echo "❌ Pre-flight checks FAILED. Fix issues above before proceeding."
    exit 1
fi
