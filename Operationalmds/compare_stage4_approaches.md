# Stage 4 Testing Approaches Comparison

## Approach 1: Direct Function Call (from RumiGeneralTests.md)

```python
from rumiai_v2.processors.feature_transformation import run_stage4_transformation

for bucket in buckets:
    bucket_path = f'{base_path}/{bucket}'
    run_stage4_transformation(bucket_path, selection_strategy='contrastive')
```

**Pros:**
- Direct control over each bucket
- Granular error handling per bucket
- Documented in RumiGeneralTests.md

**Cons:**
- Need to know function signature
- Manual iteration over buckets
- Bypasses orchestrator logic

---

## Approach 2: Orchestrator CLI (my suggestion)

```bash
python3 rumiai_ml_batch.py \
    --client test_final \
    --target test_vitamin \
    --analysis-mode top \
    --selection-strategy contrastive \
    --video-count 50 \
    --stage 4
```

**Pros:**
- Uses standard pipeline entry point
- Handles all buckets automatically
- Includes orchestration logic (checkpoints, etc.)

**Cons:**
- Less granular control
- Might fail if orchestrator has issues
- Requires full CLI parameters

---

## Recommendation

**Use Approach 1 (from RumiGeneralTests.md)** because:
1. ✅ Documented in official test guide
2. ✅ More granular (can test bucket-by-bucket)
3. ✅ Simpler for workaround testing
4. ✅ Doesn't require full orchestrator setup

