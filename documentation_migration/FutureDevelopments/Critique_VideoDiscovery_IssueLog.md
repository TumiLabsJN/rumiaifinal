# VideoDiscoveryCHILD.md - Critical Issues Resolution Log

> **Purpose**: Track resolution of critical issues identified in post-audit critique
> **Date**: 2025-01-28
> **Status**: ✅ COMPLETE - Decisions Woven into VideoDiscoveryCHILD.md v1.1

---

## CRITICAL Issue #1: Unverified Apify Engagement Formula ✅ RESOLVED

**Problem**: Document claimed engagement formula `views × (1 + share_rate × 10)` without verification. Discovered Apify doesn't provide server-side sorting.

**3 Alternatives Presented**:
1. Use simple view count only (sort by `playCount`)
2. Use documented TikTok engagement formula
3. Make formula configurable with documented default

**User Decision**: **Option 1 - View count only**

**Changes Applied**:

1. **Section 2.3.1** (lines 108, 143-149):
   - Updated purpose: "Scrape 800 videos from target and sort by engagement client-side"
   - Added client-side sorting logic:
     ```python
     if analysis_mode == "top":
         videos = sorted(videos, key=lambda v: v.get('playCount', 0), reverse=True)
     else:  # recent mode
         videos = sorted(videos, key=lambda v: v.get('createTime', 0), reverse=True)
     ```

2. **Section 2.3.1** (lines 156-168):
   - Replaced "Engagement Score Formula" with "Client-Side Engagement Sorting"
   - Removed complex formula, replaced with simple view count sorting
   - Added rationale: "View count is primary engagement metric, simple and transparent"

3. **Removed from Apify input**:
   - Deleted `"sortBy": "engagement"` parameter (Apify doesn't support)
   - Deleted `"sortOrder": "desc"` parameter

**Impact**:
- ✅ No unverified assumptions about Apify behavior
- ✅ Simple, transparent sorting logic (view count = engagement)
- ✅ TI developer has clear implementation: `sorted(videos, key=lambda v: v['playCount'], reverse=True)`

**Status**: ✅ COMPLETE

---

## CRITICAL Issue #2: No Error Recovery for Partial Failures ✅ RESOLVED

**Problem**: Winner analysis assumes ≥100 videos available. If Apify returns 50 videos → date filter reduces to 30 → `top_100 = videos[:100]` fails or returns wrong data.

**3 Alternatives Presented**:
1. Add minimum dataset size validation (fail-fast at 30 videos)
2. Graceful degradation (analyze what we have, any size)
3. Hybrid approach (minimum 10 + degradation 10-99 + normal ≥100)

**User Decision**: **Option 3 - Hybrid approach**

**Changes Applied**:

1. **Section 2.3.3** (lines 276-296):
   - Added minimum validation: `if len(videos) < 10: raise ValueError`
   - Added degraded mode: `if len(videos) < 100: analyze all + warn user`
   - Added normal mode: `if len(videos) ≥ 100: analyze top 100`
   - Clear error message: "Insufficient videos for analysis. Need ≥10, got {count}"
   - Warning message: "Small dataset ({count} videos). Statistical validity may be limited."

2. **Section 4.2** (line 672):
   - Added constant: `MIN_VIDEOS_FOR_ANALYSIS = 10`
   - Documented: "Absolute minimum videos needed (hard stop if < 10)"

3. **Section 6.2** (lines 868-869):
   - Added error case: "< 10 videos after filtering" → Fail-fast (exit code 6)
   - Added warning case: "10-99 videos (degraded mode)" → Warn + continue

**Impact**:
- ✅ Prevents crashes with small datasets
- ✅ Transparent about statistical limitations (warns user if <100)
- ✅ Flexible: works with niche hashtags (≥10 videos) but sets quality floor
- ✅ Clear user guidance: "Try different target or relax date filter"

**Status**: ✅ COMPLETE

---

---

## Summary: Decisions Woven into VideoDiscoveryCHILD.md v1.1

**All resolved issues have been integrated into the main HLD document:**

### Changes Applied to VideoDiscoveryCHILD.md:

1. **Section 2.3.1 - Apify Scraping** (lines 108, 143-149, 156-168):
   - Added client-side sorting by `playCount` (view count)
   - Removed unverified engagement formula
   - Clear rationale: "View count is primary engagement metric"

2. **Section 2.3.3 - Winner Analysis** (lines 276-296):
   - Added minimum validation (< 10 videos = fail)
   - Added degraded mode (10-99 videos = warn + analyze all)
   - Added normal mode (≥100 videos = analyze top 100)

3. **Section 4.2 - Internal Configuration** (line 672):
   - Added `MIN_VIDEOS_FOR_ANALYSIS = 10` constant

4. **Section 6.2 - Error Cases** (lines 868-869):
   - Added error case: "< 10 videos" → fail with exit code 6
   - Added warning case: "10-99 videos (degraded mode)" → continue

5. **Appendix B - Decision Log** (lines 1399-1509):
   - Decision 1: Client-Side Engagement Sorting (View Count Only)
   - Decision 2: Hybrid Minimum Dataset Size Validation
   - Decision 3: Appendix B retroactive addition
   - Documents alternatives, rationale, trade-offs for each decision

6. **Change Log** (line 1529):
   - Updated to v1.1 with summary of post-critique changes

**Result**: VideoDiscoveryCHILD.md v1.1 now includes all critical design decisions with full traceability.

---

## Issues Remaining (Deferred for Future Iterations)

The following issues were identified in post-audit critique but deferred for future enhancements:

- [ ] CRITICAL Issue #3: Race Condition in Success-Based Distribution (engagement data staleness)
- [ ] HIGH Issue #4: Contrastive Strategy Has a Flaw (bottom 20% not true failures)
- [ ] HIGH Issue #5: No Handling for Duplicate Videos (deduplication strategy needed)
- [ ] MEDIUM Issue #6: Date Filtering is Too Simplistic (no timezone handling, rigid format)
- [ ] MEDIUM Issue #7: Performance Targets Are Aspirational (no empirical basis)
- [ ] MEDIUM Issue #8: Bucket Definitions Misaligned with Business (8 defined, only 3 used)
- [ ] LOW Issue #9: Testing Strategy Incomplete (missing negative tests, load tests)
- [ ] LOW Issue #10: Winner Analysis Logic Has Edge Case Bug (MIN_WINNER_PERCENTAGE not enforced)

**Recommendation**: Address remaining issues in Phase 2 after Stage 1 TI implementation and validation.
