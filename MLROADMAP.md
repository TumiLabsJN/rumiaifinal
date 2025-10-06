# ML Development Roadmap

## Overview
Transform raw video analysis data (>50 features per video) into **duration-specific** actionable creative insights delivered to brand affiliates, recognizing that successful patterns vary dramatically between 15-second and 120-second content. Each duration bucket receives its own ML model and creative recommendations.

## Planned Developments
**Note**: All features listed here are priority developments needed for MVP. Implementation order will
  be determined by client needs and resource availability.

### Project Success Definition
Success for this ML training pipeline is measured through technical and operational achievements.

### Success Criteria
1. **Processing Capability**
   - Successfully analyze up to 300 videos per hashtag in sequential fashion
   - Support multiple hashtags per client (e.g., 4-10+ hashtags each with 300 videos)
   - Checkpoint/resume system enables recovery from failures without data loss
   - Complete end-to-end processing or clear failure identification for debugging

2. **ML Insight Generation**
   - Generate meaningful trends and patterns from analyzed videos
   - Include confidence scores and pattern validation for professional credibility

3. **Creator Report Delivery**
   - Produce PDF reports with concise, actionable instructions
   - Avoid overwhelming numeric/technical ML outputs
   - Focus on "easy to replicate" format: clear steps without complex data
   - Identical reports for both UGC Factories and individual creators

4. **Client Executive Reporting**
   - Generate bird's eye view reports covering minimum 5 hashtags per client
   - Show scope of analysis: hashtags analyzed, creative insights distributed
   - Demonstrate value through breadth of research and strategic insights
   - Top-down view for executive stakeholders

#### Key Metrics
- **Input Scale**: User-configurable via --video-count N per qualified bucket
  - Contrastive default: N=100 per bucket (80 top + 20 bottom), top 3 buckets = ~300 videos
  - Top default: N=40 per bucket, top 3 buckets = ~120 videos
  - Only top 3 most active buckets are processed (adaptive bucket processing)
- **ML Models**: Random Forest and K-means with 16 models total (2 algorithms × 8 duration buckets)
- **Output**: Duration-specific creative recommendations (5 patterns per bucket)
- **Processing**: Sequential (one-by-one) with resumption capability

**Note**: These ML training buckets (for grouping videos by duration) are separate from temporal window segments (for analyzing within videos). Temporal windows output: 0-9s (no middle), 9-18s (3 middle segments), 18-33s (4 middle segments), 33-75s (5 middle segments), >75s (5 middle segments capped). The ML training uses 8 buckets: 0-3s, 3-9s, 9-13s, 13-18s, 18-33s, 33-60s, 60-90s, 90-120s.

##### 📊 Quality Built Into Selection Process
> **Key Point**: This system should automatically select high-quality videos through:
> - Top 80% + Bottom 20% per qualified duration bucket (N configurable via --video-count, default 100)
> - Adaptive bucket processing: Only top 3 most active buckets processed
> - User-defined date filters for recency control (`--date-filter last_N_days`)
> - Composite scoring (engagement × share boost factor)
> - No arbitrary thresholds needed - market performance determines quality

**Example CLI Command**:
```bash
python rumiai_ml_batch.py \
  --client "acme" \
  --analysis-type hashtag \
  --target "#nutrition" \
  --analysis-mode top \
  --selection-strategy contrastive \
  --video-count 100 \
  --date-filter last_90_days

# Result:
# - Scrapes 800 videos from #nutrition (engagement sorted)
# - Filters to videos from last 90 days
# - Buckets by duration → identifies top 3 active buckets
# - Per bucket: Selects top 80 + bottom 20 from qualified videos
# - Trains RF + K-Means models on selected videos
# - Generates creative strategy reports
```

##### 🔄 Fail-Fast with Checkpoint/Resume Architecture
> **Key Design Principle**: This system uses fail-fast with automatic checkpointing:
> - Processing stops immediately when any analysis fails (no partial results)
> - Progress automatically saved after each successful video
> - Resume from exact failure point after fixing issues
> - No data loss, no need to reprocess completed videos
> - Full implementation details in Section 6.5

##### Analysis Approach: Contrastive + Prescriptive
- **Contrastive Method**: Analyze top 80% vs bottom 20% per bucket (e.g., 80 vs 20 if N=100, 120 vs 30 if N=150)
  * Configurable via --video-count N (default 100 for contrastive)
  * Adaptive processing: Only top 3 qualified buckets with ≥ N videos
  * Identifies what differentiates viral from poor-performing content
  * Finds patterns with largest performance gaps (e.g., 85% in top vs 20% in bottom)
- **Prescriptive Output**: Convert patterns to actionable recommendations
  * "Add text within 3 seconds (4x higher viral rate)"
  * Prioritized by impact magnitude

---

### 🤖 ML Model Strategy & Architecture Decision

#### Hybrid ML Ensemble (RF + K-Means)
**Structure**: 1 RF + 1 K-Means per bucket = 8 models total (but same data)

**ML Approach**: Contrastive-first multi-analytical approach
- **Contrastive Analysis** (foundation): Random Forest classifies top 40 vs bottom 20
- **Descriptive Segmentation**: K-Means identifies content style groups
- **Predictive Scoring**: RF provides viral probability scores
- **Prescriptive Recommendations**: Convert insights to actionable steps
- **Natural Language Reports**: Claude API transforms statistical findings into narrative recommendations


### A. System Architecture

#### A.1 Goals - Core Functionalities

##### Primary Goals
1. **Batch Video Analysis**
   - Process up to 300 videos sequentially through `rumiai_runner.py`
   - Implement checkpoint/resume system for failure recovery

2. **Client-Centric Data Organization**
   - Multi-tenant data structure: Client → Hashtags → Duration Buckets → Videos
   - Bucket-specific analysis within client/hashtag boundaries
   - Persistent client/hashtag/duration configuration management

3. **Duration-Specific ML Pattern Recognition**
   - Train **separate ML models for each duration bucket**
   - Recognize that 15-second patterns differ completely from 60-second patterns
   - Generate bucket-specific insights (no universal patterns across durations)

4. **Creative Report Generation**
   - Output 5 creative strategy reports per bucket, or 20 total per Hashtag 
   - Multiple perspectives and strategies for content creators
   - Format: "What works for 15-second #nutrition videos" (not generic advice)
   - Include bucket performance metrics for strategic content planning

##### Success Criteria
- 100% completion rate with checkpoint recovery
- < 2 hours for 200 video batch processing
- Actionable insights with confidence scores > 0.8
- Creative reports readable by non-technical users

---

### B. Video Selection Criteria & Apify Integration
Defines a TikTok video selection pipeline: scrape 400 to 800 per hashtag via Apify, apply client side date filter, bucket by duration, rank by engagement, and process sequentially with checkpoints. Also covers costs, data fields, scaling and validation, storage, and confidence reporting.

#### Duration Bucket Strategy for ML Training
**Key Insight**: While production temporal_compute.py outputs identical structure for certain duration ranges (e.g., all 9-18s videos have 3 middle segments), the ML training pipeline buckets them separately:

**Production Output** (temporal_compute.py):
- 9-18s videos → All output 3 middle segments (same JSON structure)
- 18-33s videos → All output 4 middle segments (same JSON structure)

**ML Training Buckets** (separate models):
- Bucket 3: 9-13s videos (3 segments of 1-2.33s each)
- Bucket 4: 13-18s videos (3 segments of 2.33-4s each)

**Why This Works**: Since the output structure is identical (both have 3 middle segments), the ML pipeline can simply filter videos by duration metadata to train separate models. The models learn that identical feature values (e.g., word_count=20) mean different things based on the duration context. This avoids breaking changes to production while properly handling the 4x variance issue.

Brainstorm in:
\\wsl$\Ubuntu\home\jorge\rumiaifinal\documentation_migration\FutureDevelopments\VideoSelection.md


### C. Creative Report Output
How RumiAI turns video analysis into structured creative reports, tests them with professional creators, refines for affiliates, and delivers polished multi-audience PDFs with data-backed confidence levels. Brainstorm in: 
\\wsl$\Ubuntu\home\jorge\rumiaifinal\documentation_migration\FutureDevelopments\CreativeReportsOutput.md 

### D. Competitor Handle Analysis System
Refactor of initial Apify video download and selection to be based of TikTok Handles, not Hashtags.

Brainstorm in:
\\wsl$\Ubuntu\home\jorge\rumiaifinal\documentation_migration\FutureDevelopments\CompetitorHandle.md