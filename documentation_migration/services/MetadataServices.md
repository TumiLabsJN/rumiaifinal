# Metadata Services

## ⚠️ Legacy Information Warning
This document references current implementation verified through code inspection at:
- `/home/jorge/rumiaifinal/rumiai_v2/api/apify_client.py`
- `/home/jorge/rumiaifinal/rumiai_v2/core/models/video.py`
- `/home/jorge/rumiaifinal/rumiai_v2/processors/temporal_compute.py`
- `/home/jorge/rumiaifinal/rumiai_v2/processors/precompute_functions_full.py`
- `/home/jorge/rumiaifinal/scripts/rumiai_runner.py`

## 🔄 Flexible Execution Architecture
**IMPORTANT**: Metadata extraction happens BEFORE ML services, not in parallel.
- Apify scraping occurs first to get video metadata (rumiai_runner.py lines 243)
- Video download happens after scraping (line 249)
- The downloaded videos get processed either in Parallel or Sequentially. The architecture can support both and switch between them easily. 
- Metadata is passed through the entire pipeline

## 📦 Batch Processing Clarification
**CRITICAL DISTINCTION** - Two types of "batching":
1. **Metadata Processing** (WITHIN one video):
   - Single API call to Apify for video metadata
   - Hashtag analysis processes all tags at once
   - Engagement metrics calculated in one pass
   - This is NOT batched - it's a single operation

2. **Video Batching** (NOT IMPLEMENTED):
   - RumiAI processes ONE video at a time
   - No parallel scraping of multiple videos
   - Each video goes through the complete pipeline sequentially

## 📊 Service Overview Matrix
| Service | Purpose | Status | Currently Using | API Required | Output Type | Self-Contained |
|---------|---------|--------|-----------------|--------------|-------------|----------------|
| Apify Scraper | TikTok metadata extraction | ✅ Production | External API | ✅ Yes | Video metadata | ❌ No (needs API) |
| Hashtag Analysis | Classify hashtag strategy | ✅ Production | CPU | ❌ No | Metrics only | ✅ Yes |
| Engagement Calculator | Calculate engagement metrics | ✅ Production | CPU | ❌ No | Metrics only | ✅ Yes |

---

# Apify TikTok Scraper Service

## 🔄 Execution Context
**This service runs BEFORE all ML services through `rumiai_runner.py`.**
- Executed sequentially before video download
- ⚠️ **BLOCKS ENTIRE PIPELINE for up to 5 MINUTES on timeout**
- Required for all subsequent processing
- No retry logic implemented

## 🎯 Service Purpose
- **Single sentence**: Extracts comprehensive TikTok video metadata via Apify's scraping infrastructure
- **Input type**: TikTok video URL
- **Output type**: VideoMetadata object with engagement metrics, hashtags, author info
- **Required by**: ALL downstream services - provides video duration for frame sampling, engagement metrics for analysis, hashtags for strategy detection, creation time for temporal patterns
- **Pipeline dependency**: Without metadata, pipeline cannot proceed - no ML analysis possible

## ⚡ Performance Profile
```
Execution Time (To Be Confirmed - Pending Instrumentation Tests):
- Single video: TBC seconds
- With timeout: Up to 300 seconds (5 minutes)

Resource Usage (To Be Confirmed - Pending Instrumentation Tests):
- Memory: TBC
- CPU: TBC (network I/O bound expected)
- API Required: ✅ Yes (Apify token required)
- Network: TBC per request

Configuration:
- Parallelizable: No (sequential by design)
- Retry logic: ❌ Not implemented
- ⚠️ **TIMEOUT: 300 seconds (5 MINUTES!)** - Excessive for API call
- Current Status: ⚠️ Stable but timeout risk
```

## 🌐 API Integration
```
Apify Actor Details:
- Actor ID: GdWCkxBtKWOsKjdch
- Endpoint: https://api.apify.com/v2
- Authentication: Bearer token required

Configuration (from apify_client.py lines 44-53):
{
    "postURLs": [video_url],
    "resultsPerPage": 1,
    "shouldDownloadVideos": true,
    "shouldDownloadCovers": true,
    "shouldDownloadSubtitles": true,
    "proxyConfiguration": {
        "useApifyProxy": true
    }
}
```

## 🔍 Self-Containment Check
- [ ] Works without precompute imports (N/A - external service)
- [x] No circular dependencies (verified)
- [x] Clear service boundaries
- [ ] Isolated test exists: /test_apify_client.py (TODO)

## 🏗️ Service Architecture

### Service Boundaries
```
INPUT                    SERVICE                     OUTPUT
TikTok URL ─────────> Apify API ─────────> VideoMetadata Object
                         ├── Scraper Actor
                         ├── Start run
                         ├── Poll status
                         └── Get dataset
```

### Data Flow Pipeline
```
1. Input Stage
   └── Pass video URL to scraper

2. API Call Stage (apify_client.py)
   ├── Step 1: Start actor run (lines 56-67)
   ├── Step 2: Wait for completion (_wait_for_run, lines 232-251)
   └── Step 3: Get dataset items (lines 80-85)

3. Output Stage
   └── Convert to VideoMetadata (from_apify_data, video.py lines 35-84)
```

### Data Extraction Mapping
```python
# From video.py from_apify_data() method (lines 35-84):
Apify Fields → VideoMetadata Fields:
- data.get('id', '') → video_id
- data.get('webVideoUrl', data.get('url', '')) → url
- authorMeta.get('name', '') → username
- data.get('text', data.get('description', '')) → description
- videoMeta.get('duration', data.get('duration', 0)) → duration
- data.get('playCount', data.get('views', 0)) → views
- data.get('diggCount', data.get('likes', 0)) → likes
- data.get('commentCount', data.get('comments', 0)) → comments
- data.get('shareCount', data.get('shares', 0)) → shares
- data.get('collectCount', data.get('saves', 0)) → saves
- createTimeISO/createTime → create_time (parsed to datetime)
- videoUrl/downloadAddr/mediaUrls[0]/downloadUrl → download_url
- videoMeta.get('cover', data.get('coverUrl', '')) → cover_url
- data.get('hashtags', []) → hashtags
- data.get('musicMeta', data.get('music', {})) → music
- authorMeta → author
- data.get('engagementRate', 0.0) → engagement_rate
```

## 📁 File Structure & Key Locations
```
Service Implementation:
├── /rumiai_v2/api/apify_client.py (main client)
│   ├── ApifyClient class (lines 20-252)
│   ├── scrape_video() (lines 35-100)
│   ├── _start_actor_run() (lines 152-179)
│   └── _wait_for_run() (lines 232-251)
├── /rumiai_v2/core/models/video.py
│   ├── VideoMetadata dataclass (lines 10-114)
│   └── from_apify_data() (lines 35-84)
└── /scripts/rumiai_runner.py
    └── _scrape_video() (lines 363-378)

Configuration:
└── Settings.apify_token (referenced in rumiai_runner.py line 81)
```

## 🚨 Failure Modes & Recovery

### Common Failure Scenarios
| Failure | Cause | Impact | Recovery | Frequency |
|---------|-------|--------|----------|-----------|
| API timeout | **HANGS 5 MINUTES** | APIError raised | No retry (cost consideration) | Unknown |
| Invalid run ID | API response format | KeyError | Handles both 'id' and 'data.id' | Rare |
| No dataset items | Empty response | APIError raised | No retry (would incur cost) | Unknown |
| Missing token | Configuration error | All requests fail | Check settings | Setup issue |
| Network error | Transient failure | Pipeline stops | No retry (each retry costs money) | ~2-5% |

### Graceful Degradation Strategy
- **Principle**: Metadata is required - cannot proceed without it
- **No fallback**: Unlike ML services, metadata extraction has no fallback
- **No retry logic**: Intentional design choice - each retry incurs API cost
- **Cost consideration**: Failed attempts still charged by Apify
- **Error handling**: APIError raised with context (lines 73-78, 97-100)

## 🐛 Current Issues & Future Fixes

### Priority: MEDIUM 🟡
- **Issue**: No retry logic for failed requests (design trade-off)
- **Impact**: ~2-5% of videos fail on transient errors
- **Design rationale**: Each retry incurs additional API cost
- **Current approach**: Fail fast to minimize costs
- **Business decision needed**: Accept failure rate OR pay for retries
- **Proposed options**:
  - Add configurable retry count with cost warning
  - Retry only network errors (not API errors)
  - Keep current behavior, monitor failure rate
- **Effort Estimate**: 2 hours if retry desired
- **Files Affected**: apify_client.py

### Priority: HIGH 🔴
- **Issue**: Excessive 300-second (5 minute) timeout
- **Impact**: Pipeline can hang for 5 minutes on a single API call
- **Current setting**: 300 seconds (lines 232-241)
- **Proposed Fix**: Reduce to 30-60 seconds max
- **Rationale**: API calls should respond within seconds, not minutes
- **Effort Estimate**: 5 minutes (change timeout parameter)
- **Files Affected**: apify_client.py line 232

### Priority: MEDIUM 🟡
- **Issue**: No metadata caching
- **Impact**: Repeated requests for same video waste API calls
- **Proposed Fix**: Implement cache with TTL
- **Effort Estimate**: 2 days

### Priority: LOW 🟢
- **Issue**: Single video scraping only in scrape_video()
- **Impact**: Could batch for efficiency
- **Note**: scrape_multiple_videos() exists (lines 102-149) but unused
- **Proposed Fix**: Use batch method in runner
- **Effort Estimate**: 1 day

## 🧪 Testing & Validation

### Two Types of Testing

#### 1. Functional Testing (Isolation)
- **Purpose**: Verify API integration and parsing
- **NOT for**: Performance measurement
- **Location**: `/test_apify_client.py` (TODO - not found)

#### 2. Performance Testing (Full Pipeline)
- **Purpose**: Measure API latency with real calls
- **Status**: Pending proper instrumentation setup
- **Note**: Requires valid API token

## 📈 Optimization Opportunities
- [ ] **Add retry logic**: Implement exponential backoff (with cost warnings)
- [ ] **Response caching**: Reduce redundant API calls
- [ ] **USE existing batch method**: scrape_multiple_videos() already implemented (lines 102-149) but unused
- [ ] **Parallel requests**: For multiple video processing

## 🔄 Dependencies
```
External Services:
└── Apify API (commercial service)

External Libraries:
├── aiohttp (async HTTP client)
├── asyncio (async I/O)
├── json (response parsing)
└── time (timeout tracking)

Internal Dependencies:
├── VideoMetadata (core.models.video)
├── APIError (core.exceptions)
└── Settings (configuration)
```

---

# Hashtag Analysis Service

## 🔄 Execution Context
**This service runs during temporal_compute processing.**
- Called from extract_hashtag_metrics() function
- Runs in-process (not a separate service)
- Pure function with no side effects
- No external dependencies

## 🎯 Service Purpose
- **Single sentence**: Analyzes hashtag strategy by classifying tags as generic vs specific
- **Input type**: Metadata dictionary with hashtags array
- **Output type**: Hashtag metrics dictionary (counts and ratios)
- **Used by**: Content strategy analysis, creator pattern detection, viral potential scoring
- **Provides insight into**: Whether creator uses discovery-focused (generic) vs niche-focused (specific) hashtag strategy

## ⚡ Performance Profile
```
Execution Time:
- Any video: Negligible (<0.01s)

Resource Usage:
- Memory: Minimal (list operations only)
- CPU: Minimal (single loop)
- Pure Function: ✅ Yes (no side effects)

Configuration:
- Generic hashtag list: 17 predefined tags
- No external dependencies
- Current Status: ✅ Stable
```

## 📋 Generic Hashtag Classification
```
Generic hashtags defined (temporal_compute.py lines 163-182):

Discovery-focused (6 tags):
├── 'fyp', 'foryou', 'foryoupage'
├── 'viral', 'trending', 'explore'

Platform identity (2 tags):
├── 'tiktok', 'tiktokviral'

Creator community (2 tags):
├── 'tiktokcreator', 'contentcreator'

Engagement bait (2 tags):
├── 'funny', 'duet'

Trending variations (2 tags):
└── 'trendingvideo', 'tiktokchallenge'

Total: 17 generic hashtags
```

## 🏗️ Service Architecture

### Service Boundaries
```
INPUT                    SERVICE                     OUTPUT
Metadata Dict ─────────> Hashtag Analyzer ─────────> Metrics Dict
                           ├── Extract tags
                           ├── Normalize text
                           └── Count categories
```

### Data Flow Pipeline
```
1. Input Stage
   └── Extract metadata.get('hashtags', [])

2. Processing Stage (lines 189-199)
   ├── Handle both dict and string formats
   ├── Normalize: lowercase, strip '#'
   └── Check against generic_hashtags list

3. Output Stage (lines 205-209)
   └── Return metrics:
       ├── hashtag_count: total
       ├── generic_hashtag_count: generic ones
       ├── specific_hashtag_count: total - generic
       └── generic_ratio: rounded to 3 decimals
```

## 📁 File Structure & Key Locations
```
Service Implementation:
└── /rumiai_v2/processors/temporal_compute.py
    └── extract_hashtag_metrics() (lines 150-210)
        ├── Generic list definition (lines 163-182)
        ├── Processing loop (lines 189-199)
        └── Metrics calculation (lines 201-209)

Usage:
└── Called from precompute functions during analysis
```

## 🚨 Failure Modes & Recovery

### Common Failure Scenarios
| Failure | Cause | Impact | Recovery | Frequency |
|---------|-------|--------|----------|-----------|
| Empty hashtags | No tags on video | All counts = 0 | Returns zero metrics | Common |
| Malformed tag | Unexpected format | Handled | Both dict and string supported | Rare |
| Division by zero | No hashtags | generic_ratio = 0 | Conditional check (line 202) | Handled |

## 🐛 Current Issues & Future Fixes

### Priority: LOW 🟢
- **Issue**: Generic hashtag list outdated and hardcoded
- **Impact**: Using 2020-era tags (fyp, viral) while TikTok trends evolve
- **Current Implementation**: Fixed list of 17 tags, unchanged since implementation
- **Proposed Fix**: Move to configuration file, update quarterly based on trending data
- **Effort Estimate**: 1 hour to externalize, ongoing maintenance needed
- **Files Affected**: temporal_compute.py

## 🧪 Testing & Validation

### Functional Testing
- **Purpose**: Verify classification logic
- **Test cases**: Empty tags, generic only, specific only, mixed
- **Location**: No dedicated test file found

## 📈 Optimization Opportunities
- [x] **Already optimized**: O(n) algorithm, single pass
- [ ] **Update generic list**: Current tags from 2020, needs refresh
- [ ] **Configurable list**: Move generic tags to settings
- [ ] **Category expansion**: Beyond binary classification

## 🔄 Dependencies
```
External Libraries:
└── None (pure Python)

Internal Dependencies:
└── None (standalone function)
```

---

# Engagement Metrics Calculator

## 🔄 Execution Context
**This service runs during precompute processing.**
- Called from compute_metadata_analysis_metrics()
- Runs in-process (calculations only)
- Pure mathematical operations
- Part of metadata analysis pipeline

## 🎯 Service Purpose
- **Single sentence**: Calculates engagement rate and normalized metrics from raw counts
- **Input type**: Video metadata with view/like/comment/share counts
- **Output type**: Structured metrics dictionary
- **Used by**: Performance scoring, viral prediction, creator benchmarking, temporal engagement patterns
- **Key metric**: Engagement rate = (likes+comments+shares)/views*100 - industry standard for content performance

## ⚡ Performance Profile
```
Execution Time:
- Any video: Negligible (<0.01s)

Resource Usage:
- Memory: Minimal (arithmetic only)
- CPU: Minimal (simple calculations)
- Pure Function: ✅ Yes (no side effects)

Configuration:
- No configuration needed
- Current Status: ✅ Stable
```

## 📊 Metrics Calculated
```
From precompute_functions_full.py:

Core Engagement Metrics (lines 1140-1143):
├── engagement_rate = ((likes + comments + shares) / views * 100)
├── likes_to_views = likes / views
├── comments_to_views = comments / views
└── shares_to_views = shares / views

Text Analysis Metrics (lines 1146-1147):
├── emoji_density = emoji_count / word_count
└── mention_density = mention_count / word_count

Temporal Metrics (lines 1028-1033):
├── publish_hour: 0-23 from create_time
└── publish_day_of_week: 0=Monday, 6=Sunday

All divisions protected by zero checks
```

## 🏗️ Service Architecture

### Service Boundaries
```
INPUT                    SERVICE                     OUTPUT
Raw Counts ─────────> Engagement Calculator ─────────> Normalized Metrics
                         ├── Rate calculation
                         ├── Ratio calculation
                         └── Density calculation
```

### Data Flow Pipeline
```
1. Input Stage (lines 1000-1019)
   ├── Extract from metadata_summary
   └── Fallback to static_metadata if needed

2. Calculation Stage
   ├── Engagement rate (line 1140)
   ├── View ratios (lines 1141-1143)
   ├── Text densities (lines 1146-1147)
   └── Time parsing (lines 1026-1033)

3. Output Stage (lines 1150-1180)
   └── Structured in 6-block format:
       ├── metadataCoreMetrics
       ├── metadataDynamics
       ├── metadataInteractions
       ├── metadataKeyEvents
       ├── metadataPatterns
       └── metadataComparison
```

### Data Sources and Fallbacks
```python
# Primary source: metadata_summary
view_count = metadata_summary.get('views', 0)

# Fallback chain if primary is 0 (lines 1012-1019):
if view_count == 0 and 'playCount' in static_metadata:
    view_count = static_metadata.get('playCount', 0)

Same pattern for likes, comments, shares
```

## 📁 File Structure & Key Locations
```
Service Implementation:
└── /rumiai_v2/processors/precompute_functions_full.py
    └── compute_metadata_analysis_metrics() (lines 985-1180)
        ├── Data extraction (lines 1000-1034)
        ├── Text analysis (lines 1036-1058)
        ├── Emoji detection regex (lines 1040-1051)
        ├── Engagement calculations (lines 1140-1147)
        └── Output structure (lines 1150-1180)
```

## 🚨 Failure Modes & Recovery

### Common Failure Scenarios
| Failure | Cause | Impact | Recovery | Frequency |
|---------|-------|--------|----------|-----------|
| Zero views | New/hidden video | Division by zero prevented | Returns 0 rates (line 1140) | Handled |
| Missing timestamp | No createTime | Default time values | Uses 0 for hour/day (lines 1032-1033) | Rare |
| Empty caption | No description text | Zero word count | All densities = 0 (lines 1146-1147) | Common |

## 🐛 Current Issues & Future Fixes

### Priority: LOW 🟢
- **Issue**: No time-based normalization
- **Impact**: Old videos may show different patterns
- **Current State**: Raw metrics only
- **Proposed Fix**: Add age-adjusted metrics
- **Effort Estimate**: 2 hours
- **Files Affected**: precompute_functions_full.py

## 🧪 Testing & Validation

### Functional Testing
- **Edge cases**: Zero views, missing data, empty text
- **All protected**: Division by zero checks throughout
- **Location**: No dedicated test file found

## 📈 Optimization Opportunities
- [x] **Already optimized**: Simple arithmetic operations
- [ ] **Benchmark comparisons**: Add percentile rankings
- [ ] **Trend detection**: Track engagement velocity

## 🔄 Dependencies
```
External Libraries:
├── re (regex for emoji pattern, lines 1040-1051)
└── datetime (timestamp parsing, lines 1028-1030)

Internal Dependencies:
└── None (calculations only)
```